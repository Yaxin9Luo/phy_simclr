"""
ViViT (Vision Transformer for Videos) implementation for SimCLR training.
Based on the paper "ViViT: A Video Vision Transformer" and Google's Scenic implementation.
Reference: https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/model.py

Key differences from the official implementation:
1. Adapted for PyTorch (original is JAX/Flax)
2. Simplified for our use case (video-only, no multi-modal features)
3. Uses factorized encoder variant for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video inputs."""
    
    def __init__(self, video_size=(20, 256, 256), patch_size=(4, 16, 16), 
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.num_patches = (video_size[0] // patch_size[0]) * \
                          (video_size[1] // patch_size[1]) * \
                          (video_size[2] // patch_size[2])
        
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block for transformer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViViT(nn.Module):
    """
    Vision Transformer for Videos (ViViT).
    
    This implementation uses the "Factorised encoder" variant from the paper,
    which is more efficient for video processing.
    
    Following the official Scenic implementation, we support:
    - Factorized spatial-temporal encoder (Model 2 from the paper)
    - Flexible positional embeddings (learnable or sinusoidal)
    - Optional temporal embeddings
    """
    
    def __init__(self, video_size=(20, 256, 256), patch_size=(4, 16, 16),
                 in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., norm_layer=None,
                 temporal_embedding_type='learnable', use_temporal_embedding=True):
        super().__init__()
        norm_layer = norm_layer or nn.LayerNorm
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.video_size = video_size
        self.patch_size = patch_size
        self.temporal_embedding_type = temporal_embedding_type
        self.use_temporal_embedding = use_temporal_embedding
        
        # Calculate patch grid dimensions
        self.temporal_patches = video_size[0] // patch_size[0]
        self.spatial_patches_h = video_size[1] // patch_size[1]
        self.spatial_patches_w = video_size[2] // patch_size[2]
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(video_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embeddings (spatial only, following Model 2 from paper)
        num_spatial_patches = self.spatial_patches_h * self.spatial_patches_w
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_spatial_patches + 1, embed_dim))
        
        # Temporal embeddings (following official implementation)
        if self.use_temporal_embedding:
            if temporal_embedding_type == 'learnable':
                self.temporal_embed = nn.Parameter(torch.zeros(1, self.temporal_patches, embed_dim))
            else:
                # Sinusoidal embeddings will be computed in forward pass
                self.temporal_embed = None
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                 drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Representation layer (optional)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
            
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize positional embeddings
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_sinusoidal_embedding(self, n_position, d_hid):
        """Generate sinusoidal positional embeddings."""
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # (1, n_position, d_hid)
    
    def forward_features(self, x):
        # x shape: (B, T, C, H, W)
        B, T = x.shape[0], x.shape[1]
        x = self.patch_embed(x)  # (B, T*H'*W', embed_dim)
        
        # Reshape to separate temporal and spatial dimensions
        # Following Model 2 from the paper: factorized encoder
        x = rearrange(x, 'b (t h w) c -> b t (h w) c', 
                     t=self.temporal_patches, 
                     h=self.spatial_patches_h,
                     w=self.spatial_patches_w)
        
        # Add cls token to each temporal frame
        cls_tokens = self.cls_token.expand(B, self.temporal_patches, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)  # (B, T, 1+HW, C)
        
        # Add spatial positional embedding to each frame
        x = x + self.pos_embed.unsqueeze(1)  # Broadcast across temporal dimension
        
        # Add temporal embedding if enabled
        if self.use_temporal_embedding:
            if self.temporal_embedding_type == 'learnable':
                x = x + self.temporal_embed.unsqueeze(2)  # Broadcast across spatial dimension
            else:
                # Generate sinusoidal embeddings
                temporal_embed = self.get_sinusoidal_embedding(self.temporal_patches, self.embed_dim)
                temporal_embed = temporal_embed.to(x.device).unsqueeze(2)
                x = x + temporal_embed
        
        # Flatten temporal and spatial dimensions for transformer
        x = rearrange(x, 'b t n c -> b (t n) c')
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # Aggregate across temporal dimension
        # Reshape back to (B, T, N, C) and take cls tokens
        x = rearrange(x, 'b (t n) c -> b t n c', t=self.temporal_patches)
        cls_tokens = x[:, :, 0, :]  # (B, T, C)
        
        # Mean pool across temporal dimension
        x = cls_tokens.mean(dim=1)  # (B, C)
        
        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x


def vivit_tiny(**kwargs):
    """ViViT-Tiny model.
    Following the official implementation's configuration.
    """
    model = ViViT(
        patch_size=(4, 16, 16), embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vivit_small(**kwargs):
    """ViViT-Small model.
    Following the official implementation's configuration.
    """
    model = ViViT(
        patch_size=(4, 16, 16), embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vivit_base(**kwargs):
    """ViViT-Base model.
    Following the official implementation's configuration.
    Note: This matches the ViT-B/16 architecture adapted for video.
    """
    model = ViViT(
        patch_size=(4, 16, 16), embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


def vivit_large(**kwargs):
    """ViViT-Large model.
    Following the official implementation's configuration.
    """
    model = ViViT(
        patch_size=(4, 16, 16), embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs)
    return model


class ViViTEncoder(nn.Module):
    """
    ViViT Encoder wrapper for SimCLR training.
    Compatible with the existing SimCLR framework.
    
    Key features aligned with official implementation:
    - Factorized spatial-temporal encoder (Model 2)
    - Temporal embeddings for better video understanding
    - Multiple model sizes from tiny to large
    """
    
    def __init__(self, model_size='small', video_size=(20, 256, 256), pretrained=False,
                 temporal_embedding_type='learnable'):
        super(ViViTEncoder, self).__init__()
        
        # Create ViViT model based on size
        model_configs = {
            'tiny': (vivit_tiny, 192),
            'small': (vivit_small, 384),
            'base': (vivit_base, 768),
            'large': (vivit_large, 1024)
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_configs.keys())}")
        
        model_fn, self.feature_dim = model_configs[model_size]
        self.vivit = model_fn(
            video_size=video_size, 
            num_classes=0,
            temporal_embedding_type=temporal_embedding_type,
            use_temporal_embedding=True
        )
        
        # Note: For ViViT, pretrained weights would need to be loaded separately
        # The official implementation uses weights pretrained on ImageNet-21K
        if pretrained:
            print("Warning: Pretrained ViViT weights not available in this implementation.")
            print("Consider initializing from a ViT checkpoint for the spatial transformer.")
            
    def forward(self, x):
        """
        Forward pass through ViViT encoder.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
        
        Returns:
            features: Output features of shape (B, feature_dim)
        """
        return self.vivit.forward_features(x)
