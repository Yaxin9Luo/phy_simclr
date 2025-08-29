"""
Model architecture for SimCLR: Encoder and Projection Head.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from vivit_model import ViViTEncoder


class Encoder(nn.Module):
    """
    Video encoder that can use either ResNet (CNN) or ViViT (Transformer) architectures.
    """
    
    def __init__(self, base_model: str = 'resnet18', pretrained: bool = True, 
                 video_size: tuple = (20, 256, 256)):
        super(Encoder, self).__init__()
        
        self.base_model_name = base_model
        
        # Check if using ViViT
        if base_model.startswith('vivit'):
            # Extract model size (vivit_tiny, vivit_small, vivit_base)
            model_size = base_model.split('_')[1] if '_' in base_model else 'small'
            self.encoder = ViViTEncoder(model_size=model_size, video_size=video_size, pretrained=pretrained)
            self.feature_dim = self.encoder.feature_dim
            self.is_vivit = True
        else:
            # Use ResNet for CNN-based encoding
            self.is_vivit = False
            if base_model == 'resnet18':
                resnet = models.resnet18(pretrained=pretrained)
            elif base_model == 'resnet34':
                resnet = models.resnet34(pretrained=pretrained)
            elif base_model == 'resnet50':
                resnet = models.resnet50(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported base model: {base_model}")
            
            # Remove the final fully connected layer
            self.feature_dim = resnet.fc.in_features
            self.base_model = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
               B: batch size, T: number of frames, C: channels, H: height, W: width
        
        Returns:
            features: Aggregated features of shape (B, feature_dim)
        """
        if self.is_vivit:
            # ViViT directly processes video sequences
            features = self.encoder(x)  # Shape: (B, feature_dim)
        else:
            # ResNet processing (original implementation)
            B, T, C, H, W = x.shape
            
            # Reshape to process all frames at once
            x = x.view(B * T, C, H, W)
            
            # Pass through ResNet
            features = self.base_model(x)  # Shape: (B * T, feature_dim, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # Shape: (B * T, feature_dim)
            
            # Reshape back to separate batch and time dimensions
            features = features.view(B, T, self.feature_dim)
            
            # Aggregate features across time dimension (mean pooling)
            features = torch.mean(features, dim=1)  # Shape: (B, feature_dim)
        
        return features


class ProjectionHead(nn.Module):
    """
    MLP projection head for SimCLR.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(ProjectionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x: Input features of shape (B, input_dim)
        
        Returns:
            projections: Output projections of shape (B, output_dim)
        """
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model combining encoder and projection head.
    """
    
    def __init__(self, base_model: str = 'resnet18', projection_dim: int = 128,
                 video_size: tuple = (20, 256, 256)):
        super(SimCLRModel, self).__init__()
        
        # Initialize encoder
        self.encoder = Encoder(base_model=base_model, video_size=video_size)
        
        # Initialize projection head
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=256,
            output_dim=projection_dim
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            return_features: If True, return encoder features instead of projections
        
        Returns:
            Output tensor of shape (B, projection_dim) or (B, feature_dim)
        """
        # Get features from encoder
        features = self.encoder(x)
        
        if return_features:
            return features
        
        # Get projections from projection head
        projections = self.projection_head(features)
        
        return projections
