"""
Temporal encoder and VideoSimCLR model that leverage existing image Encoder
and ProjectionHead from model.py. Processes video clips as sequences of frames.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn

from model import Encoder, ProjectionHead


class PositionalEncoding1D(nn.Module):
    """Standard sine-cosine positional encoding over time dimension.

    Produces (T, D) pe that can be added to sequence features.
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)


class TemporalEncoder(nn.Module):
    """
    Apply the image Encoder per frame, then aggregate temporally.

    Inputs:
      x: (B, T, C, H, W)
    Outputs:
      h: (B, D) aggregated features
    """
    def __init__(
        self,
        base_model: str = "vit_base",
        img_size: int = 224,
        pretrained: bool = True,
        vit_representation: Literal["cls", "mean"] = "cls",
        use_flash_attn: bool = True,
        temporal_agg: Literal["mean", "transformer"] = "transformer",
        transformer_hidden: Optional[int] = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frame_encoder = Encoder(
            base_model=base_model,
            pretrained=pretrained,
            img_size=img_size,
            vit_representation=vit_representation,
            use_flash_attn=use_flash_attn,
        )
        self.feature_dim = self.frame_encoder.feature_dim
        self.temporal_agg = temporal_agg

        if temporal_agg == "mean":
            self.temporal = None
        elif temporal_agg == "transformer":
            d_model = self.feature_dim if transformer_hidden is None else transformer_hidden
            self.proj = nn.Identity() if d_model == self.feature_dim else nn.Linear(self.feature_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                                       dropout=dropout, batch_first=False, norm_first=True)
            self.posenc = PositionalEncoding1D(d_model)
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.readout = nn.Linear(d_model, self.feature_dim) if d_model != self.feature_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown temporal_agg: {temporal_agg}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        h = self.frame_encoder(x)            # (B*T, D)
        h = h.view(B, T, -1)                # (B, T, D)

        if self.temporal_agg == "mean":
            h_agg = h.mean(dim=1)           # (B, D)
        else:
            # transformer expects (T, B, D)
            h_seq = h.transpose(0, 1)       # (T, B, D or d_model)
            if hasattr(self, 'proj'):
                h_seq = self.proj(h_seq)
            h_seq = self.posenc(h_seq)
            h_enc = self.temporal(h_seq)    # (T, B, d_model)
            # simple CLS-free readout: mean pool over time then (optional) map back to D
            h_agg = h_enc.mean(dim=0)       # (B, d_model)
            h_agg = self.readout(h_agg)     # (B, D)
        return h_agg


class VideoSimCLRModel(nn.Module):
    """Video SimCLR model = TemporalEncoder + ProjectionHead (L2-normalized)."""
    def __init__(
        self,
        base_model: str = "vit_base",
        projection_dim: int = 512,
        img_size: int = 224,
        pretrained: bool = True,
        vit_representation: Literal["cls", "mean"] = "cls",
        proj_hidden_dim: int = 2048,
        use_bn_out: bool = True,
        use_flash_attn: bool = True,
        temporal_agg: Literal["mean", "transformer"] = "transformer",
    ):
        super().__init__()
        self.encoder = TemporalEncoder(
            base_model=base_model,
            img_size=img_size,
            pretrained=pretrained,
            vit_representation=vit_representation,
            use_flash_attn=use_flash_attn,
            temporal_agg=temporal_agg,
        )
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=projection_dim,
            use_bn_out=use_bn_out,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x: (B, T, C, H, W)
        h = self.encoder(x)
        if return_features:
            return h
        z = self.projection_head(h)
        return z

