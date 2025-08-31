"""
Model architecture for SimCLR: Encoder and Projection Head (ViT or ResNet).
- ViT uses last_hidden_state (CLS or mean pooling), not pooler_output.
- Projection head follows SimCLR: 2-layer MLP with BN and ReLU; L2-norm on output.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

# torchvision imports guarded for older/newer versions
try:
    import torchvision
    from torchvision.models import resnet18, resnet34, resnet50
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

from transformers import ViTModel, ViTConfig

# Check PyTorch version for Flash Attention support via SDPA
_HAS_SDPA = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if _HAS_SDPA:
    print("PyTorch SDPA detected - Flash Attention will be used automatically when available")
else:
    print("PyTorch version does not support SDPA - using standard attention")


# -----------------------------
# Utilities
# -----------------------------
class L2Normalize(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + self.eps)


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    """
    Image encoder that can use either ResNet (CNN) or ViT (Transformer).
    For ViT, we avoid `pooler_output` (tanh) and use CLS or mean of patch tokens.
    """

    def __init__(
        self,
        base_model: str = "resnet18",
        pretrained: bool = True,
        img_size: int = 224,
        vit_representation: Literal["cls", "mean"] = "cls",
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.base_model_name = base_model
        self.img_size = img_size
        self.is_vit = base_model.startswith("vit")
        self.vit_representation = vit_representation
        self.use_flash_attn = use_flash_attn

        if self.is_vit:
            # Determine size (vit_tiny, vit_small, vit_base, vit_large, or "vit"->base)
            parts = base_model.split("_")
            model_size = parts[1] if len(parts) > 1 else "base"

            model_mapping = {
                "tiny": "WinKawaks/vit-tiny-patch16-224",
                "small": "WinKawaks/vit-small-patch16-224",
                "base": "google/vit-base-patch16-224",
                "large": "google/vit-large-patch16-224",
            }
            if model_size not in model_mapping:
                raise ValueError(f"Unsupported ViT model size: {model_size}")
            model_name = model_mapping[model_size]

            if pretrained:
                # Load the pretrained backbone; DO NOT use pooler_output.
                self.encoder = ViTModel.from_pretrained(model_name)
                # Ensure you feed the model with its native image_size to keep pos-embeds consistent
                native_size = getattr(self.encoder.config, "image_size", 224)
                if img_size != native_size:
                    raise ValueError(
                        f"Pretrained ViT expects image_size={native_size}, but got img_size={img_size}. "
                        "Please set img_size to the native size or re-init from config (pretrained=False)."
                    )
                
                # Enable SDPA (PyTorch's scaled_dot_product_attention) for Flash Attention
                if self.use_flash_attn and _HAS_SDPA:
                    self.encoder.config._attn_implementation = "sdpa"
                    print(f"Using pretrained ViT encoder with SDPA (Flash Attention): {model_name} (image_size={native_size})")
                else:
                    print(f"Using pretrained ViT encoder: {model_name} (image_size={native_size})")
            else:
                # Random init with a config whose image_size matches your pipeline
                config = ViTConfig.from_pretrained(model_name)
                config.image_size = img_size
                
                # Enable SDPA (PyTorch's scaled_dot_product_attention) for Flash Attention
                if self.use_flash_attn and _HAS_SDPA:
                    config._attn_implementation = "sdpa"
                    print(f"Using ViT encoder with SDPA (Flash Attention) and random initialization: {model_name}, image_size={img_size}")
                else:
                    print(f"Using ViT encoder with random initialization: {model_name}, image_size={img_size}")
                
                # keep other config fields (hidden_size, patch_size, etc.) the same
                self.encoder = ViTModel(config)

            self.feature_dim = self.encoder.config.hidden_size

        else:
            if not _HAS_TORCHVISION:
                raise ImportError("torchvision is required for ResNet encoders.")

            # Handle legacy vs new API for pretrained weights
            def _load_resnet(which: str, pretrained_flag: bool):
                if hasattr(torchvision.models, "ResNet18_Weights"):
                    weights = None
                    if pretrained_flag:
                        weights_map = {
                            "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                            "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
                            "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
                        }
                        weights = weights_map[which]
                    return getattr(torchvision.models, which)(weights=weights)
                else:
                    # older torchvision
                    return getattr(torchvision.models, which)(pretrained=pretrained_flag)

            if base_model not in {"resnet18", "resnet34", "resnet50"}:
                raise ValueError(f"Unsupported base model: {base_model}")

            resnet = _load_resnet(base_model, pretrained)
            self.feature_dim = resnet.fc.in_features
            # remove avgpool+fc head, keep up to penultimate
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, C, 1, 1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            print(f"Using ResNet encoder with model size: {base_model} (pretrained={pretrained})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features h: (B, feature_dim)
        """
        if self.is_vit:
            # ViTModel expects pixel_values normalized appropriately by your pipeline.
            outputs = self.encoder(pixel_values=x)
            # Prefer last_hidden_state; choose CLS or mean pooling
            tokens = outputs.last_hidden_state  # (B, 1+num_patches, D)
            if self.vit_representation == "cls":
                h = tokens[:, 0]  # CLS
            else:  # "mean"
                # exclude CLS (index 0) for mean of patches
                h = tokens[:, 1:].mean(dim=1)
        else:
            feats = self.backbone(x)                 # (B, C, 1, 1) typically
            feats = self.avgpool(feats)              # safety if spatial >1x1
            h = feats.flatten(1)                     # (B, feature_dim)

        return h  # encoder features (pre-projection)


# -----------------------------
# Projection Head (SimCLR)
# -----------------------------
class ProjectionHead(nn.Module):
    """
    SimCLR projection head:
      z = MLP(h) with 2 Linear layers, BN on hidden AND output, ReLU between.
      Final z is L2-normalized for cosine similarity.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 512, use_bn_out: bool = True):
        super().__init__()
        # As in SimCLR, biases can be disabled when using BN
        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        ]
        if use_bn_out:
            layers.append(nn.BatchNorm1d(output_dim, affine=True))
        self.net = nn.Sequential(*layers)
        self.l2 = L2Normalize()

        # Initialization (optional; PyTorch defaults are fine, but we can do this)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)          # (B, output_dim)
        z = self.l2(z)           # L2-normalize for cosine similarity
        return z


# -----------------------------
# SimCLR Model
# -----------------------------
class SimCLRModel(nn.Module):
    """
    Encoder + ProjectionHead for SimCLR. Forward returns normalized z by default.
    Set return_features=True to get encoder features h instead of z.
    """

    def __init__(
        self,
        base_model: str = "resnet18",
        projection_dim: int = 512,
        img_size: int = 224,
        pretrained: bool = True,
        vit_representation: Literal["cls", "mean"] = "cls",
        proj_hidden_dim: int = 2048,
        use_bn_out: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            base_model=base_model,
            pretrained=pretrained,
            img_size=img_size,
            vit_representation=vit_representation,
            use_flash_attn=use_flash_attn,
        )
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=projection_dim,
            use_bn_out=use_bn_out,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        h = self.encoder(x)  # (B, D)
        if return_features:
            return h
        z = self.projection_head(h)  # (B, projection_dim), L2-normalized
        return z