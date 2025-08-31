"""
Utility functions including the NT-Xent loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import torch.distributed as dist

# All-Gather functionality removed for performance optimization

class NTXentLoss(nn.Module):
    """
    Local NT-Xent loss (no all_gather). Expects z = [xi; xj] for the *local* batch.
    """
    def __init__(self, temperature: float = 0.5, device: Optional[torch.device] = None):
        super().__init__()
        self.temperature = float(temperature)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.device = device

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (2 * B_local, D); assume L2-normalized upstream
        device = z.device
        n = z.size(0)
        assert n % 2 == 0 and n >= 4, "Local batch must be even and >= 4 (needs negatives)."

        # Sanitize inputs (keep graph intact)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = F.normalize(z.float(), p=2, dim=1)

        # Similarity matrix
        sim = (z @ z.T) / self.temperature  # (n, n)

        # Positive index mapping within the local batch
        half = n // 2
        idx  = torch.arange(n, device=device)
        pos  = torch.empty_like(idx)
        pos[:half] = idx[:half] + half
        pos[half:] = idx[half:] - half

        # Build logits: [pos | negatives]
        pos_logits = sim[idx, pos].unsqueeze(1)  # (n, 1)

        mask = torch.ones((n, n), dtype=torch.bool, device=device)
        mask[idx, idx] = False
        mask[idx, pos] = False
        neg_logits = sim[mask].view(n, -1)       # (n, n-2)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        # Stabilize numerics without breaking grads
        logits = torch.nan_to_num(logits, neginf=-50.0, posinf=50.0).clamp_(-50.0, 50.0)

        labels = torch.zeros(n, dtype=torch.long, device=device)
        loss = self.criterion(logits, labels)

        # Final safety: if non-finite, replace by a zero-like loss attached to graph
        if not torch.isfinite(loss):
            loss = (z.sum() * 0.0)  # keeps graph, zero gradient step
        return loss


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, filepath: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    filepath: str, device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
