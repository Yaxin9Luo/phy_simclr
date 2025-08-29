"""
Utility functions including the NT-Xent loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.
    """
    
    def __init__(self, temperature: float = 0.5, device: Optional[str] = None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate NT-Xent loss for a batch of projection vectors.
        
        Args:
            z: Tensor of shape (2N, D) where N is the batch size and D is the projection dimension.
               The first N vectors are z_i and the second N vectors are z_j (positive pairs).
        
        Returns:
            loss: The NT-Xent loss value
        """
        # Get batch size
        batch_size = z.shape[0] // 2
        
        # Normalize the projection vectors
        z = F.normalize(z, dim=1)
        
        # Calculate cosine similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # Shape: (2N, 2N)
        
        # Create mask to identify positive pairs
        # For each z_i, the positive is z_j at position i+N
        # For each z_j, the positive is z_i at position i-N
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool).to(self.device)
        mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)
        
        # Remove diagonal elements (self-similarity)
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        mask = mask & ~diagonal_mask
        
        # Get positive and negative similarities
        pos_sim = sim_matrix[mask].view(2 * batch_size, 1)
        
        # Create mask for negative samples (all samples except self and positive pair)
        neg_mask = ~mask & ~diagonal_mask
        
        # Calculate loss for each sample
        loss = 0
        for i in range(2 * batch_size):
            # Get positive similarity for this sample
            pos = pos_sim[i]
            
            # Get negative similarities for this sample
            neg = sim_matrix[i][neg_mask[i]]
            
            # Calculate log_sum_exp for numerical stability
            logits = torch.cat([pos, neg], dim=0)
            labels = torch.zeros(1, dtype=torch.long).to(self.device)  # Positive pair is at index 0
            
            loss += F.cross_entropy(logits.unsqueeze(0), labels)
        
        # Average loss over all samples
        loss = loss / (2 * batch_size)
        
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
