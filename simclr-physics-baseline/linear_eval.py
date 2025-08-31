"""
Linear evaluation script for SimCLR pretrained models.
This script loads a pretrained SimCLR model and trains a linear classifier
on top of frozen features to evaluate the quality of learned representations.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import time
from pathlib import Path

from model import SimCLRModel
from config import Config
from utils import set_seed


class LinearClassifier(nn.Module):
    """Linear classifier head for evaluation."""
    
    def __init__(self, input_dim, num_classes=1000):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


class SimCLRLinearEval(nn.Module):
    """SimCLR model with frozen encoder and linear classifier head."""
    
    def __init__(self, simclr_model, num_classes=1000):
        super(SimCLRLinearEval, self).__init__()
        
        # Extract the encoder (backbone) from SimCLR model
        self.encoder = simclr_model.encoder
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Get feature dimension from encoder
        if hasattr(simclr_model.encoder, 'embed_dim'):
            feature_dim = simclr_model.encoder.embed_dim
        elif hasattr(simclr_model.encoder, 'num_features'):
            feature_dim = simclr_model.encoder.num_features
        else:
            # For ViT models, typically 768 for base
            feature_dim = 768
            
        # Add linear classifier
        self.classifier = LinearClassifier(feature_dim, num_classes)
        
    def forward(self, x):
        # Extract features with frozen encoder
        with torch.no_grad():
            features = self.encoder(x)
            
        # Classify with trainable linear head
        return self.classifier(features)


def get_imagenet_dataloaders(data_dir, batch_size=256, num_workers=4):
    """Create ImageNet train/val dataloaders for linear evaluation."""
    
    # Standard ImageNet preprocessing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def load_simclr_checkpoint(checkpoint_path, config):
    """Load SimCLR model from checkpoint."""
    
    # Create SimCLR model
    model = SimCLRModel(
        base_model=config.base_model,
        projection_dim=config.projection_dim,
        img_size=config.img_size,
        pretrained=False,  # We're loading from checkpoint
        use_flash_attn=config.use_flash_attn
    )
    
    # Load checkpoint using DeepSpeed's zero_to_fp32 utility
    if os.path.exists(os.path.join(checkpoint_path, 'zero_to_fp32.py')):
        # Use DeepSpeed's utility to convert checkpoint
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        # Try loading directly
        checkpoint_file = os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_path}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{acc:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            avg_loss = total_loss / len(pbar)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.2f}%'
            })
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Linear evaluation of SimCLR pretrained models')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to SimCLR checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='/home/yaxin/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=90,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='linear_eval_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config (use default config for model architecture)
    config = Config()
    
    # Load pretrained SimCLR model
    print(f"Loading SimCLR checkpoint from {args.checkpoint_path}")
    simclr_model = load_simclr_checkpoint(args.checkpoint_path, config)
    
    # Create linear evaluation model
    model = SimCLRLinearEval(simclr_model, num_classes=1000)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Create dataloaders
    print("Loading ImageNet dataset...")
    train_loader, val_loader = get_imagenet_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.classifier.parameters(),  # Only train classifier
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    results = []
    
    print(f"\nStarting linear evaluation training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        # Log results
        result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }
        results.append(result)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc: {best_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
    
    # Save final results
    total_time = time.time() - start_time
    
    final_results = {
        'best_val_acc': best_acc,
        'final_val_acc': val_acc,
        'total_time': total_time,
        'args': vars(args),
        'epoch_results': results
    }
    
    torch.save(final_results, os.path.join(args.save_dir, 'results.pth'))
    
    print(f"\nLinear evaluation completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
