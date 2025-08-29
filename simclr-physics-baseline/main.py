"""
Main training script for SimCLR with DDP support.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import CLEVRERDataset, preprocess_clevrer_dataset
from model import SimCLRModel
from utils import NTXentLoss, save_checkpoint, load_checkpoint, set_seed


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP environment."""
    dist.destroy_process_group()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, rank=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
    else:
        progress_bar = dataloader
    
    for batch_idx, (clip_i, clip_j) in enumerate(progress_bar):
        # Move data to device
        clip_i = clip_i.to(device)  # Shape: (B, T, C, H, W)
        clip_j = clip_j.to(device)
        
        # Concatenate positive pairs
        clips = torch.cat([clip_i, clip_j], dim=0)  # Shape: (2B, T, C, H, W)
        
        # Forward pass
        projections = model(clips)  # Shape: (2B, projection_dim)
        
        # Calculate loss
        loss = criterion(projections)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar (only on rank 0)
        if rank == 0 and hasattr(progress_bar, 'set_postfix'):
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
        
        # Log at intervals (only on rank 0)
        if rank == 0 and batch_idx % config.log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def train_ddp(rank, world_size, args, config):
    """Training function for each DDP process."""
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Only rank 0 handles preprocessing
    if rank == 0:
        # Preprocess CLEVRER dataset if requested
        if args.preprocess:
            if args.clevrer_root is None:
                print("Error: --clevrer_root must be specified when using --preprocess")
                cleanup_ddp()
                return
            print(f"Preprocessing CLEVRER dataset...")
            preprocess_clevrer_dataset(
                args.clevrer_root, 
                args.processed_dir,
                split='train',
                max_videos=args.max_videos
            )
    
    # Synchronize all processes
    dist.barrier()
    
    # Check if processed dataset exists
    metadata_file = os.path.join(args.processed_dir, 'train_metadata.json')
    if not os.path.exists(metadata_file):
        if rank == 0:
            print(f"Processed dataset not found at {args.processed_dir}.")
            print("Please preprocess the CLEVRER dataset first using --preprocess flag.")
            print("Example: python main.py --preprocess --clevrer_root /path/to/clevrer")
        cleanup_ddp()
        return
    
    # Create dataset
    if rank == 0:
        print("Loading dataset...")
    dataset = CLEVRERDataset(
        args.processed_dir, 
        metadata_file,
        num_frames=config.num_frames,
        frame_size=config.box_size
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create dataloader with distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size // world_size,  # Divide batch size by number of GPUs
        shuffle=False,  # Shuffling is handled by DistributedSampler
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True  # Important for DDP to ensure equal batch sizes
    )
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = SimCLRModel(
        base_model=config.base_model, 
        projection_dim=config.projection_dim,
        video_size=(config.num_frames, config.box_size, config.box_size)
    )
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create loss function
    criterion = NTXentLoss(temperature=config.temperature, device=device)
    
    # Create optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate * world_size,  # Scale learning rate
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'lars':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate * world_size,  # Scale learning rate
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume and rank == 0:
        # Load checkpoint only on rank 0, then broadcast
        start_epoch, _ = load_checkpoint(model.module, optimizer, args.resume, device=device)
        start_epoch += 1
    
    # Synchronize start epoch across all processes
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()
    
    # Training loop
    if rank == 0:
        print(f"\nStarting distributed training on {world_size} GPUs...")
        print(f"Config: {config}")
        print(f"Per-GPU batch size: {config.batch_size // world_size}")
        print(f"Total batch size: {config.batch_size}")
        print(f"Number of batches per epoch: {len(dataloader)}")
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        # Train for one epoch
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, rank)
        
        # Gather average loss from all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = avg_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss_global:.4f}")
        
        # Save checkpoint (only on rank 0)
        if rank == 0 and epoch % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(model.module, optimizer, epoch, avg_loss_global, checkpoint_path)
    
    # Save final model (only on rank 0)
    if rank == 0:
        final_checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
        save_checkpoint(model.module, optimizer, config.num_epochs, avg_loss_global, final_checkpoint_path)
        print("\nTraining completed!")
    
    # Clean up
    cleanup_ddp()


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Training with DDP Support')
    parser.add_argument('--batch_size', type=int, default=None, help='Total batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--base_model', type=str, default=None, 
                       help='Base model architecture (resnet18, resnet50, vivit_tiny, vivit_small, vivit_base)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess CLEVRER dataset before training')
    parser.add_argument('--clevrer_root', type=str, default=None, help='Path to CLEVRER dataset root')
    parser.add_argument('--processed_dir', type=str, default='processed_clevrer', help='Directory for processed frames')
    parser.add_argument('--max_videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--ddp', action='store_true', help='Use Distributed Data Parallel training')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use for DDP (default: all available)')
    
    args = parser.parse_args()
    
    # Create config object
    config_kwargs = {}
    if args.batch_size is not None:
        config_kwargs['batch_size'] = args.batch_size
    if args.epochs is not None:
        config_kwargs['num_epochs'] = args.epochs
    if args.lr is not None:
        config_kwargs['learning_rate'] = args.lr
    if args.temperature is not None:
        config_kwargs['temperature'] = args.temperature
    if args.data_dir is not None:
        config_kwargs['data_dir'] = args.data_dir
    if args.checkpoint_dir is not None:
        config_kwargs['checkpoint_dir'] = args.checkpoint_dir
    if args.processed_dir is not None:
        config_kwargs['data_dir'] = args.processed_dir
    if args.base_model is not None:
        config_kwargs['base_model'] = args.base_model
    
    config = Config(**config_kwargs)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Check if using DDP
    if args.ddp:
        # Determine number of GPUs
        if args.num_gpus is None:
            world_size = torch.cuda.device_count()
        else:
            world_size = min(args.num_gpus, torch.cuda.device_count())
        
        if world_size < 2:
            print("DDP requires at least 2 GPUs. Falling back to single GPU training.")
            args.ddp = False
        else:
            print(f"Starting DDP training with {world_size} GPUs...")
            # Spawn processes for DDP
            mp.spawn(train_ddp, args=(world_size, args, config), nprocs=world_size, join=True)
            return
    
    # Single GPU training (original code)
    if not args.ddp:
        # Preprocess CLEVRER dataset if requested
        if args.preprocess:
            if args.clevrer_root is None:
                print("Error: --clevrer_root must be specified when using --preprocess")
                return
            print(f"Preprocessing CLEVRER dataset...")
            preprocess_clevrer_dataset(
                args.clevrer_root, 
                args.processed_dir,
                split='train',
                max_videos=args.max_videos
            )
        
        # Check if processed dataset exists
        metadata_file = os.path.join(args.processed_dir, 'train_metadata.json')
        if not os.path.exists(metadata_file):
            print(f"Processed dataset not found at {args.processed_dir}.")
            print("Please preprocess the CLEVRER dataset first using --preprocess flag.")
            print("Example: python main.py --preprocess --clevrer_root /path/to/clevrer")
            return
        
        # Create dataset and dataloader
        print("Loading dataset...")
        dataset = CLEVRERDataset(
            args.processed_dir, 
            metadata_file,
            num_frames=config.num_frames,
            frame_size=config.box_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if config.device == 'cuda' else False
        )
        
        # Create model
        print("Creating model...")
        model = SimCLRModel(
            base_model=config.base_model,
            projection_dim=config.projection_dim,
            video_size=(config.num_frames, config.box_size, config.box_size)
        )
        device = torch.device(config.device)
        model = model.to(device)
        
        # Create loss function
        criterion = NTXentLoss(temperature=config.temperature, device=config.device)
        
        # Create optimizer
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'lars':
            # LARS optimizer would need to be implemented or imported
            # For now, fall back to Adam with momentum-like behavior
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Resume from checkpoint if specified
        start_epoch = 1
        if args.resume:
            start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device=config.device)
            start_epoch += 1
        
        # Training loop
        print(f"\nStarting training on {device}...")
        print(f"Config: {config}")
        print(f"Number of batches per epoch: {len(dataloader)}")
        
        for epoch in range(start_epoch, config.num_epochs + 1):
            # Train for one epoch
            avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, config)
            
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # Save final model
        final_checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
        save_checkpoint(model, optimizer, config.num_epochs, avg_loss, final_checkpoint_path)
        
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
