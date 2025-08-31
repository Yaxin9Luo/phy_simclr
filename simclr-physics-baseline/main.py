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

from config import Config
from dataset import ImageNetSimCLR
from model import SimCLRModel
from utils import NTXentLoss, save_checkpoint, load_checkpoint, set_seed

# DeepSpeed imports
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    _HAS_DEEPSPEED = True
    print("DeepSpeed available")
except ImportError:
    _HAS_DEEPSPEED = False
    print("DeepSpeed not available, using standard DDP")

# Optional: better perf on Ampere+
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Removed all_gather imports to eliminate communication overhead


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Use available port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP environment."""
    dist.destroy_process_group()


def maybe_cosine_with_warmup(optimizer, num_epochs, steps_per_epoch, warmup_ratio=0.1):
    """Cosine LR with warmup; returns scheduler, total_steps."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay from 1 → 0 over remaining steps
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, total_steps


# Removed _all_gather_concat function to eliminate communication overhead


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, rank=0, scaler=None, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}') if rank == 0 else dataloader

    for batch_idx, (img_i, img_j, _) in enumerate(iterator):
        img_i = img_i.to(device, non_blocking=True)
        img_j = img_j.to(device, non_blocking=True)

        imgs = torch.cat([img_i, img_j], dim=0)  # (2B, C, H, W)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=getattr(config, "amp", True)):
            projections = model(imgs)  # (2B, D), already L2-normalized in our model

            # Use local negatives only for optimal performance
            loss = criterion(projections)

        if scaler is not None:
            scaler.scale(loss).backward()
            if getattr(config, 'gradient_clip_norm', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if getattr(config, 'gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0 and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{(total_loss/num_batches):.4f}'})

        if rank == 0 and batch_idx % config.log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / max(1, num_batches)


def train_deepspeed(args, config):
    """Training function using DeepSpeed for ZeRO-1 optimization."""
    # Set environment variables for distributed training if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    # DeepSpeed handles distributed initialization
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Check if ImageNet dataset exists
    if not os.path.exists(config.data_dir):
        if local_rank == 0:
            print(f"ImageNet dataset not found at {config.data_dir}.")
            print("Please ensure ImageNet dataset is available at the specified path.")
            print("Expected structure: <data_dir>/train/, <data_dir>/val/")
        return
    
    # Create dataset
    if local_rank == 0:
        print("Loading dataset...")
    dataset = ImageNetSimCLR(
        root=config.data_dir,
        split='train',
        img_size=config.img_size
    )
    
    # Create dataloader - DeepSpeed will handle the sampler
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size // torch.distributed.get_world_size(),
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=4
    )
    
    if local_rank == 0:
        print("Creating model...")
    
    model = SimCLRModel(
        base_model=config.base_model,
        projection_dim=config.projection_dim,
        img_size=config.img_size,
        pretrained=args.pretrained,
        use_flash_attn=config.use_flash_attn
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=dataset,
        config=args.deepspeed_config
    )
    
    # Use the optimized NTXentLoss (no all_gather)
    criterion = NTXentLoss(temperature=config.temperature, device=device)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        # All ranks need to load the checkpoint
        model_engine.load_checkpoint(args.resume)
        start_epoch = model_engine.global_steps // len(dataloader) + 1
        if local_rank == 0:
            print(f"Resuming from checkpoint at epoch {start_epoch}, global step {model_engine.global_steps}")
    
    if local_rank == 0:
        print(f"\nStarting DeepSpeed training with ZeRO-2...")
        print(f"Config: {config}")
        print(f"Per-GPU batch size: {config.batch_size // torch.distributed.get_world_size()}")
        print(f"Total batch size: {config.batch_size}")
        print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        model_engine.train()
        total_loss = 0.0
        num_batches = 0
        
        # Only show progress bar on local rank 0
        if local_rank == 0:
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')
        else:
            progress_bar = dataloader
        
        for batch_idx, (img_i, img_j, _) in enumerate(progress_bar):
            # Move data to device
            img_i = img_i.to(device, non_blocking=True)
            img_j = img_j.to(device, non_blocking=True)
            
            imgs = torch.cat([img_i, img_j], dim=0)  # (2B, C, H, W)
            
            # Forward pass
            projections = model_engine(imgs)
            # Use local negatives only to avoid all_gather overhead
            loss = criterion(projections)
            
            # DeepSpeed handles gradient accumulation automatically
            model_engine.backward(loss)
            model_engine.step()  # DeepSpeed will only step when accumulation is complete
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar (only on local rank 0)
            if local_rank == 0 and hasattr(progress_bar, 'set_postfix'):
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
            
            # Log at intervals (only on local rank 0)
            if local_rank == 0 and batch_idx % config.log_interval == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        
        if local_rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint (DeepSpeed handles this automatically)
        if epoch % config.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(config.checkpoint_dir, f'epoch_{epoch}')
            model_engine.save_checkpoint(checkpoint_dir)
    
    # Save final model
    if local_rank == 0:
        final_checkpoint_dir = os.path.join(config.checkpoint_dir, 'final_model')
        model_engine.save_checkpoint(final_checkpoint_dir)
        print("\nDeepSpeed training completed!")


def train_ddp(rank, world_size, args, config):
    """Training function for each DDP process."""
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Important for different shuffles per rank/epoch
    set_seed(config.seed + rank)

    # Check dataset path
    if not os.path.exists(config.data_dir):
        if rank == 0:
            print(f"ImageNet dataset not found at {config.data_dir}.")
            print("Expected structure: <data_dir>/train/, <data_dir>/val/")
        cleanup_ddp()
        return

    if rank == 0:
        print("Loading dataset...")

    dataset = ImageNetSimCLR(root=config.data_dir, split='train', img_size=config.img_size)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size // world_size,  # total batch = config.batch_size
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(config.num_workers > 0),
    )

    if rank == 0:
        print("Creating model...")

    model = SimCLRModel(
        base_model=config.base_model,
        projection_dim=config.projection_dim,
        img_size=config.img_size,
        pretrained=args.pretrained,
        use_flash_attn=config.use_flash_attn
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Use the optimized NTXentLoss (no all_gather)
    criterion = NTXentLoss(temperature=config.temperature, device=device)

    # ---- Optimizer
    # Keep your linear LR scaling assumption (learning_rate is per-GPU base).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate * world_size,  # assumes config.learning_rate is per-GPU base LR
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # ---- Scheduler (cosine with warmup)
    steps_per_epoch = len(dataloader)
    scheduler, _ = maybe_cosine_with_warmup(optimizer, config.num_epochs, steps_per_epoch, warmup_ratio=0.1)

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=getattr(config, "amp", True))

    # ---- Resume
    start_epoch = 1
    if args.resume and rank == 0:
        start_epoch, _ = load_checkpoint(model.module, optimizer, args.resume, device=device)
        start_epoch += 1
    # broadcast start epoch
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())

    if rank == 0:
        print(f"\nStarting distributed training on {world_size} GPUs...")
        print(f"Config: {config}")
        print(f"Per-GPU batch size: {config.batch_size // world_size}")
        print(f"Total batch size: {config.batch_size}")
        print(f"Number of batches per epoch: {len(dataloader)}")

    for epoch in range(start_epoch, config.num_epochs + 1):
        sampler.set_epoch(epoch)  # shuffle seed
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, rank, scaler, scheduler)

        # Reduce loss across ranks
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_global = (avg_loss_tensor.item() / world_size)

        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {avg_loss_global:.4f}")

        if rank == 0 and epoch % config.checkpoint_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model.module, optimizer, epoch, avg_loss_global, checkpoint_path)

    if rank == 0:
        final_checkpoint_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
        save_checkpoint(model.module, optimizer, config.num_epochs, avg_loss_global, final_checkpoint_path)
        print("\nTraining completed!")

    cleanup_ddp()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='SimCLR Training with DDP Support')
    parser.add_argument('--batch_size', type=int, default=None, help='Total batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (per-GPU if using DDP scaling)')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for NT-Xent loss')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model architecture (resnet18, resnet50, vit_tiny, vit_small, vit_base, vit_large)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights for ViT (default: True)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Use scratch initialization instead of pretrained weights')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--ddp', action='store_true', help='Use Distributed Data Parallel training')
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed training with ZeRO-2')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json', help='DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training (automatically set by DeepSpeed)')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use for DDP (default: all available)')
    args = parser.parse_args()

    if args.no_pretrained:
        args.pretrained = False

    # Build config from CLI overrides
    kwargs = {}
    if args.batch_size is not None: kwargs['batch_size'] = args.batch_size
    if args.epochs is not None: kwargs['num_epochs'] = args.epochs
    if args.lr is not None: kwargs['learning_rate'] = args.lr
    if args.temperature is not None: kwargs['temperature'] = args.temperature
    if args.data_dir is not None: kwargs['data_dir'] = args.data_dir
    if args.checkpoint_dir is not None: kwargs['checkpoint_dir'] = args.checkpoint_dir
    if args.base_model is not None: kwargs['base_model'] = args.base_model

    config = Config(**kwargs)
    set_seed(config.seed)  # base seed; DDP adds rank offset inside train_ddp

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Check if using DeepSpeed
    if args.deepspeed and _HAS_DEEPSPEED:
        print("Starting DeepSpeed training with ZeRO-1...")
        train_deepspeed(args, config)
        return
    elif args.deepspeed and not _HAS_DEEPSPEED:
        print("DeepSpeed requested but not available. Falling back to DDP.")
        args.ddp = True

    if args.ddp:
        world_size = torch.cuda.device_count() if args.num_gpus is None else min(args.num_gpus, torch.cuda.device_count())
        if world_size < 2:
            print("DDP requires at least 2 GPUs. Falling back to single GPU training.")
        else:
            print(f"Starting DDP training with {world_size} GPUs...")
            mp.spawn(train_ddp, args=(world_size, args, config), nprocs=world_size, join=True)
            return  # DDP done

    # ----- Single GPU / single-process
    if not os.path.exists(config.data_dir):
        print(f"ImageNet dataset not found at {config.data_dir}.")
        print("Please ensure ImageNet dataset is available at the specified path.")
        print("Expected structure: <data_dir>/train/, <data_dir>/val/")
        return

    print("Loading dataset...")
    dataset = ImageNetSimCLR(root=config.data_dir, split='train', img_size=config.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(config.num_workers > 0),
    )

    print("Creating model...")
    device = torch.device(config.device)
    model = SimCLRModel(
        base_model=config.base_model,
        projection_dim=config.projection_dim,
        img_size=config.img_size,
        pretrained=args.pretrained,
        use_flash_attn=config.use_flash_attn
    ).to(device)

    # Use the optimized NTXentLoss (no all_gather)
    criterion = NTXentLoss(temperature=config.temperature, device=device)

    # Optimizer
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'lars':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Scheduler + AMP
    steps_per_epoch = len(dataloader)
    scheduler, _ = maybe_cosine_with_warmup(optimizer, config.num_epochs, steps_per_epoch, warmup_ratio=0.1)
    scaler = torch.amp.GradScaler(enabled=getattr(config, "amp", True))

    # Resume
    start_epoch = 1
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device=device)
        start_epoch += 1

    print(f"\nStarting training on {device}...")
    print(f"Config: {config}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    for epoch in range(start_epoch, config.num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, config, rank=0, scaler=scaler, scheduler=scheduler)
        print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")

        if epoch % config.checkpoint_interval == 0:
            path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, path)

    final_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, config.num_epochs, avg_loss, final_path)
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
