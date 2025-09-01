"""
Minimal single-GPU VideoSimCLR training on disk-backed synthetic videos.

Usage:
  python train_video.py \
    --data_dir simclr-physics-baseline/synthetic_videos \
    --epochs 5 --batch_size 64 --img_size 224 \
    --lr 3e-4 --projection_dim 128 --base_model vit_base
"""

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from video_dataset import DiskVideoClipsDataset
from flow_proxy_dataset import FlowProxyDataset
from video_model import VideoSimCLRModel
from utils import NTXentLoss, set_seed, save_checkpoint

# Optional DeepSpeed
try:
    import deepspeed
    _HAS_DEEPSPEED = True
except Exception:
    _HAS_DEEPSPEED = False


def maybe_cosine_with_warmup(optimizer, num_epochs, steps_per_epoch, warmup_ratio=0.1):
    total_steps = max(1, num_epochs * max(1, steps_per_epoch))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, scaler=None, scheduler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (clip_i, clip_j, _) in enumerate(iterator):
        clip_i = clip_i.to(device, non_blocking=True)  # (B,T,C,H,W)
        clip_j = clip_j.to(device, non_blocking=True)

        x = torch.cat([clip_i, clip_j], dim=0)  # (2B,T,C,H,W)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=args.amp and torch.cuda.is_available()):
            z = model(x)              # (2B,D) L2-normalized
            loss = criterion(z)       # Local NT-Xent (expects [xi; xj])

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        iterator.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{(total_loss/num_batches):.4f}'})

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description='VideoSimCLR training on synthetic data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of saved synthetic videos (folder-of-folders)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--base_model', type=str, default='vit_base')
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='video_runs')
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--temporal', type=str, default='mean', choices=['mean', 'transformer'])
    parser.add_argument('--seed', type=int, default=42)
    # Proxy sampling
    parser.add_argument('--use_proxy_sampler', action='store_true', help='Use flow-proxy KNN positives')
    parser.add_argument('--flow_stats', type=str, default=None, help='Path to flow_stats.json (relative to data_dir if not absolute)')
    parser.add_argument('--k_pos', type=int, default=5)
    parser.add_argument('--no_same_clip_pos', action='store_true', help='Disallow positives from same clip')
    parser.add_argument('--min_time_sep', type=int, default=0, help='Min start-index separation for same-clip positives')
    # DeepSpeed
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed for multi-GPU training')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json')
    parser.add_argument('--local_rank', type=int, default=0, help='Set by launcher')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Backends
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # Data
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"--data_dir not found: {args.data_dir}")
    if args.use_proxy_sampler:
        stats_path = args.flow_stats
        if stats_path is None:
            stats_path = os.path.join(args.data_dir, 'flow_stats.json')
        if not os.path.isabs(stats_path):
            stats_path = os.path.join(args.data_dir, os.path.basename(stats_path))
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"flow_stats not found: {stats_path}. Run compute_flow_stats.py first.")
        dataset = FlowProxyDataset(
            root_dir=args.data_dir,
            stats_path=stats_path,
            img_size=args.img_size,
            k_pos=args.k_pos,
            allow_same_clip_pos=not args.no_same_clip_pos,
            min_time_separation=args.min_time_sep,
        )
    else:
        dataset = DiskVideoClipsDataset(root_dir=args.data_dir, img_size=args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )

    # DeepSpeed multi-GPU path
    if args.deepspeed and _HAS_DEEPSPEED:
        # init distributed
        deepspeed.init_distributed()
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        model = VideoSimCLRModel(
            base_model=args.base_model,
            projection_dim=args.projection_dim,
            img_size=args.img_size,
            pretrained=False,
            temporal_agg=args.temporal,
        )

        # Initialize DeepSpeed engine; let DS build the distributed DataLoader
        model_engine, optimizer, ds_loader, ds_scheduler = deepspeed.initialize(
            model=model,
            training_data=dataset,
            config=args.deepspeed_config,
        )

        criterion = NTXentLoss(temperature=args.temperature, device=device)

        if local_rank == 0:
            print("Starting DeepSpeed VideoSimCLR training...")
            print(f"Dataset size: {len(dataset)}, steps/epoch: {len(ds_loader)}")

        for epoch in range(1, args.epochs + 1):
            model_engine.train()
            total_loss = 0.0
            num_batches = 0
            progress = tqdm(ds_loader, desc=f"Epoch {epoch}/{args.epochs}") if local_rank == 0 else ds_loader
            for batch_idx, (clip_i, clip_j, _) in enumerate(progress):
                clip_i = clip_i.to(device, non_blocking=True)
                clip_j = clip_j.to(device, non_blocking=True)
                x = torch.cat([clip_i, clip_j], dim=0)  # (2B,T,C,H,W)
                z = model_engine(x)
                loss = criterion(z)
                model_engine.backward(loss)
                model_engine.step()
                total_loss += loss.item()
                num_batches += 1
                if local_rank == 0 and hasattr(progress, 'set_postfix'):
                    progress.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{(total_loss/num_batches):.4f}'})

            avg_loss = total_loss / max(1, num_batches)
            if local_rank == 0:
                print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

            if epoch % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(args.checkpoint_dir, f'epoch_{epoch}')
                model_engine.save_checkpoint(ckpt_dir)

        if local_rank == 0:
            final_dir = os.path.join(args.checkpoint_dir, 'final_model')
            model_engine.save_checkpoint(final_dir)
            print("DeepSpeed training complete.")
        return

    # ----- Single GPU path -----
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = VideoSimCLRModel(
        base_model=args.base_model,
        projection_dim=args.projection_dim,
        img_size=args.img_size,
        pretrained=False,          # synthetic; start from scratch
        temporal_agg=args.temporal,
    ).to(device)

    # Loss
    criterion = NTXentLoss(temperature=args.temperature, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Scheduler + AMP
    scheduler = maybe_cosine_with_warmup(optimizer, args.epochs, len(dataloader), warmup_ratio=0.1)
    scaler = torch.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    print(f"Starting VideoSimCLR training on {device}")
    print(f"Dataset size: {len(dataset)}, steps/epoch: {len(dataloader)}")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, scaler, scheduler)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        if epoch % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, avg_loss, final_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
