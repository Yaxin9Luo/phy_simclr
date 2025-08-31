"""
Testing script for trained linear classifier on ImageNet validation set.
Evaluates the quality of SimCLR representations using the trained linear head.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import time
import json

from model import SimCLRModel
from config import Config
from linear_eval import SimCLRLinearEval
from utils import set_seed


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_trained_model(checkpoint_path, simclr_checkpoint_path, config, device):
    """Load the trained linear classifier model."""
    
    # Load the original SimCLR model
    simclr_model = SimCLRModel(
        base_model=config.base_model,
        projection_dim=config.projection_dim,
        img_size=config.img_size,
        pretrained=False,
        use_flash_attn=config.use_flash_attn
    )
    
    # Load SimCLR checkpoint
    if os.path.exists(os.path.join(simclr_checkpoint_path, 'zero_to_fp32.py')):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(simclr_checkpoint_path)
        simclr_model.load_state_dict(state_dict, strict=False)
    else:
        checkpoint_file = os.path.join(simclr_checkpoint_path, 'mp_rank_00_model_states.pt')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
        simclr_model.load_state_dict(state_dict, strict=False)
    
    # Create the linear evaluation model
    model = SimCLRLinearEval(simclr_model, num_classes=1000)
    
    # Load the trained linear classifier weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def get_val_dataloader(data_dir, batch_size=256, num_workers=4):
    """Create ImageNet validation dataloader."""
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return val_loader


def test_model(model, dataloader, device):
    """Test the model and compute metrics."""
    
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Track per-class accuracy
    class_correct = torch.zeros(1000)
    class_total = torch.zeros(1000)
    
    end = time.time()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        
        for i, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Compute output
            outputs = model(images)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Per-class accuracy
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            for j in range(targets.size(0)):
                label = targets[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            pbar.set_postfix({
                'Top1': f'{top1.avg:.2f}%',
                'Top5': f'{top5.avg:.2f}%',
                'Time': f'{batch_time.avg:.3f}s'
            })
    
    # Calculate per-class accuracies
    class_accuracies = 100 * class_correct / class_total
    mean_class_acc = class_accuracies.mean().item()
    
    return {
        'top1_accuracy': top1.avg.item(),
        'top5_accuracy': top5.avg.item(),
        'mean_class_accuracy': mean_class_acc,
        'class_accuracies': class_accuracies.tolist(),
        'total_samples': len(dataloader.dataset),
        'batch_time': batch_time.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Test trained linear classifier')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained linear classifier checkpoint')
    parser.add_argument('--simclr_checkpoint', type=str, required=True,
                       help='Path to original SimCLR checkpoint directory')
    parser.add_argument('--data_dir', type=str, default='/home/yaxin/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for testing')
    parser.add_argument('--save_results', type=str, default='test_results.json',
                       help='Path to save test results')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = Config()
    
    # Load trained model
    print(f"Loading trained model from {args.model_path}")
    print(f"Loading SimCLR checkpoint from {args.simclr_checkpoint}")
    
    model, checkpoint_info = load_trained_model(
        args.model_path, 
        args.simclr_checkpoint, 
        config, 
        device
    )
    
    print(f"Model loaded successfully!")
    print(f"Training info:")
    print(f"  - Best validation accuracy: {checkpoint_info.get('best_acc', 'N/A'):.2f}%")
    print(f"  - Training epoch: {checkpoint_info.get('epoch', 'N/A')}")
    print(f"  - Final training accuracy: {checkpoint_info.get('train_acc', 'N/A'):.2f}%")
    print(f"  - Final validation accuracy: {checkpoint_info.get('val_acc', 'N/A'):.2f}%")
    
    # Create test dataloader
    print("Loading ImageNet validation dataset...")
    val_loader = get_val_dataloader(args.data_dir, args.batch_size, args.num_workers)
    print(f"Test samples: {len(val_loader.dataset)}")
    
    # Test model
    print("\nRunning inference on validation set...")
    start_time = time.time()
    
    results = test_model(model, val_loader, device)
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['checkpoint_info'] = {
        'model_path': args.model_path,
        'simclr_checkpoint': args.simclr_checkpoint,
        'training_epoch': checkpoint_info.get('epoch', 'N/A'),
        'training_best_acc': checkpoint_info.get('best_acc', 'N/A')
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"IMAGENET LINEAR EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"Mean Class Accuracy: {results['mean_class_accuracy']:.2f}%")
    print(f"Total Samples: {results['total_samples']:,}")
    print(f"Inference Time: {total_time:.2f} seconds")
    print(f"Samples/sec: {results['total_samples']/total_time:.1f}")
    print(f"{'='*60}")
    
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
