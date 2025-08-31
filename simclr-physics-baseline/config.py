"""
Configuration file for SimCLR training.
"""

import os


class Config:
    """Configuration class for SimCLR training."""
    
    # Data settings
    data_dir = '/data/yaxin/data/imagenet'  # ImageNet1K dataset path
    img_size = 224  # image resolution (224 to match pretrained ViT)
    
    # Model settings
    base_model = 'vit_base'  # Standard ViT instead of ViViT
    projection_dim = 512
    use_flash_attn = True  # Enable Flash Attention via PyTorch SDPA
    
    # Training settings
    batch_size = 256  # Total batch size across all GPUs (32 per GPU * 8 GPUs)
    num_epochs = 100
    learning_rate = 1e-3  # Learning rate for ViT-based SimCLR training
    weight_decay = 0.05   # Standard ViT weight decay
    temperature = 0.5
    
    # New training stability settings
    gradient_clip_norm = 1.0  # Gradient clipping
    warmup_steps = 1000       # LR warmup steps
    use_proper_weight_decay = True  # Exclude bias/norm from weight decay
    amp = True  # Automatic Mixed Precision
    
    # DeepSpeed settings
    use_deepspeed = True  # Enable DeepSpeed ZeRO-2 by default
    deepspeed_config_file = 'deepspeed_config.json'
    
    # Optimizer settings
    optimizer = 'adamw'  # 'adamw', 'adam', or 'lars'
    
    # Checkpoint settings
    checkpoint_dir = 'runs'
    checkpoint_interval = 10
    
    # Device settings
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    num_workers = 8
    
    # Logging settings
    log_interval = 10
    
    # Evaluation settings
    eval_batch_size = 128
    eval_num_classes = 1000  # ImageNet has 1000 classes
    
    # Random seed
    seed = 42
    
    def __init__(self, **kwargs):
        """Initialize config and update with any provided arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def __str__(self):
        """String representation of configuration."""
        attrs = [f"{key}={value}" for key, value in self.__dict__.items() 
                if not key.startswith('_')]
        return f"Config({', '.join(attrs)})"
