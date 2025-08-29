"""
Configuration file for SimCLR training.
"""

import os


class Config:
    """Configuration class for SimCLR training."""
    
    # Data settings
    data_dir = 'processed_clevrer'
    num_frames = 20
    box_size = 256  # image frame resolution
    
    # Model settings
    base_model = 'resnet18'
    projection_dim = 128
    
    # Training settings
    batch_size = 256  # Total batch size across all GPUs (32 per GPU * 8 GPUs)
    num_epochs = 100
    learning_rate = 3e-4
    weight_decay = 1e-4
    temperature = 0.5
    
    # Optimizer settings
    optimizer = 'adam'  # 'adam' or 'lars'
    momentum = 0.9  # for LARS
    
    # Checkpoint settings
    checkpoint_dir = 'runs'
    checkpoint_interval = 10
    
    # Device settings
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    num_workers = 4
    
    # Logging settings
    log_interval = 10
    
    # Evaluation settings
    eval_batch_size = 128
    eval_num_classes = 3  # For classifying number of balls (1, 2, or 3)
    
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
