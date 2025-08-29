"""
Evaluation script for linear probing of pre-trained SimCLR encoder.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from model import SimCLRModel
from utils import set_seed
from dataset import Ball, render_frame, generate_event


class BallCountDataset(Dataset):
    """Dataset for ball counting task (downstream evaluation)."""
    
    def __init__(self, data_dir: str, num_samples: int = 1000, num_frames: int = 20,
                 box_size: int = 64, split: str = 'train'):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.box_size = box_size
        self.split = split
        
        # Generate or load data
        self.data = []
        self.labels = []
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        # Generate samples
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate samples with labels (number of balls)."""
        print(f"Generating {self.num_samples} samples for {self.split} set...")
        
        split_dir = os.path.join(self.data_dir, f"eval_{self.split}")
        os.makedirs(split_dir, exist_ok=True)
        
        for i in range(self.num_samples):
            # Random number of balls (1-3) - this is our label
            num_balls = np.random.randint(1, 4)
            self.labels.append(num_balls - 1)  # Convert to 0-indexed
            
            # Generate event
            event_dir = os.path.join(split_dir, f"sample_{i:04d}")
            os.makedirs(event_dir, exist_ok=True)
            
            # Initialize balls
            balls = []
            for _ in range(num_balls):
                radius = np.random.uniform(2.0, 4.0)
                x = np.random.uniform(radius, self.box_size - radius)
                y = np.random.uniform(radius, self.box_size - radius)
                vx = np.random.uniform(-2.0, 2.0)
                vy = np.random.uniform(-2.0, 2.0)
                balls.append(Ball(x, y, vx, vy, radius))
            
            # Generate frames
            frames = []
            for frame_idx in range(self.num_frames):
                frame = render_frame(balls, self.box_size)
                img = Image.fromarray(frame, mode='L')
                img.save(os.path.join(event_dir, f"frame_{frame_idx:02d}.png"))
                frames.append(frame)
                
                # Update ball positions
                for ball in balls:
                    ball.update(self.box_size)
            
            self.data.append(event_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Load video clip and return with label."""
        event_dir = self.data[idx]
        label = self.labels[idx]
        
        # Load frames
        frames = []
        for i in range(self.num_frames):
            frame_path = os.path.join(event_dir, f"frame_{i:02d}.png")
            frame = Image.open(frame_path).convert('L')
            frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames
        clip = torch.stack(frames, dim=0)  # Shape: (T, C, H, W)
        
        return clip, label


class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def extract_features(encoder, dataloader, device):
    """Extract features using the frozen encoder."""
    encoder.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for clips, batch_labels in tqdm(dataloader, desc="Extracting features"):
            clips = clips.to(device)
            # Get features from encoder
            batch_features = encoder(clips)
            features.append(batch_features.cpu())
            labels.append(batch_labels)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels


def train_linear_classifier(classifier, train_features, train_labels, 
                          val_features, val_labels, config, device):
    """Train the linear classifier."""
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.eval_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(50):  # Train for 50 epochs
        # Train
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validate
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return best_val_acc


def evaluate_model(encoder, test_features, test_labels, classifier, device):
    """Evaluate the model on test set."""
    classifier.eval()
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate SimCLR with Linear Probing')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SimCLR checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/evaluation', help='Directory for evaluation data')
    parser.add_argument('--num_train', type=int, default=800, help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=500, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = torch.device(config.device)
    
    # Load pre-trained model
    print("Loading pre-trained SimCLR model...")
    model = SimCLRModel(base_model=config.base_model, projection_dim=config.projection_dim)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Freeze encoder
    encoder = model.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # Create evaluation datasets
    print("\nGenerating evaluation datasets...")
    os.makedirs(args.data_dir, exist_ok=True)
    
    train_dataset = BallCountDataset(args.data_dir, num_samples=args.num_train, split='train')
    val_dataset = BallCountDataset(args.data_dir, num_samples=args.num_val, split='val')
    test_dataset = BallCountDataset(args.data_dir, num_samples=args.num_test, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.eval_batch_size, 
                            shuffle=False, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, 
                          shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    
    # Extract features
    print("\nExtracting features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)
    val_features, val_labels = extract_features(encoder, val_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    # Create linear classifier
    feature_dim = encoder.feature_dim
    classifier = LinearClassifier(feature_dim, config.eval_num_classes).to(device)
    
    # Train linear classifier
    print("\nTraining linear classifier...")
    best_val_acc = train_linear_classifier(
        classifier, train_features, train_labels,
        val_features, val_labels, config, device
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc = evaluate_model(encoder, test_features, test_labels, classifier, device)
    
    print(f"\n{'='*50}")
    print(f"Linear Probing Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")
    
    # Baseline comparison (random features)
    print("\nComputing baseline with random features...")
    random_encoder = SimCLRModel(base_model=config.base_model, projection_dim=config.projection_dim).encoder
    random_encoder = random_encoder.to(device)
    random_encoder.eval()
    
    # Extract random features
    random_train_features, _ = extract_features(random_encoder, train_loader, device)
    random_val_features, _ = extract_features(random_encoder, val_loader, device)
    random_test_features, _ = extract_features(random_encoder, test_loader, device)
    
    # Train classifier on random features
    random_classifier = LinearClassifier(feature_dim, config.eval_num_classes).to(device)
    random_val_acc = train_linear_classifier(
        random_classifier, random_train_features, train_labels,
        random_val_features, val_labels, config, device
    )
    random_test_acc = evaluate_model(random_encoder, random_test_features, test_labels, 
                                   random_classifier, device)
    
    print(f"\nRandom Features Baseline:")
    print(f"Validation Accuracy: {random_val_acc:.2f}%")
    print(f"Test Accuracy: {random_test_acc:.2f}%")
    
    print(f"\nImprovement over random: {test_acc - random_test_acc:.2f}%")


if __name__ == "__main__":
    main()
