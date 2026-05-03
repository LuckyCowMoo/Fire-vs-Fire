"""
Training Script for AI-Generated Image Detection
Uses PyTorch with ResNet-50 + FFT to train a binary classifier.

Usage:
    python train.py --data-dir "W:\\Datasets\\AI categorisation dataset" --epochs 10 --batch-size 32 --device cpu
    python train.py --data-dir "W:\\Datasets\\AI categorisation dataset" --resume  # Resume from checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from transformers import DistilBertModel
import numpy as np
from scipy import fftpack
from pathlib import Path
import json
from datetime import datetime, timedelta
import argparse
import time
from tqdm import tqdm
from datasets import ImageDataset

# Parse arguments
parser = argparse.ArgumentParser(description='Train AI content detectors')
parser.add_argument('--device', type=str, default='cpu', 
                    help='Device to use: cpu, cuda (default: cpu)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='Number of epochs (default: 10)')
parser.add_argument('--batch-size', type=int, default=32, 
                    help='Batch size for image training (default: 32)')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to dataset directory with ai/ and Real/ folders')
parser.add_argument('--img-size', type=int, default=224,
                    help='Image size for training (default: 224)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--split', type=float, default=0.8,
                    help='Train/validation split ratio (default: 0.8)')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from latest checkpoint')
args = parser.parse_args()

# Consolidated one-time environment summary
DEVICE = torch.device(args.device)
print(f"Env: torch {torch.__version__}, device: {DEVICE}")
if torch.cuda.is_available() and DEVICE.type == 'cuda':
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
print()


# ============================================================================
# IMAGE CLASSIFIER (ResNet-50 backbone with FFT preprocessing)
# ============================================================================

class ImageClassifier(nn.Module):
    """ResNet-50 classifier with FFT channel for AI-generated detection.
    
    Input: 4-channel images (RGB + FFT magnitude)
    - Channels 0-2: Standard RGB image
    - Channel 3: FFT magnitude (detects frequency artifacts from AI generators)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        # Modify first conv layer to accept 4 channels instead of 3
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize new channel with small weights
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
            self.backbone.conv1.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True) * 0.1
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# TEXT CLASSIFIER (DistilBERT backbone)
# ============================================================================

class TextClassifier(nn.Module):
    """Text classifier using DistilBERT for AI-generated detection."""
    def __init__(self, num_classes=2, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs[0][:, 0, :]  # [CLS] token
        return self.classifier(pooled)


# (Removed dummy datasets; using real dataset loader from training/datasets.py)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, task="image"):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if task == "image":
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        else:  # text
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.2%}"})
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, task="image"):
    """Evaluate model on validation set with progress bar."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            if task == "image":
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            else:  # text
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.2%}"})
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path="checkpoint.pt"):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, checkpoint_path)
    print(f"  [CHECKPOINT] Saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path="checkpoint.pt"):
    """Load training checkpoint. Returns checkpoint dict or None."""
    if not Path(checkpoint_path).exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"[CHECKPOINT] Loaded from {checkpoint_path}")
    print(f"  Resuming from epoch {checkpoint['epoch'] + 1}, best_val_acc={checkpoint['best_val_acc']:.2%}")
    return checkpoint


# ============================================================================
# MAIN: Train both models
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IMAGE CLASSIFICATION TRAINING - AI vs Real Images")
    print("=" * 70)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    full_dataset = ImageDataset(root_dir=args.data_dir, img_size=args.img_size, nested=True)
    
    # Split into train/val
    train_size = int(args.split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = ImageClassifier(num_classes=2, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = Path("checkpoint.pt")
    
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print()
        else:
            print("[WARNING] No checkpoint found, starting fresh\n")
    
    # Calculate total batches and estimated time
    total_batches = len(train_loader) * (args.epochs - start_epoch)
    
    # Training loop with progress
    epoch_start_time = time.time()
    with tqdm(total=args.epochs, desc="Overall Progress", initial=start_epoch) as pbar_epochs:
        for epoch in range(start_epoch, args.epochs):
            epoch_time = time.time()
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, task="image")
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, task="image")
            
            epoch_elapsed = time.time() - epoch_time
            
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2%}")
            print(f"  Time:  {epoch_elapsed:.1f}s")
            
            # Save checkpoint every epoch
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = Path("models") / "image_classifier.pt"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"  [SAVED] New best model with val_acc={val_acc:.2%}")
            
            pbar_epochs.update(1)
    
    total_time = time.time() - epoch_start_time
    print(f"\n[OK] Training complete! Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: models/image_classifier.pt")
    print(f"Total training time: {timedelta(seconds=int(total_time))}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
