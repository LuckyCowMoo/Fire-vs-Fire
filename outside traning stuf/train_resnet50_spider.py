#!/usr/bin/env python3
"""
Train ResNet-50 spider detector on binary classification task.

Architecture:
- Backbone: ResNet-50 (ImageNet pretrained, frozen)
- Head: FC(2048) -> 2 classes (spider/non-spider)
- Input: 224x224 RGB
- Preprocessing: ImageNet normalization
- Augmentation: Horizontal flip only (50%)
- Loss: Cross-entropy
- Optimizer: Adam (lr=1e-4) or DirectML-compatible SimpleAdamW

Usage:
    python training/train_resnet50_spider.py --manifest training/spider_manifest.csv --output models/spider_detector.pt
"""

import os
import sys
import csv
import argparse
import logging
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpiderDataset(Dataset):
    """Load images and labels from manifest CSV."""
    
    def __init__(self, manifest_path, transform=None, max_samples=None):
        """
        Args:
            manifest_path: Path to CSV with columns [path, label, dataset]
            transform: Optional transforms to apply
            max_samples: Limit dataset size (for testing)
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_root = self.manifest_path.parent
        self.project_root = self.manifest_root.parent
        self.transform = transform
        
        # Load manifest
        self.items = []
        try:
            with open(self.manifest_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.items.append({
                        'path': row['path'],
                        'label': int(row['label']),
                        'dataset': row.get('dataset', 'unknown')
                    })
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise
        
        # Limit size if specified
        if max_samples:
            self.items = self.items[:max_samples]
        
        logger.info(f"Loaded {len(self.items)} items from manifest")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        rel_path = Path(item['path'])
        if rel_path.is_absolute():
            img_path = rel_path
        else:
            candidate_paths = [
                self.manifest_root / rel_path,
                self.project_root / rel_path,
                Path.cwd() / rel_path,
            ]
            img_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
        label = item['label']
        
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e} - using blank image")
            from PIL import Image
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class SimpleAdamW:
    """Minimal AdamW optimizer to avoid backend-specific fallbacks (e.g., DirectML lerp_).
    Compatible with DirectML on AMD GPUs.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("SimpleAdamW does not support sparse gradients")

            state = self.state.get(p)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format),
                }
                self.state[p] = state

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Decoupled weight decay
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            # Adam moments (avoid lerp_)
            exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            # Bias corrections
            bias_correction1 = 1.0 - (self.beta1**step)
            bias_correction2 = 1.0 - (self.beta2**step)
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(self.eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)


class ResNet50SpiderDetector(nn.Module):
    """ResNet-50 with frozen backbone + binary classification head."""
    
    def __init__(self, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        final_features = self.backbone.fc.in_features
        
        # Replace final FC with identity to get features
        self.backbone.fc = nn.Identity()
        
        # Add binary classification head
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(final_features, 2)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen (no gradients)")
        else:
            logger.info("Backbone trainable (full fine-tuning)")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{correct / total:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    if len(val_loader) == 0:
        return 0.0, 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{correct / total:.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def train_model(
    manifest_path,
    output_path,
    num_epochs=12,
    batch_size=32,
    learning_rate=1e-4,
    max_samples=None,
    val_split=0.1,
    device_id=0,
    freeze_backbone=True
):
    """
    Train spider detector model.
    
    Args:
        manifest_path: Path to manifest CSV
        output_path: Where to save model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_samples: Limit dataset size (for testing)
        val_split: Validation split fraction
        device_id: GPU device ID
        freeze_backbone: Whether to freeze ResNet-50 backbone
    """
    
    # Try DirectML first (AMD GPU), then CUDA, then CPU
    use_simple_adam = False
    try:
        import torch_directml
        device = torch_directml.device()
        try:
            device_name = torch_directml.device_name(0)
            logger.info(f"Using DirectML device: {device_name}")
        except Exception:
            logger.info("Using DirectML device")
        use_simple_adam = True
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
            use_simple_adam = False
        else:
            device = torch.device('cpu')
            logger.warning(
                "Using CPU device because torch-directml is not installed in this environment and CUDA is unavailable"
            )
            use_simple_adam = False
    pin_memory = getattr(device, 'type', '') != 'cpu'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Data loading
    logger.info("Loading dataset...")
    
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Only horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load full dataset
    full_dataset = SpiderDataset(
        manifest_path,
        transform=train_transform,
        max_samples=max_samples
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    if len(full_dataset) >= 2 and val_size == 0:
        val_size = 1
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train set: {len(train_dataset)}")
    logger.info(f"Val set: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )
    
    # Model
    logger.info("Creating model...")
    model = ResNet50SpiderDetector(freeze_backbone=freeze_backbone).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize head if backbone is frozen
    if freeze_backbone:
        params_to_optimize = model.head.parameters()
    else:
        params_to_optimize = model.parameters()
    
    # Use DirectML-compatible optimizer if on DirectML
    if use_simple_adam:
        optimizer = SimpleAdamW(params_to_optimize, lr=learning_rate, weight_decay=0.0)
        logger.info("Using SimpleAdamW optimizer (DirectML-compatible)")
    else:
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
        logger.info("Using Adam optimizer")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        logger.info(f"  Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  Saved best model (val_acc: {val_acc:.4f})")
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ResNet-50 spider detector'
    )
    parser.add_argument(
        '--manifest',
        required=True,
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--output',
        default='training/models/spider_detector_resnet50.pt',
        help='Output model path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=12,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit dataset size (for testing)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split fraction'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        default=True,
        help='Freeze ResNet-50 backbone'
    )
    parser.add_argument(
        '--unfreeze-backbone',
        action='store_true',
        help='Unfreeze backbone for full fine-tuning'
    )
    
    args = parser.parse_args()
    
    # Handle freeze/unfreeze
    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone
    
    train_model(
        manifest_path=args.manifest,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        val_split=args.val_split,
        device_id=args.device,
        freeze_backbone=freeze_backbone
    )
