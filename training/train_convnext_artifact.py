"""CPU-focused training script for ConvNeXt-Large + artifact branch.

This is intentionally pragmatic:
- Uses ImageNet-pretrained ConvNeXt-Large.
- Freezes the backbone by default (train head + artifact branch).
- Optionally unfreezes the last ConvNeXt stage for longer fine-tuning.
- Computes artifact channels on-the-fly (fixed residual filters + wavelet + low-res FFT).

Outputs a checkpoint compatible with `ConvNeXtLargeArtifactClassifier`.

Example:
    & .\\.venv311\\Scripts\\python.exe .\\training\\build_manifest.py --root "W:\\Datasets\\AI immage classifier 3.0 datasets" --out .\\training\\manifest.csv
    & .\\.venv311\\Scripts\\python.exe .\\training\\train_convnext_artifact.py --manifest .\\training\\manifest.csv --out .\\training\\models\\immage_classifier_V3 ConvNeXtLarge Artifact.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# Some web-scraped datasets include slightly truncated/corrupted files.
# This keeps training resilient instead of crashing a DataLoader worker.
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pywt
except Exception:
    pywt = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Item:
    path: str
    label: int
    dataset: str


def read_manifest(path: Path) -> List[Item]:
    items: List[Item] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append(Item(path=row["path"], label=int(row["label"]), dataset=row.get("dataset", "unknown")))
    return items


def split_items(
    items: List[Item],
    holdout_dataset: Optional[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Item], List[Item]]:
    if holdout_dataset:
        train = [x for x in items if x.dataset != holdout_dataset]
        val = [x for x in items if x.dataset == holdout_dataset]
        return train, val

    rng = random.Random(seed)
    items2 = items[:]
    rng.shuffle(items2)
    n_val = int(len(items2) * val_fraction)
    return items2[n_val:], items2[:n_val]


class ConvNeXtArtifactModel(nn.Module):
    def __init__(self, pretrained_rgb: bool = True, artifact_in_ch: int = 14):
        super().__init__()

        if pretrained_rgb:
            try:
                rgb = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
            except Exception:
                rgb = models.convnext_large(weights=None)
        else:
            rgb = models.convnext_large(weights=None)

        self.rgb_model = rgb
        self.rgb_norm = self.rgb_model.classifier[0]
        rgb_feat_dim = self.rgb_model.classifier[-1].in_features

        self.artifact_branch = nn.Sequential(
            nn.Conv2d(artifact_in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Linear(rgb_feat_dim + 256, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2),
        )

    def forward(self, rgb: torch.Tensor, artifact: torch.Tensor) -> torch.Tensor:
        x = self.rgb_model.features(rgb)
        x = self.rgb_model.avgpool(x)
        x = self.rgb_norm(x)
        x = x.flatten(1)

        a = self.artifact_branch(artifact).flatten(1)
        z = torch.cat([x, a], dim=1)
        return self.head(z)


def build_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    srm1 = torch.tensor(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    srm2 = torch.tensor(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        dtype=torch.float32,
    )
    k3 = torch.stack([lap, sobel_x, sobel_y], dim=0)[:, None, :, :]
    k5 = torch.stack([srm1, srm2], dim=0)[:, None, :, :]
    return k3, k5


def fft_channel(gray: torch.Tensor) -> torch.Tensor:
    h, w = int(gray.shape[-2]), int(gray.shape[-1])
    gray_small = F.avg_pool2d(gray, kernel_size=2, stride=2)
    fft = torch.fft.fft2(gray_small)
    mag = torch.abs(fft)
    mag = torch.log1p(mag)
    mag = mag - mag.amin(dim=(-2, -1), keepdim=True)
    denom = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mag = mag / denom
    return F.interpolate(mag, size=(h, w), mode="bilinear", align_corners=False)


def wavelet_channels(gray_np: np.ndarray) -> torch.Tensor:
    height, width = int(gray_np.shape[-2]), int(gray_np.shape[-1])
    if pywt is None:
        return torch.zeros((1, 3, height, width), dtype=torch.float32)
    try:
        coeffs2 = pywt.dwt2(gray_np.astype("float32"), "haar")
        _, (lh, hl, hh) = coeffs2
        wcoef = np.stack([lh, hl, hh], axis=0).astype("float32")
        w_min = wcoef.reshape(3, -1).min(axis=1)[:, None, None]
        w_max = wcoef.reshape(3, -1).max(axis=1)[:, None, None]
        denom = (w_max - w_min)
        denom[denom < 1e-6] = 1.0
        wcoef = (wcoef - w_min) / denom
        wt = torch.from_numpy(wcoef)[None, :, :, :]
        return F.interpolate(wt, size=(height, width), mode="bilinear", align_corners=False)
    except Exception:
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


def artifact_from_rgb(rgb: torch.Tensor, k3: torch.Tensor, k5: torch.Tensor) -> torch.Tensor:
    h, w = int(rgb.shape[-2]), int(rgb.shape[-1])
    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).unsqueeze(0)

    f3 = F.conv2d(gray, k3, padding=1)
    f5 = F.conv2d(gray, k5, padding=2)
    feats_224 = torch.cat([f3, f5], dim=1)

    gray_small = F.avg_pool2d(gray, kernel_size=2, stride=2)
    f3_s = F.conv2d(gray_small, k3, padding=1)
    f5_s = F.conv2d(gray_small, k5, padding=2)
    feats_112 = torch.cat([f3_s, f5_s], dim=1)
    feats_112 = F.interpolate(feats_112, size=(h, w), mode="bilinear", align_corners=False)

    fft_ch = fft_channel(gray)
    gray_np = gray.squeeze(0).squeeze(0).detach().cpu().numpy()
    wave = wavelet_channels(gray_np)

    artifact = torch.cat([feats_224, feats_112, wave.to(feats_224.device), fft_ch], dim=1)
    mean = artifact.mean(dim=(-2, -1), keepdim=True)
    std = artifact.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    artifact = (artifact - mean) / std

    return artifact.squeeze(0)


class ManifestDataset(Dataset):
    def __init__(self, items: List[Item], augment: bool, image_size: int):
        self.items = items
        self.image_size = int(image_size)

        self.resize_to_tensor = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # Mild augmentations that won't obliterate residual cues.
        self.augment = augment
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])

        # ImageNet normalization
        try:
            w = models.ConvNeXt_Large_Weights.DEFAULT
            mean = w.meta.get("mean", [0.485, 0.456, 0.406])
            std = w.meta.get("std", [0.229, 0.224, 0.225])
        except Exception:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.k3, self.k5 = build_kernels()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        try:
            img = Image.open(item.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))

        if self.augment:
            img = self.aug(img)

        rgb = self.resize_to_tensor(img)
        artifact = artifact_from_rgb(rgb, self.k3, self.k5)
        rgb = self.normalize(rgb)

        y = torch.tensor(item.label, dtype=torch.long)
        return rgb, artifact, y


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


class SimpleAdamW:
    """Minimal AdamW optimizer to avoid backend-specific fallbacks (e.g., DirectML lerp_).

    This is intentionally small and only supports what this training script uses.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--holdout-dataset", type=str, default="")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "dml"],
        help="Training device. 'dml' uses torch-directml on Windows; 'auto' tries dml then falls back to cpu.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--finetune-stage4", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    set_seed(args.seed)

    items = read_manifest(Path(args.manifest))
    train_items, val_items = split_items(items, args.holdout_dataset or None, args.val_fraction, args.seed)

    print("Train items: {}".format(len(train_items)))
    print("Val items: {}".format(len(val_items)))

    if args.device in ("auto", "dml"):
        try:
            import torch_directml  # type: ignore

            device = torch_directml.device()
            try:
                print("Using DirectML device:", torch_directml.device_name(0))
            except Exception:
                print("Using DirectML device")
        except Exception as exc:
            if args.device == "dml":
                raise SystemExit(f"DirectML requested but torch-directml is not available: {exc}")
            device = torch.device("cpu")
            print("DirectML not available; falling back to CPU")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    model = ConvNeXtArtifactModel(pretrained_rgb=True, artifact_in_ch=14)

    # Freeze all RGB backbone by default
    for p in model.rgb_model.parameters():
        p.requires_grad = False

    if args.finetune_stage4:
        # Unfreeze last stage (stage4) + norm/head.
        for name, p in model.rgb_model.named_parameters():
            if "features.7" in name or "classifier.0" in name:
                p.requires_grad = True

    for p in model.artifact_branch.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    model = model.to(device)

    train_ds = ManifestDataset(train_items, augment=True, image_size=args.image_size)
    val_ds = ManifestDataset(val_items, augment=False, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if getattr(device, "type", "") == "privateuseone":
        optim = SimpleAdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
        print("Optimizer: SimpleAdamW (DirectML-friendly)")
    else:
        optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
        print("Optimizer: AdamW")
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int, train_loss: float, train_acc: float, val_acc: float, tag: str = "") -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "args": vars(args),
            "ai_class_index": 0,
        }

        stem = out_path.stem
        suffix = out_path.suffix or ".pt"
        tag2 = ("_" + tag) if tag else ""
        epoch_copy = out_path.with_name(f"{stem}_epoch{epoch:04d}{tag2}{suffix}")
        latest_copy = out_path.with_name(f"{stem}_latest{suffix}")

        torch.save(payload, epoch_copy)
        torch.save(payload, latest_copy)
        torch.save(payload, out_path)

        print(f"Saved checkpoint: {epoch_copy}")

    epoch = 0
    try:
        while True:
            epoch += 1
            if args.epochs > 0 and epoch > args.epochs:
                break

            model.train()
            running_loss = 0.0
            running_acc = 0.0
            n_batches = 0

            train_iter = train_loader
            if tqdm is not None:
                train_iter = tqdm(train_loader, desc=f"train e{epoch}", unit="batch", leave=False)

            for rgb, artifact, y in train_iter:
                rgb = rgb.to(device)
                artifact = artifact.to(device)
                y = y.to(device)

                optim.zero_grad(set_to_none=True)
                logits = model(rgb, artifact)
                loss = criterion(logits, y)
                loss.backward()
                optim.step()

                batch_loss = float(loss.item())
                batch_acc = accuracy_from_logits(logits.detach(), y)

                running_loss += batch_loss
                running_acc += batch_acc
                n_batches += 1

                if tqdm is not None:
                    train_iter.set_postfix(loss=f"{running_loss/max(1,n_batches):.4f}", acc=f"{running_acc/max(1,n_batches):.4f}")

                if args.max_train_batches and n_batches >= args.max_train_batches:
                    break

            train_loss = running_loss / max(1, n_batches)
            train_acc = running_acc / max(1, n_batches)

            model.eval()
            val_acc_sum = 0.0
            val_loss_sum = 0.0
            val_batches = 0

            val_iter = val_loader
            if tqdm is not None:
                val_iter = tqdm(val_loader, desc=f"val   e{epoch}", unit="batch", leave=False)

            with torch.no_grad():
                for rgb, artifact, y in val_iter:
                    rgb = rgb.to(device)
                    artifact = artifact.to(device)
                    y = y.to(device)
                    logits = model(rgb, artifact)
                    val_loss_sum += float(criterion(logits, y).item())
                    val_acc_sum += accuracy_from_logits(logits, y)
                    val_batches += 1

                    if tqdm is not None:
                        val_iter.set_postfix(acc=f"{val_acc_sum/max(1,val_batches):.4f}")

                    if args.max_val_batches and val_batches >= args.max_val_batches:
                        break

            val_acc = val_acc_sum / max(1, val_batches)
            val_loss = val_loss_sum / max(1, val_batches)

            print(
                "Epoch {} | train_loss {:.4f} | val_loss {:.4f} | train_acc {:.4f} | val_acc {:.4f}".format(
                    epoch, train_loss, val_loss, train_acc, val_acc
                )
            )

            save_checkpoint(epoch, train_loss=train_loss, train_acc=train_acc, val_acc=val_acc)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
    except KeyboardInterrupt:
        print("Interrupted by user; saving interrupt checkpoint...")
        # If interruption happens mid-epoch before val, just reuse the last computed metrics.
        try:
            save_checkpoint(epoch, train_loss=train_loss, train_acc=train_acc, val_acc=val_acc, tag="interrupt")
        except Exception:
            # Fallback: save weights-only.
            torch.save({"model_state_dict": model.state_dict(), "epoch": int(epoch)}, out_path.with_suffix(".interrupt.pt"))
        raise

    print("Best val_acc: {:.4f}".format(best_val_acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
