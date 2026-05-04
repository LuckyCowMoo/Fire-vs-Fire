"""Training entrypoint for the hybrid AI-vs-human classifier.

This script is intentionally conservative about dependencies. It can read the
existing feature CSV and train a DistilBERT + handcrafted feature model when
torch/transformers are installed. If the environment only has the source code,
the script still parses and documents the exact training flow without forcing
imports at module load time.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from hybrid_classifier import FEATURE_COLUMNS, HybridTextClassifier

# Increase CSV field size limit for large text fields
csv.field_size_limit(10_000_000)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class Sample:
    text: str
    label: int
    features: List[float]


def load_feature_rows(csv_path: Path, limit: int | None = None) -> Tuple[List[Sample], List[Dict[str, str]]]:
    """Load samples and return both Sample objects and raw row dicts."""
    samples: List[Sample] = []
    raw_rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            text = (row.get("text") or "").strip()
            if not text:
                continue
            label = int(float(row.get("label", 0)))
            features = [float(row.get(name, 0.0) or 0.0) for name in FEATURE_COLUMNS]
            samples.append(Sample(text=text, label=label, features=features))
            raw_rows.append(dict(row))
    return samples, raw_rows


def _backfill_missing_perplexity(
    csv_path: Path,
    samples: List[Sample],
    raw_rows: List[Dict[str, str]],
    device: str,
    batch_size: int = 32,
) -> None:
    """Generate missing GPT-2 features and write them back to CSV."""
    import sys
    
    # GPT-2 feature columns
    gpt2_cols = ["ppl_mean", "token_logprob_mean", "token_logprob_std",
                 "token_top1_frac", "token_top5_frac", "token_top10_frac", "token_entropy_mean"]
    
    # Find rows where all GPT-2 features are zero
    missing_indices = []
    for idx, sample in enumerate(samples):
        feature_dict = {name: val for name, val in zip(FEATURE_COLUMNS, sample.features)}
        if all(abs(feature_dict.get(col, 0.0)) < 1e-9 for col in gpt2_cols):
            missing_indices.append(idx)
    
    if not missing_indices:
        print("All rows have GPT-2 features — skipping backfill", file=sys.stderr, flush=True)
        return
    
    print(f"\nBackfilling {len(missing_indices)} rows with missing GPT-2 features...", file=sys.stderr, flush=True)
    
    # Import the batched scorer
    from prepare_features import BatchedGPT2Scorer
    scorer = BatchedGPT2Scorer(device=device)
    
    # Process in batches
    iterator = range(0, len(missing_indices), batch_size)
    if tqdm:
        iterator = tqdm(iterator, desc="backfill", unit="batch")
    
    for start in iterator:
        batch_indices = missing_indices[start:start + batch_size]
        texts = [samples[i].text for i in batch_indices]
        gpt2_features = scorer.score_batch(texts)
        
        # Update samples in-memory
        for local_idx, global_idx in enumerate(batch_indices):
            gpt2_dict = gpt2_features[local_idx]
            feature_dict = {name: val for name, val in zip(FEATURE_COLUMNS, samples[global_idx].features)}
            feature_dict.update(gpt2_dict)
            samples[global_idx].features = [feature_dict[name] for name in FEATURE_COLUMNS]
            
            # Update raw row
            for col in gpt2_cols:
                raw_rows[global_idx][col] = str(gpt2_dict.get(col, 0.0))
    
    # Write updated CSV
    print("Writing updated CSV with backfilled features...", file=sys.stderr, flush=True)
    fieldnames = list(raw_rows[0].keys()) if raw_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"✓ Backfill complete — updated {csv_path}", file=sys.stderr, flush=True)


def split_samples(samples: Sequence[Sample], train_ratio: float, validation_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train: List[Sample] = []
    validation: List[Sample] = []
    test: List[Sample] = []

    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        total = len(label_samples)
        train_end = int(total * train_ratio)
        validation_end = train_end + int(total * validation_ratio)
        train.extend(label_samples[:train_end])
        validation.extend(label_samples[train_end:validation_end])
        test.extend(label_samples[validation_end:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def _run_torch_training(args: argparse.Namespace, train_samples: Sequence[Sample], validation_samples: Sequence[Sample]) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from hybrid_classifier import _HybridTextTorchModel

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for training the hybrid model") from exc
    
    if tqdm is None:
        raise RuntimeError("tqdm is required for training progress bars")

    class HybridDataset(Dataset):
        def __init__(self, samples: Sequence[Sample], tokenizer, max_length: int) -> None:
            self.samples = list(samples)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            encoded = self.tokenizer(
                sample.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "features": torch.tensor(sample.features, dtype=torch.float32),
                "label": torch.tensor(1.0 - sample.label, dtype=torch.float32),
            }

    def collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "features": torch.stack([item["features"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
        }

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import torch_directml
        if args.device == "dml":
            device = torch_directml.device()
            print(f"Using DirectML device: {torch_directml.device_name(0)}")
    except ImportError:
        pass
    model = _HybridTextTorchModel(
        feature_names=FEATURE_COLUMNS,
        text_model_name=args.text_model_name,
        feature_importance_init=[1.0] * len(FEATURE_COLUMNS),
        feature_hidden_dim=args.feature_hidden_dim,
        merge_hidden_dim=args.merge_hidden_dim,
        dropout=args.dropout,
    )
    model.to(device)

    train_loader = DataLoader(HybridDataset(train_samples, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(HybridDataset(validation_samples, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Give DistilBERT backbone 10x lower learning rate so feature layers dominate
    backbone_params = list(model.module.text_backbone.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.learning_rate * 0.1},
        {"params": other_params, "lr": args.learning_rate},
    ])
    loss_function = torch.nn.BCEWithLogitsLoss()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp for unique checkpoint names
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_checkpoint = checkpoint_dir / f"latest_{timestamp}.pt"
    best_checkpoint = checkpoint_dir / f"best_{timestamp}.pt"
    
    # Try to resume from most recent checkpoint if --resume flag is set
    if args.resume:
        existing_checkpoints = sorted(checkpoint_dir.glob("latest_*.pt"))
        if existing_checkpoints:
            latest_checkpoint_to_load = existing_checkpoints[-1]
        else:
            latest_checkpoint_to_load = None
    else:
        latest_checkpoint_to_load = None
    
    start_epoch = 0
    best_validation_loss = float("inf")

    if latest_checkpoint_to_load and latest_checkpoint_to_load.exists():
        checkpoint = torch.load(latest_checkpoint_to_load, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        
        # Verify weights loaded by checking a sample parameter
        sample_param = next(model.parameters())
        print(f"Sample weight mean: {sample_param.mean().item():.6f}, std: {sample_param.std().item():.6f}")
        
        # Load optimizer state after model is on correct device
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", best_validation_loss))
        print(f"resumed_from={latest_checkpoint_to_load} start_epoch={start_epoch} best_val_loss={best_validation_loss:.4f}")
        
        # Optimizer state must be loaded after model is on device
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load optimizer state ({e}), starting with fresh optimizer")

    def save_checkpoint(path: Path, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": validation_loss,
                "config": {
                    "text_model_name": args.text_model_name,
                    "feature_hidden_dim": args.feature_hidden_dim,
                    "merge_hidden_dim": args.merge_hidden_dim,
                    "dropout": args.dropout,
                },
                "feature_importance_init": [1.0] * len(FEATURE_COLUMNS),
                "feature_names": list(FEATURE_COLUMNS),
                "backend": "CPU" if device.type == "cpu" else "CUDA",
                "device_name": str(device),
            },
            path,
        )

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_items = 0
        correct = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                targets = batch["label"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
                loss = loss_function(logits, targets)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                total_loss += loss.item() * targets.size(0)
                total_items += targets.size(0)
                correct += int((predictions == targets).sum().item())
        return total_loss / max(total_items, 1), correct / max(total_items, 1)

    def test_external_files():
        """Test on external text files to check for overfitting."""
        test_files = [
            ("gemeni text.txt", "AI"),
            ("wikipedia.txt", "Human"),
            ("Reddit.txt", "Human"),
        ]
        model.eval()
        print("\n" + "="*60)
        print("Testing on external files:")
        for filename, expected_label in test_files:
            filepath = Path(filename)
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            segments = [s.strip() for s in text.split("\n\n") if len(s.strip()) >= 150]
            if not segments:
                continue
            
            # Create mini-batches
            ai_count = 0
            total = len(segments)
            for i in range(0, len(segments), 8):
                batch_texts = segments[i:i+8]
                encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
                
                # Extract features
                from hybrid_classifier import TextFeatureExtractor, GPT2PerplexityScorer
                extractor = TextFeatureExtractor(GPT2PerplexityScorer(device=str(device)))
                features_list = [extractor.extract(t).as_list() for t in batch_texts]
                
                with torch.no_grad():
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    features_tensor = torch.tensor(features_list, dtype=torch.float32, device=device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features_tensor)
                    probs = torch.sigmoid(logits)
                    ai_count += int((probs < 0.5).sum().item())
            
            accuracy = (ai_count / total * 100) if expected_label == "AI" else ((total - ai_count) / total * 100)
            print(f"  {filename}: {ai_count}/{total} classified as AI ({accuracy:.1f}% {'correct' if expected_label == 'AI' else 'WRONG'})")
        print("="*60 + "\n")

    patience = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_items = 0
        correct_train = 0
        batch_iterator = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = loss_function(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track training accuracy
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct_train += int((preds == targets).sum().item())

            running_loss += loss.item() * targets.size(0)
            running_items += targets.size(0)
            
            # Update progress bar with loss and running accuracy
            current_acc = correct_train / running_items if running_items > 0 else 0.0
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.3f}")
            
            # Clear cache periodically to prevent memory accumulation
            if (running_items // args.batch_size) % 100 == 0:
                import gc
                gc.collect()

        train_loss = running_loss / max(running_items, 1)
        validation_loss, validation_accuracy = evaluate(validation_loader)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} validation_accuracy={validation_accuracy:.4f}")
        print(f"  (train: {correct_train}/{running_items} = {correct_train/max(running_items,1):.4f})")
        
        # Test on external files after each epoch
        test_external_files()

        save_checkpoint(latest_checkpoint, epoch, validation_loss)
        save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}_{timestamp}.pt", epoch, validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience = 0
            save_checkpoint(best_checkpoint, epoch, validation_loss)
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print("early stopping triggered")
                break


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hybrid AI-vs-human text classifier")
    parser.add_argument("--data", type=Path, default=Path("Dataset/model_training_dataset_en_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/hybrid_classifier.pt"))
    parser.add_argument("--text-model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--feature-hidden-dim", type=int, default=128)
    parser.add_argument("--merge-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/hybrid_classifier"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", help="Device for backfill: 'dml' or 'cpu'")
    parser.add_argument("--backfill-batch-size", type=int, default=32, help="Batch size for GPT-2 backfill")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    samples, raw_rows = load_feature_rows(args.data, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples found in {args.data}")
    
    print(f"loaded={len(samples)} samples from {args.data}")
    
    # Backfill missing GPT-2 features before training
    device_str = "dml" if args.device == "dml" else "cpu"
    _backfill_missing_perplexity(
        csv_path=args.data,
        samples=samples,
        raw_rows=raw_rows,
        device=device_str,
        batch_size=args.backfill_batch_size,
    )

    train_samples, validation_samples, test_samples = split_samples(samples, args.train_ratio, args.validation_ratio, args.seed)
    print(f"train={len(train_samples)} validation={len(validation_samples)} test={len(test_samples)}")

    _run_torch_training(args, train_samples, validation_samples)

    best_checkpoint = Path(args.checkpoint_dir) / "best.pt"
    if best_checkpoint.exists():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_checkpoint, args.output)
        print(f"exported_best_checkpoint={args.output}")


if __name__ == "__main__":
    main()