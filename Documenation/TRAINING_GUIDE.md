# Training Guide

## Start Fresh Training

```powershell
cd "C:\Stuff\coding\Test Project\training"
python train.py `
  --data-dir "W:\Datasets\AI categorisation dataset" `
  --epochs 10 `
  --batch-size 32 `
  --device cpu
```

**Arguments:**

- `--data-dir`: Path to dataset folder with `ai/` and `Real/` subdirectories
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--device`: `cpu` or `cuda` (default: cpu)
- `--lr`: Learning rate (default: 1e-4)
- `--split`: Train/validation split ratio (default: 0.8)
- `--img-size`: Image size for training (default: 224)

## Pause Training

Press **Ctrl+C** in the terminal to pause training gracefully.

A checkpoint will be saved automatically at the end of each epoch to `checkpoint.pt`.

## Resume From Checkpoint

```powershell
cd "C:\Stuff\coding\Test Project\training"
python train.py `
  --data-dir "W:\Datasets\AI categorisation dataset" `
  --epochs 10 `
  --batch-size 32 `
  --device cpu `
  --resume
```

Add `--resume` flag to continue from the last saved checkpoint.

**What gets restored:**

- Model weights
- Optimizer state
- Current epoch
- Best validation accuracy

## Progress Tracking

**Overall Progress**: Top progress bar shows total epoch progress
**Training Progress**: Per-batch progress with live loss and accuracy updates
**Validation Progress**: Per-batch validation with loss and accuracy

### Sample Output

```
Overall Progress: 30%|███       | 3/10 [2:45<6:25, 51.4s/it]

Epoch [3/10]
Training:  45%|████▌     | 103/228 [5:12<6:22, 1.66s/it, loss=0.4521, acc=82.31%]
```

## Model Outputs

**Best Model**: `models/image_classifier.pt`

- Automatically saved when validation accuracy improves

**Checkpoint**: `checkpoint.pt`

- Saved after each epoch
- Contains full training state for resume

**Training Time**: Displayed at end of training

## Performance Tips

1. **Use GPU** (if available): `--device cuda` - ~10x faster than CPU
2. **Larger batches** (if memory allows): `--batch-size 64` - more stable gradients
3. **More epochs**: `--epochs 20` - better accuracy (diminishing returns after ~15)
4. **Longer training on small dataset**: This dataset is ~9K images. Consider 15-20 epochs for better results.

## Dataset Structure

The script expects:

```
W:\Datasets\AI categorisation dataset\
  ai\
    abbey\
      image1.jpg
      image2.png
      ...
    access_road\
      ...
    ... (192 total categories)
  Real\
    abbey\
      image1.jpg
      ...
    ... (192 total categories)
```

Label assignments:

- `ai/` → label 0
- `Real/` → label 1

## Example: Resume After Pause

```powershell
# Start training
python train.py --data-dir "W:\Datasets\AI categorisation dataset" --epochs 20 --device cpu
# After some time, press Ctrl+C to pause

# Later, resume
python train.py --data-dir "W:\Datasets\AI categorisation dataset" --epochs 20 --device cpu --resume
```

The resumed training will continue from epoch N+1 and keep tracking the best validation accuracy.
