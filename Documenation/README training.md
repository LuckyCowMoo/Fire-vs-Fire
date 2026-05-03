# GPU Training Setup for AI Content Detector

## Hardware

- **GPU**: 7900 XTX (RDNA 3, 384-bit)
- **Driver**: AMD Radeon Adrenalin (latest)
- **OS**: Windows 11

## Installation

### 1. Install PyTorch with ROCm (AMD GPU support)

For your 7900 XTX on Windows, you **must** use ROCm. Standard PyTorch won't use AMD GPUs on Windows.

**Option A: Using pip with ROCm 5.7 (Recommended for your GPU)**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**Option B: Using conda**

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install HIP manually for AMD support.

### 2. Verify GPU Detection

```powershell
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

If `False`, check:

- AMD drivers are up to date
- ROCm libraries are installed (`rocm-core`, `rocminfo`)
- PyTorch was installed with ROCm support

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Usage

### First Run: Train Dummy Models

```powershell
python train.py
```

This trains:

- **ResNet-18** on random images (learns quickly to see GPU speed)
- **DistilBERT** on random text

You'll see GPU memory usage and training speed. On your 7900 XTX, expect:

- Image training: ~50-100ms per batch
- Text training: ~200-400ms per batch (BERT is slower)

### Next: Use Real Data

Edit `DummyImageDataset` and `DummyTextDataset` in `train.py` to load your actual datasets:

```python
# Replace DummyImageDataset with:
class RealImageDataset(Dataset):
    def __init__(self, image_dir, label_file):
        # Load images from disk
        # Return (image_tensor, label)
        pass

# Replace DummyTextDataset with:
class RealTextDataset(Dataset):
    def __init__(self, text_file):
        # Load texts from file or database
        # Return ({"input_ids": ..., "attention_mask": ..., "label": ...})
        pass
```

## Data Collection Tips

For your "AI-generated vs. real" classifier:

**Images:**

- AI-generated: DALL-E, Midjourney, Stable Diffusion outputs (~500-1000)
- Real: Stock photos, web images (~500-1000)

**Text:**

- AI-generated: ChatGPT, Claude outputs (~500-1000 snippets)
- Real: News articles, Reddit posts, books (~500-1000 snippets)

Start with ~1000 labeled samples per class, then expand.

## Performance Tips

1. **Batch size**: Adjust based on GPU memory
   - If OOM: reduce batch_size in DataLoader
   - Your 7900 XTX: Can handle batch_size=32+ for ResNet, batch_size=8-16 for BERT

2. **Learning rate**: Start conservative
   - ResNet: 1e-4 (already set)
   - BERT: 2e-5 (already set)

3. **Early stopping**: Add validation checks to stop if loss plateaus

4. **Mixed precision**: Can speed up training by 2-3x
   ```python
   from torch.cuda.amp import autocast, GradScaler
   # Wrap forward pass in: with autocast():
   ```

## WSL vs Windows

- **Windows PyTorch + ROCm**: Works fine for your use case, simpler setup
- **WSL2 + PyTorch + ROCm**: Better if you want Linux workflow, needs extra setup
- **Recommendation**: Start on Windows, move to WSL if you need Linux tools

## Troubleshooting

**PyTorch says GPU unavailable:**

- Check `rocminfo` in terminal
- Ensure AMD driver is 23.x or later
- Try: `pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7`

**Out of Memory (OOM):**

- Reduce batch_size
- Use gradient checkpointing for BERT: `model.bert.gradient_checkpointing_enable()`

**Very slow training (using CPU instead of GPU):**

- Verify with `torch.cuda.is_available()` and `next(model.parameters()).device`
- Check GPU task manager to see if GPU is active

## Next Steps

1. Run `train.py` and watch GPU usage in Task Manager (Adrenalin overlay)
2. Collect real labeled data
3. Replace dummy datasets with real ones
4. Adjust hyperparameters based on training curves
5. Export models to ONNX or TorchScript for the native host
