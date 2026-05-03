# AI Content Detector Training - Quick Start

## Current Status

- ✅ Extension working (classifies items, sends to native host)
- ✅ Native messaging working (batch file wrapper fixed Firefox issues)
- ✅ Training pipeline ready (CPU-based, dummy data)
- ⏸️ GPU support pending (AMD 7900 XTX needs ROCm on Ubuntu 22.04 or cloud GPU)
- ⏸️ Real labeled data needed

## What Just Happened

The training script ran on **dummy random data** to test the pipeline. ~50% accuracy means it's guessing randomly (expected with random labels). You need real labeled samples to train a useful model.

## Next Steps

### 1. Collect Real Data

**Images** (~500-1000 per class):

```
data/images/
  ai_generated/    # DALL-E, Midjourney, Stable Diffusion outputs
  real/            # Unsplash, Pexels, your photos
```

**Text** (~1000-2000 per class):

```
data/text/
  ai_generated.txt  # ChatGPT, Claude outputs (one per line)
  real.txt          # News articles, Reddit, books (one per line)
```

See `datasets.py` for detailed collection tips and data loaders.

### 2. Train on Real Data

Once you have data:

```powershell
cd "C:\Stuff\coding\Test Project\training"
& "C:\Stuff\coding\Test Project\.venv\Scripts\python.exe" train.py --epochs 10 --batch-size 32
```

The script will save models to `models/` when done.

### 3. GPU Training Options

**Option A: Cloud GPU (Easiest)**

- Google Colab (free T4 GPU): https://colab.research.google.com
- Upload training script + data, train there, download models

**Option B: WSL2 + ROCm (Your 7900 XTX)**

- Requires Ubuntu 22.04 (you have 24.04, needs downgrade or wait for ROCm support)
- Once working: massive speedup (hours → minutes)

**Option C: CPU (Current, works fine)**

- Slower but functional
- Good for iteration and small datasets

### 4. Integrate Trained Models

After training, update `native/echo_host.py` to:

1. Load `models/image_classifier.pt` and `models/text_classifier.pt`
2. Run inference on incoming items
3. Return real predictions instead of "uncertain"

## Command Reference

```powershell
# Train with custom settings
python train.py --epochs 5 --batch-size 16 --device cpu

# Show all options
python train.py --help

# Test GPU detection
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

## Current Limitations

- Models trained on dummy data (not useful yet)
- CPU-only (AMD GPU needs ROCm setup)
- No real dataset loaders yet (templates in `datasets.py`)

## Files

- `train.py` - Main training script
- `datasets.py` - Real dataset loader templates + collection tips
- `models/` - Saved models (after training)
- `README.md` - Full setup guide
