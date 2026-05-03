# Echo Host V2 - Modular Multi-Model System

## Overview

The new echo_host_V2.py implements a modular, extensible architecture for AI detection with support for:

- **Multiple Models**: Run multiple classifiers and average their results
- **Pluggable Classifiers**: Add new models by dropping files in `classifiers/` folder
- **Mini-Batch Processing**: Configurable batch sizes for responsive UX
- **Weighted Ensembles**: Assign different weights to different models
- **Lazy Loading**: Automatically manage VRAM by unloading models when not in use

## Architecture

```
echo_host_V2.py (orchestrator)
    ├─ model_registry.py (discovers & manages classifiers)
    └─ classifiers/
        ├─ base_classifier.py (abstract interface)
        └─ resnet50_fft_classifier.py (your current model)
```

## Key Changes from V1

### 1. Images as PIL Objects (Not URLs)

- **V1**: Classifiers received URLs and handled fetching
- **V2**: Orchestrator fetches images once, passes PIL.Image objects to classifiers
- **Benefit**: Multiple classifiers can reuse the same fetched image

### 2. One Image at a Time? NO - Mini-Batches!

Your concern about performance was correct. Instead of processing one-by-one:

- **Images are fetched first** (network is slowest, so batch this)
- **Then processed in mini-batches** (default: 10 images)
- **GPU efficiency preserved**: Each classifier still processes batches
- **Progressive UX**: User sees updates every mini-batch instead of waiting for all images

**Performance comparison:**

```
One-by-one:    10 images × 200ms = 2000ms total
Mini-batch(5): (5×200ms) + (5×200ms) = 400ms + 400ms = 800ms total ✓
Full batch:    10 images = 200ms total (fastest but no progressive updates)
```

### 3. Modular Classifier System

Each classifier is a separate file with standard interface:

- `load_model()`: Load weights
- `preprocess_batch(images)`: Convert PIL Images to tensors
- `classify_batch(tensors)`: Run inference
- `get_device_info()`: Report GPU/CPU
- `get_model_name()`: Model identifier

### 4. Ensemble Averaging

When multiple models are configured:

```python
{
  "classifiers": ["resnet50_fft", "efficientnet_b3"],
  "weights": [0.7, 0.3],  # Trust ResNet50 more
  "miniBatchSize": 10
}
```

Final score = (0.7 × ResNet50_score) + (0.3 × EfficientNet_score)

### 5. Lazy Loading

When `lazyLoad: true`:

1. Load Model 1 → Classify all images → Unload Model 1
2. Load Model 2 → Classify all images → Unload Model 2
3. Average scores

**VRAM usage**: Only ONE model in memory at a time
**Trade-off**: Slower than keeping all models loaded, but enables using multiple large models

## Configuration

### Browser Extension Payload

```json
{
  "type": "classify",
  "payload": {
    "items": [...],
    "model": {
      "classifiers": ["resnet50_fft"],  // List of classifier IDs
      "weights": [1.0],                 // Optional: relative weights
      "miniBatchSize": 10,              // Images per mini-batch
      "lazyLoad": true                  // Unload between classifiers
    }
  }
}
```

### Options UI Integration

Add to options page:

- **Model Selection**: Dropdown with available classifiers
- **Ensemble Mode**: Checkbox to enable multiple models
- **Mini-Batch Size**: Slider (1-50, default 10)
- **Weights**: Input fields when ensemble enabled

## Adding New Classifiers

1. Create new file in `native/classifiers/` (e.g., `efficientnet_classifier.py`)
2. Inherit from `BaseClassifier`
3. Implement required methods:
   ```python
   class EfficientNetClassifier(BaseClassifier):
       def load_model(self): ...
       def preprocess_batch(self, images): ...
       def classify_batch(self, batch_tensor): ...
       def get_device_info(self): ...
       def get_model_name(self): return "EfficientNet-B3"
   ```
4. **That's it!** Registry auto-discovers on startup

## Testing

Run the test script:

```powershell
cd "c:\Stuff\coding\Test Project\native"
python test_v2.py
```

Tests:

1. ✓ Model registry discovers classifiers
2. ✓ ResNet50-FFT loads successfully
3. ✓ Classification produces valid scores

## Migration from V1

### Native Manifest

Update `manifest.json` to point to V2:

```json
{
  "path": "c:\\Stuff\\coding\\Test Project\\native\\echo_host_V2.py"
}
```

### Browser Extension

**No changes needed!** V2 is backward compatible:

- Old format: `"model": "resnet50_fft"` → Auto-converted to `["resnet50_fft"]`
- Old results format: Still supported

### Gradual Migration

1. Test V2 with single model (same as V1)
2. Add mini-batch size option to UI
3. Add ensemble support when ready

## Performance Tuning

### For Responsiveness (quick updates)

```json
{
  "miniBatchSize": 5, // Smaller batches = faster first results
  "lazyLoad": true // Less VRAM
}
```

### For Speed (fastest total time)

```json
{
  "miniBatchSize": 50, // Larger batches = better GPU utilization
  "lazyLoad": false // Keep models loaded (if VRAM allows)
}
```

### For Multiple Models

```json
{
  "classifiers": ["resnet50_fft", "model2", "model3"],
  "weights": [0.5, 0.3, 0.2], // Confidence-weighted
  "miniBatchSize": 10,
  "lazyLoad": true // Required for 3+ large models
}
```

## Next Steps

1. **Test V2** with your existing model
2. **Update browser extension** to send `miniBatchSize` config
3. **Train additional models** (different architectures)
4. **Implement ensemble UI** in options page
5. **Fine-tune weights** based on model accuracy

## File Structure

```
native/
├── echo_host.py              # Original (keep for fallback)
├── echo_host_V2.py           # New orchestrator ✨
├── model_registry.py         # Classifier discovery
├── ensemble_config.json      # Configuration examples
├── test_v2.py                # Test suite
└── classifiers/
    ├── __init__.py
    ├── base_classifier.py           # Abstract interface
    └── resnet50_fft_classifier.py   # Your current model
```

## Questions Answered

**Q: Will processing images one-by-one slow things down?**
A: Yes! That's why we use mini-batches. You get progressive UX (updates every 5-10 images) while preserving GPU efficiency.

**Q: Will VRAM usage increase with multiple models?**
A: With `lazyLoad: true`, VRAM usage stays the same (only one model at a time). Without lazy loading, yes, all models load simultaneously.

**Q: What's a PIL Image?**
A: Python Imaging Library (Pillow) - it's the standard Python image type. Your code already uses it (`from PIL import Image`).

**Q: How does lazy loading work?**
A: Loads model → runs all images → unloads model → loads next model. Trades speed for memory.

**Q: What's weighted averaging?**
A: Instead of simple mean, you can trust certain models more:

- Simple: (Model1 + Model2) / 2
- Weighted: (0.7×Model1) + (0.3×Model2) if Model1 is more accurate

## Support

Check logs in `native_host_v2.log` for detailed execution traces.
