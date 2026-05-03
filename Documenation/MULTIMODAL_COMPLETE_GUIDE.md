# Multi-Modal Support - Complete Integration Guide

## Overview

The system now supports classifying multiple input modalities (images, text, audio, video) through a unified interface. Each classifier declares what it can process, and the echo host automatically routes items accordingly.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Detect images and text sections on page              │   │
│  │ 2. Extract images as base64 (Canvas API)               │   │
│  │ 3. Extract text content (DOM traversal)                │   │
│  │ 4. Send as unified item list                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ├─ {id: 'img-0', modality: 'image', url: 'data:...'}   │
│         ├─ {id: 'img-1', modality: 'image', url: 'data:...'}   │
│         ├─ {id: 'text-0', modality: 'text', text: 'Lorem...'}  │
│         └─ {id: 'text-1', modality: 'text', text: 'Ipsum...'}  │
│                                    │                            │
└────────────────────────────────────┼────────────────────────────┘
                                     │
                            native messaging
                                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ECHO HOST (Python)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Receive items from browser                           │   │
│  │ 2. Separate by modality:                                │   │
│  │    - image_items: fetch via HTTP or decode base64       │   │
│  │    - text_items: extract directly (instant)             │   │
│  │    - audio_items: (future)                              │   │
│  │    - video_items: (future)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│    ┌────┴────────────────────────────────────────────┐           │
│    │ Query model registry for classifier support    │           │
│    └────────────────────────────────────────────────┘           │
│         │                                                        │
│    ┌────┴────────────────────────────────────────────┐           │
│    │ Route by modality:                              │           │
│    │                                                 │           │
│    │  res_net50_fft (images)  ←─── image items     │           │
│    │  simple_text (text)      ←─── text items      │           │
│    │  future_audio (audio)    ←─── audio items     │           │
│    │  (parallel processing)                         │           │
│    └────────────────────────────────────────────────┘           │
│         │                                                        │
│    ┌────┴────────────────────────────────────────────┐           │
│    │ Run classification ensemble:                   │           │
│    │                                                 │           │
│    │  1. Load appropriate models (lazy load)        │           │
│    │  2. Mini-batch processing                      │           │
│    │  3. Aggregate weighted scores                  │           │
│    │  4. Unload models (save VRAM)                  │           │
│    └────────────────────────────────────────────────┘           │
│         │                                                        │
│    ┌────┴────────────────────────────────────────────┐           │
│    │ Merge results:                                 │           │
│    │                                                 │           │
│    │  - image results with modality='image'        │           │
│    │  - text results with modality='text'          │           │
│    │  - unified response format                     │           │
│    └────────────────────────────────────────────────┘           │
│         │                                                        │
└─────────┼──────────────────────────────────────────────────────┘
          │
          │ unified result format
          ↓
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER                                   │
│  Results: [                                                      │
│    {id: 'img-0', modality: 'image', label: 'ai', score: 0.87}, │
│    {id: 'img-1', modality: 'image', label: 'real', score: 0.12},│
│    {id: 'text-0', modality: 'text', label: 'real', score: 0.23},│
│    {id: 'text-1', modality: 'text', label: 'ai', score: 0.91}  │
│  ]                                                               │
│                                                                  │
│  Render UI markers (image=border color, text=annotation, etc.)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### BaseClassifier Interface

**Location:** `native/classifiers/base_classifier.py`

Every classifier implements these methods:

```python
class BaseClassifier(ABC):

    @abstractmethod
    def get_supported_modalities(self) -> Set[str]:
        """Declare what this classifier can process."""
        # Return: {'image'}, {'text'}, {'image', 'text'}, etc.

    @abstractmethod
    def load_model(self) -> Tuple[bool, Optional[str]]:
        """Load model weights from disk."""
        # Return: (success_bool, error_string_or_none)

    @abstractmethod
    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        """Prepare inputs for inference."""
        # Args: inputs (PIL.Images, strings, etc.), modality ('image', 'text', etc.)
        # Return: (processed_tensor, valid_indices)

    @abstractmethod
    def classify_batch(self, batch_tensor: Any) -> List[float]:
        """Run inference on preprocessed batch."""
        # Return: [scores in 0.0-1.0 range]

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Report compute device (GPU, CPU, etc.)."""
        # Return: {device, name, backend}

    @abstractmethod
    def get_model_name(self) -> str:
        """Human-readable model name."""
        # Return: "ResNet50-FFT", "SimpleText-Detector", etc.

    # Convenience method (provided by base class)
    def process_batch(self, inputs: List[Any], modality: str) -> List[float]:
        """Combines load/preprocess/classify."""
        # Handles loading, preprocessing, classification, error handling
```

### Current Classifiers

#### 1. ResNet50FFTClassifier

**File:** `native/classifiers/resnet50_fft_classifier.py`
**Supports:** Images only (`{'image'}`)
**Processing:**

- Input: PIL.Image (RGB)
- Preprocess: Resize (224x224) → Normalize → Add FFT channel
- Model: ResNet-50 with 4 input channels (RGB + FFT)
- Output: AI probability (0.0-1.0)
  **Performance:** ~2-3 seconds for 100 images (GPU)

#### 2. SimpleTextClassifier

**File:** `native/classifiers/text_classifier.py`
**Supports:** Text only (`{'text'}`)
**Processing:**

- Input: Text string
- Preprocess: Lowercase, validate
- Detection: Keyword matching + heuristics
  - "as an ai", "I don't have", etc. → high score
  - Short text → moderate score
  - Long text → lower score
- Output: AI probability (0.0-1.0)
  **Performance:** ~1-5ms per 100 texts (CPU)
  **Future:** Can be replaced with BERT, GPT-2 detector, etc.

#### 3. (Future) AudioClassifier

**Supports:** Audio (`{'audio'}`)
**Processing:**

- Decode MP3/WAV
- Extract features (mel-spectrogram, etc.)
- Classification model
  **Not yet implemented**

---

## Echo Host Processing Pipeline

### Phase 1: Item Separation

```python
# Classify items by modality
image_items = [it for it in items if it.get('modality') == 'image' and it.get('url')]
text_items = [it for it in items if it.get('modality') == 'text' and it.get('text')]
other_items = [it for it in items if it not in image_items and it not in text_items]

# Log breakdown
print(f"{len(image_items)} images, {len(text_items)} texts, {len(other_items)} other")
```

### Phase 1a: Fetch Images (Network)

```python
# Fetch in parallel (network is bottleneck)
image_results = fetch_images_from_urls(image_items)
# Returns: [(item_id, PIL.Image, error, duration_ms), ...]

# For base64 URLs (browser-extracted): instant
# For HTTP URLs (from page): ~10ms per image (cached) to ~100ms (uncached)
```

### Phase 1b: Extract Text (Instant)

```python
# No network needed - text already in browser
text_results = fetch_text_from_items(text_items)
# Returns: [(item_id, text_string, None, 0), ...]
# Essentially free - no delay
```

### Phase 2: Initialize Ensemble

```python
ensemble = EnsembleClassifier(
    classifier_ids=['res_net50_fft', 'simple_text'],
    weights=None,  # Equal weight
    lazy_load=True  # Unload after each batch
)
```

### Phase 3: Process Images

```python
for batch_start in range(0, len(image_results), mini_batch_size):
    batch = image_results[batch_start:batch_end]

    # Call ensemble with modality='image'
    batch_results, device_info = ensemble.classify_batch(
        inputs=[img for _, img, _, _ in batch],
        fetch_durations=[dur for _, _, _, dur in batch],
        modality='image'
    )

    # Returns: [{score, label, classifiers, ensemble_size}, ...]
```

### Phase 4: Process Text

```python
for batch_start in range(0, len(text_results), mini_batch_size):
    batch = text_results[batch_start:batch_end]

    # Call ensemble with modality='text'
    batch_results, device_info = ensemble.classify_batch(
        inputs=[text for _, text, _ in batch],
        fetch_durations=[0] * len(batch),  # Text has zero fetch time
        modality='text'
    )

    # Returns: [{score, label, classifiers, ensemble_size}, ...]
```

### Phase 5: Merge Results

```python
# All results have same format:
# {id, modality, label, score, model, durationMs}

# Browser receives unified list:
[
    {id: 'img-0', modality: 'image', label: 'ai', score: 0.87, ...},
    {id: 'text-0', modality: 'text', label: 'real', score: 0.23, ...},
    ...
]
```

---

## Modality Routing Logic

```
Browser Item → Echo Host → Modality Check
                    │
        ┌───────────┴──────────┬──────────────┐
        │                      │              │
    modality='image'      modality='text'   other
        │                      │              │
     url?                    text?       SKIP
      / \                      / \
    YES NO                   YES NO
     │   │                    │   │
     │  SKIP                 SKIP SKIP
     │
  HTTP or base64?
     / \
  HTTP  base64
   │      │
 FETCH  INSTANT
   │      │
   └──→ HAVE PIL IMAGE
        │
   Query Registry:
   "Which classifiers support 'image'?"
        │
   [res_net50_fft, future_hybrid, ...]
        │
   Route to each classifier:
   - res_net50_fft: SUPPORTS → classify
   - other_text_model: NO SUPPORT → skip
        │
   Aggregate Results
        │
   Return {label, score, ensemble_size}
```

---

## Configuration

### Browser Request

```json
{
  "payload": {
    "items": [
      { "id": "img-0", "modality": "image", "url": "data:..." },
      { "id": "text-0", "modality": "text", "text": "Lorem ipsum..." }
    ],
    "model": {
      "classifiers": ["res_net50_fft"], // Can specify any classifiers
      "weights": null, // Or: [0.7, 0.3] for weighted ensemble
      "miniBatchSize": 1000,
      "lazyLoad": true,
      "streamResults": true
    }
  }
}
```

### Echo Host Response

```json
{
  "results": [
    {
      "id": "img-0",
      "modality": "image",
      "label": "ai",
      "score": 0.87,
      "model": "res_net50_fft",
      "durationMs": 1234,
      "classifiers": { "res_net50_fft": 0.87 }
    },
    {
      "id": "text-0",
      "modality": "text",
      "label": "real",
      "score": 0.23,
      "model": "simple_text",
      "durationMs": 5,
      "classifiers": { "simple_text": 0.23 }
    }
  ],
  "errors": [
    {
      "type": "info",
      "message": "Processing 2 item(s) with 2 model(s) | DirectML: AMD Radeon RX 7900 XTX"
    }
  ]
}
```

---

## Adding Support for New Modalities

### Step 1: Create Classifier

```python
# native/classifiers/video_classifier.py

from typing import Set, Tuple, List, Any, Optional, Dict
from .base_classifier import BaseClassifier

class VideoClassifier(BaseClassifier):

    def get_supported_modalities(self) -> Set[str]:
        return {'video'}  # ← Declare support

    def load_model(self) -> Tuple[bool, Optional[str]]:
        # Load video model (e.g., I3D, R3D)
        pass

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        # inputs: list of video URLs or base64
        # modality: 'video'
        # Extract frames, resample, etc.
        pass

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        # Run video model on frames
        pass

    def get_device_info(self) -> Dict[str, Any]:
        pass

    def get_model_name(self) -> str:
        return "I3D-VideoClassifier"
```

### Step 2: Echo Host Automatically Routes

```
Browser sends video items with modality='video'
    ↓
Echo host separates video_items
    ↓
Registry finds VideoClassifier with {'video'} support
    ↓
Ensemble routes to VideoClassifier
    ↓
Results returned with modality='video'
```

### Step 3: No other changes needed!

- No echo host modification
- No registry change (auto-discovers via reflection)
- No browser change (already sends modality)
- Just add the classifier class

---

## Performance Characteristics

### Image Classification

- Fetch: 50-100ms (base64 from browser) or 10-1000ms (HTTP)
- Processing: 2-3 seconds for 100 images (GPU)
- **Total: ~9.5 seconds for 119 images**

### Text Classification

- Fetch: 0ms (instant, in browser)
- Processing: 0.5-5ms per 100 texts (CPU)
- **Total: ~5ms for 100 texts** (even faster than images!)

### Mixed (119 images + 100 texts)

- Phase 1: Fetch images + Extract text (parallel) = ~100ms
- Phase 2: Process images (2-3s) + Process text (5ms) = ~2-3s total
- **Total: ~9.5 seconds (same as image-only!)**
- Text processing is "free" (happens during image GPU wait)

---

## Error Handling

### Image Fetch Failures

- HTTP timeout → mark as None
- Invalid base64 → mark as None
- Classifier scores as -1.0 (error)
- Aggregation ignores -1.0, reports as "uncertain"

### Text Issues

- Empty text → skip
- Non-string → skip
- Classifier errors → -1.0
- Aggregation as above

### Modality Mismatches

- Text sent to image classifier → skipped (logged)
- Image sent to text classifier → skipped (logged)
- Unsupported modality → returns uncertain result

---

## Future Enhancements

1. **Hybrid Classifiers:** CLIP-style models supporting multiple modalities
2. **Cross-Modal Ensembles:** Combine text + image scores for decision
3. **Streaming:** Process items as they arrive (not batched)
4. **Caching:** Cache text analysis (immutable content)
5. **Confidence Scoring:** Per-modality confidence + thresholds
6. **Progressive Loading:** Load classifiers on-demand as items arrive
7. **Custom Preprocessing:** Let classifiers define preferred batch format
8. **Multi-GPU:** Route different modalities to different GPUs
