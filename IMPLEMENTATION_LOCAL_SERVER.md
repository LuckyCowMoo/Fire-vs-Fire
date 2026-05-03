# Implementation: Local Classification Server

## 5.1 Native Messaging Protocol

The local classification server ([echo_host_V2.py](native/echo_host_V2.py)) communicates with the browser extension via Firefox's Native Messaging API, which uses a binary protocol over stdin/stdout.

### 5.1.1 Message Encoding

Native messaging enforces a strict binary protocol:
1. **Length prefix**: 4-byte unsigned integer (little-endian) indicating message length
2. **JSON payload**: UTF-8 encoded JSON message

```python
def read_message():
    raw_length = sys.stdin.buffer.read(4)
    message_length = struct.unpack('<I', raw_length)[0]
    message = sys.stdin.buffer.read(message_length).decode('utf-8')
    return json.loads(message)

def send_message(message):
    encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('<I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
```

This protocol imposes a **1MB hard limit** per message, necessitating the chunked request system implemented in the extension.

### 5.1.2 Message Types

The server handles four message types:

**PING**: Health check for connection testing
```json
{"type": "ping"}
→ {"ok": true, "time": 1234567890000}
```

**classifyJobChunk**: Chunked classification request (primary workflow)
```json
{
  "type": "classifyJobChunk",
  "jobId": "job-1234567890-abc123",
  "chunkIndex": 0,
  "totalChunks": 3,
  "payload": {
    "items": [...],
    "model": {...}
  }
}
```

**classify**: Legacy single-request classification (deprecated)

**listModels**: Query available classifiers
```json
{"type": "listModels"}
→ {"ok": true, "classifiers": [...], "model_files": [...]}
```

### 5.1.3 Logging Strategy

Since stdout is reserved for native messaging, all debug output is written to stderr or a log file (`native_host_v2.log`). The `log()` function appends timestamped messages to the file without blocking the main thread:

```python
LOG_FILE = Path(__file__).parent.parent / "native_host_v2.log"
def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
```

---

## 5.2 Job Reconstruction System

The `JobAssembler` class reconstructs multi-chunk classification requests sent by the extension when content exceeds the 1MB message limit.

### 5.2.1 Chunk Accumulation

Each job is tracked by a unique `jobId` (e.g., `job-1234567890-abc123`). The assembler maintains a dictionary of in-progress jobs:

```python
self.jobs[job_id] = {
    'created_at': time.time(),
    'total_chunks': total_chunks,
    'received': {},  # chunk_index -> items
    'bytes': 0,
    'model': None
}
```

When a chunk arrives, the assembler:
1. **Validates job existence**: Creates new job entry if first chunk
2. **Checks for duplicates**: Ignores chunks with already-received indices
3. **Enforces size limits**: Rejects jobs exceeding 30MB total (prevents memory exhaustion)
4. **Accumulates items**: Stores chunk items in `received` dictionary keyed by index
5. **Checks completion**: When `len(received) >= total_chunks`, assembles final envelope

### 5.2.2 Chunk Ordering

Chunks may arrive out-of-order due to parallel `Promise.all()` execution in the extension. The assembler reconstructs the original order by sorting chunks by index:

```python
all_items = []
for idx in range(job['total_chunks']):
    all_items.extend(job['received'].get(idx, []))
```

This ensures items maintain their original `itemId` mappings for result correlation.

### 5.2.3 Garbage Collection

Jobs are automatically expired after 120 seconds (configurable TTL) to prevent memory leaks from abandoned requests:

```python
def _cleanup(self):
    now = time.time()
    to_delete = [job_id for job_id, job in self.jobs.items() 
                 if now - job['created_at'] > self.ttl_seconds]
    for job_id in to_delete:
        self.jobs.pop(job_id, None)
```

Cleanup runs before each new chunk is added, ensuring stale jobs don't accumulate.

---

## 5.3 Classifier Registry and Dynamic Discovery

The `ModelRegistry` class ([model_registry.py](native/model_registry.py)) provides dynamic classifier discovery, lazy loading, and lifecycle management.

### 5.3.1 Automatic Discovery

On initialization, the registry scans the `classifiers/` directory for Python files and imports any classes inheriting from `BaseClassifier`:

```python
def discover_classifiers(self):
    classifiers_dir = Path(__file__).parent / 'classifiers'
    for py_file in classifiers_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue
        
        spec = importlib.util.spec_from_file_location(
            f"classifiers.{module_name}", py_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseClassifier) and obj is not BaseClassifier:
                classifier_id = self._generate_id(name)
                self._classifiers[classifier_id] = obj
```

This approach eliminates the need for manual registration—adding a new classifier is as simple as dropping a Python file into the `classifiers/` folder.

### 5.3.2 ID Generation and Display Names

The registry uses a two-tier naming system:

**Classifier IDs** (internal identifiers): Automatically generated from class names using CamelCase-to-snake_case conversion:

- `ResNet50FFTClassifier` → `resnet50_fft`
- `ConvNeXtLargeArtifactClassifier` → `convnext_large_artifact`
- `TextClassifier` → `text`

The algorithm handles consecutive uppercase letters (e.g., "FFT") and numeric suffixes (e.g., "B3") correctly:

```python
def _generate_id(self, class_name: str) -> str:
    name = class_name.replace('Classifier', '')
    name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return name.lower()
```

**Display Names** (user-facing labels): Defined by each classifier's `get_model_name()` method, allowing custom branding:

```python
# In resnet50_fft_classifier.py
def get_model_name(self) -> str:
    return "ResNet50-FFT"

# In text_classifier.py
def get_model_name(self) -> str:
    return "TF-IDF model A"
```

This separation allows the extension to reference classifiers by stable IDs (`resnet50_fft`) while displaying friendly names ("ResNet50-FFT") in the options GUI. The `list_available_details()` method instantiates each classifier to query its display name:

```python
for classifier_id, classifier_class in self._classifiers.items():
    instance = classifier_class(model_path=None)
    name = instance.get_model_name()  # "ResNet50-FFT"
    details.append({'id': classifier_id, 'name': name, ...})
```

### 5.3.3 Lazy Loading and Caching

Classifiers are instantiated on-demand and cached to avoid repeated initialization overhead:

```python
def get_classifier(self, classifier_id: str, model_path: Optional[Path] = None, 
                   lazy_load: bool = True) -> Optional[BaseClassifier]:
    cache_key = f"{classifier_id}:{model_path}"
    if lazy_load and cache_key in self._loaded_instances:
        return self._loaded_instances[cache_key]
    
    classifier_class = self._classifiers[classifier_id]
    instance = classifier_class(model_path=model_path)
    
    if lazy_load:
        self._loaded_instances[cache_key] = instance
    
    return instance
```

The cache key includes the model path, allowing multiple instances of the same classifier with different weights (e.g., for A/B testing).

### 5.3.4 Modality Metadata

The registry exposes classifier capabilities via the `list_available_details()` method, which queries each classifier's supported modalities:

```python
def list_available_details(self) -> List[Dict[str, Any]]:
    details = []
    for classifier_id, classifier_class in self._classifiers.items():
        instance = classifier_class(model_path=None)
        modalities = sorted(list(instance.get_supported_modalities()))
        name = instance.get_model_name()
        details.append({
            'id': classifier_id,
            'name': name,
            'modalities': modalities
        })
    return details
```

This metadata is sent to the extension's options page, which uses it to disable incompatible classifiers in the GUI (e.g., text classifiers are grayed out for image categories).

---

## 5.4 Base Classifier Interface

All classifiers implement the `BaseClassifier` abstract class ([base_classifier.py](native/classifiers/base_classifier.py)), which defines a standardized interface for multi-modal AI detection.

### 5.4.1 Required Methods

**get_supported_modalities()**: Returns a set of supported input types (`{'image'}`, `{'text'}`, `{'image', 'text'}`)

**load_model()**: Loads model weights from disk and initializes inference engine. Returns `(success: bool, error: Optional[str])`

**preprocess_batch()**: Converts raw inputs (PIL Images, text strings) into model-ready tensors. Returns `(batch_tensor, valid_indices)` where `valid_indices` tracks which inputs preprocessed successfully

**classify_batch()**: Runs inference on preprocessed batch. Returns `List[float]` with scores in [0.0, 1.0] where 1.0 = AI-generated, 0.0 = authentic, -1.0 = error

**get_device_info()**: Returns hardware information (`{'device': 'cuda', 'name': 'RTX 4090', 'backend': 'CUDA'}`)

**get_model_name()**: Returns human-readable model name for UI display

### 5.4.2 Optional Methods

**get_label()**: Returns custom badge label (e.g., "Spider" for spider detection). Defaults to "AI generated"

**unload_model()**: Frees model from memory. Default implementation sets `self.model = None`

**process_batch()**: Convenience method combining preprocessing and classification. Automatically handles model loading and error propagation

### 5.4.3 Helper Utilities

The base class provides utility methods to reduce boilerplate:

**_log_to_stderr()**: Safe logging that doesn't corrupt native messaging output

**_try_cuda_then_directml()**: Automatic GPU detection with fallback chain (DirectML → CUDA → CPU)

**_ensure_imports_safely()**: Lazy import helper for avoiding startup delays

Example usage:
```python
class MyClassifier(BaseClassifier):
    def load_model(self):
        if not self._ensure_imports_safely({'torch': 'torch', 'nn': 'torch.nn'}):
            return False, "torch not available"
        
        self.device = self._try_cuda_then_directml(self.torch)
        self.model = self.torch.load(self.model_path).to(self.device)
        self._is_loaded = True
        return True, None
```

### 5.4.4 Score Normalization

All classifiers must return scores in [0.0, 1.0] range where:
- **1.0**: Definitely AI-generated (high confidence)
- **0.5**: Uncertain (classifier cannot determine)
- **0.0**: Definitely authentic (high confidence)
- **-1.0**: Classification error (preprocessing failed, model crashed, etc.)

This standardization allows the ensemble system to combine scores from heterogeneous models without normalization overhead.

---

## 5.5 Ensemble Classification System

The `EnsembleClassifier` class implements weighted averaging of multiple classifiers, allowing users to combine complementary detection strategies.

### 5.5.1 Initialization and Weight Normalization

Ensembles are configured with a list of classifier IDs and optional weights:

```python
ensemble = EnsembleClassifier(
    classifier_ids=['resnet50_fft', 'convnext_large_artifact'],
    weights=[0.6, 0.4],  # ResNet50 weighted 60%, ConvNeXt 40%
    lazy_load=True
)
```

If weights are omitted, equal weighting is applied. Weights are automatically normalized to sum to 1.0:

```python
if weights is None:
    self.weights = [1.0 / len(classifier_ids)] * len(classifier_ids)
else:
    total = sum(weights)
    self.weights = [w / total for w in weights]
```

### 5.5.2 Parallel Classification

The ensemble runs each classifier sequentially (not parallel) to avoid GPU memory contention:

```python
def classify_batch(self, inputs: List[Any], modality: str) -> tuple:
    all_scores = []  # List of (classifier_id, weight, scores_list, label)
    
    for clf_id, weight in zip(self.classifier_ids, self.weights):
        classifier = self.registry.get_classifier(clf_id, lazy_load=not self.lazy_load)
        
        # Check modality support
        if modality not in classifier.get_supported_modalities():
            continue
        
        # Load model if needed
        if not classifier.is_loaded():
            success, error = classifier.load_model()
            if not success:
                continue
        
        # Classify batch
        scores = classifier.process_batch(inputs, modality)
        all_scores.append((clf_id, weight, scores, classifier.get_label()))
        
        # Unload if lazy loading enabled
        if self.lazy_load:
            classifier.unload_model()
```

### 5.5.3 Weighted Averaging

After all classifiers complete, scores are combined using weighted averaging:

```python
for i in range(n_items):
    item_scores = []
    item_weights = []
    
    for clf_id, weight, scores, label in all_scores:
        score = scores[i]
        if score >= 0:  # Valid score (not error)
            item_scores.append(score)
            item_weights.append(weight)
    
    # Normalize weights (in case some classifiers failed)
    weight_sum = sum(item_weights)
    normalized_weights = [w / weight_sum for w in item_weights]
    avg_score = sum(s * w for s, w in zip(item_scores, normalized_weights))
```

This approach is robust to partial failures—if one classifier crashes, the ensemble continues with the remaining classifiers and renormalizes weights accordingly.

### 5.5.4 Inactivity Timeout

To prevent idle models from consuming VRAM, the ensemble tracks the last activity timestamp and automatically unloads models after 15 minutes of inactivity (configurable):

```python
def check_inactivity_timeout(self):
    elapsed_minutes = (time.time() - self.last_activity_time) / 60
    if elapsed_minutes > self.inactivity_timeout_minutes:
        return True
    return False

# In classify_batch():
should_unload = self.lazy_load or self.check_inactivity_timeout()
if should_unload:
    classifier.unload_model()
```

This provides a middle ground between aggressive lazy loading (unload after every batch) and persistent loading (never unload).

---

## 5.6 Data Fetching and Preprocessing

The classification pipeline begins with fetching and decoding content from the extension.

### 5.6.1 Image Fetching

The `fetch_images_from_urls()` function handles two image sources:

**Base64 Data URLs** (preferred): Images extracted from the DOM via canvas are sent as `data:image/png;base64,...` strings. The server decodes them directly:

```python
if url.startswith('data:'):
    header, data_part = url.split(',', 1)
    if ';base64' in header:
        raw = base64.b64decode(data_part)
    else:
        raw = urllib.parse.unquote_to_bytes(data_part)
    img = Image.open(BytesIO(raw)).convert('RGB')
```

**HTTP/HTTPS URLs** (fallback): If canvas extraction fails due to CORS, the server fetches the image from the network:

```python
elif url.startswith(('http://', 'https://')):
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (native-host)',
        'Accept': 'image/*,*/*;q=0.8'
    })
    with urlopen(req, timeout=15) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert('RGB')
```

The function returns 4-tuples: `(item_id, PIL.Image or None, error_reason or None, fetch_duration_ms)`, allowing the orchestrator to track fetch failures and timing.

### 5.6.2 Text Extraction

Text is already extracted by the content script, so `fetch_text_from_items()` simply unpacks it from the JSON payload:

```python
def fetch_text_from_items(text_items: List[Dict[str, Any]]) -> List[tuple]:
    results = []
    for item in text_items:
        item_id = item.get('id')
        text = item.get('text', '')
        results.append((item_id, text, None, 0))
    return results
```

This asymmetry (images require fetching, text doesn't) reflects the different data sources: images are binary blobs that may be cross-origin, while text is always same-origin DOM content.

### 5.6.3 Debug Image Dumping

For debugging preprocessing issues, the server can dump incoming images to disk when the `FIRE_SAVE_INCOMING_IMAGES=1` environment variable is set:

```python
if os.environ.get('FIRE_SAVE_INCOMING_IMAGES') == '1':
    dbg_dir = Path(__file__).parent / 'incoming_debug'
    dbg_dir.mkdir(exist_ok=True)
    out_path = dbg_dir / f"{item_id}_{int(time.time()*1000)}.png"
    with open(out_path, 'wb') as fh:
        fh.write(raw)
```

This allows developers to inspect the exact pixel data received by classifiers, which is critical for diagnosing issues like JPEG artifacts or color space mismatches.

---

## 5.7 Classification Orchestration

The `classify_items()` function orchestrates the entire classification pipeline, handling modality routing, ensemble execution, and result streaming.

### 5.7.1 Item Segregation

Items are first segregated by modality:

```python
image_items = [it for it in items if it.get('modality') == 'image' and it.get('url')]
text_items = [it for it in items if it.get('modality') == 'text' and it.get('text')]
other_items = [it for it in items if it not in image_items and it not in text_items]
```

This allows parallel processing of different modalities (though currently sequential due to GPU memory constraints).

### 5.7.2 Ensemble Configuration Normalization

The server supports both legacy single-ensemble configs and modern multi-ensemble configs:

**Legacy format** (single ensemble per modality):
```json
{
  "classifiers": ["resnet50_fft"],
  "weights": null
}
```

**Modern format** (multiple ensembles per modality):
```json
{
  "ensemblesByModality": {
    "image": [
      {"id": "ai_images", "classifiers": ["resnet50_fft"], "weights": null},
      {"id": "spiders", "classifiers": ["resnet50_spider"], "weights": null}
    ]
  }
}
```

The `_normalize_ensembles()` function converts legacy configs to modern format for uniform processing.

### 5.7.3 Mini-Batch Processing

Large jobs are split into mini-batches (default 1000 items) to prevent GPU memory exhaustion:

```python
for batch_start in range(0, total_images, mini_batch_size):
    batch_end = min(batch_start + mini_batch_size, total_images)
    batch_slice = image_results[batch_start:batch_end]
    batch_images = [img for _, img, _, _ in batch_slice]
    
    batch_results, device_info = ensemble.classify_batch(batch_images, modality='image')
```

This approach allows classification of arbitrarily large jobs (e.g., 10,000 images) without running out of VRAM.

### 5.7.4 Result Streaming

When `streamResults: true` is enabled (default), the server sends results incrementally via `classifyResultChunk` messages:

```python
def emit_chunk(ensemble_id: str, modality: str, chunk_results: List[Dict], chunk_errors: List[Dict]):
    if send_chunk and stream_results:
        send_chunk({
            'type': 'classifyResultChunk',
            'jobId': job_id,
            'ensembleId': ensemble_id,
            'modality': modality,
            'results': chunk_results,
            'errors': chunk_errors
        })
```

This provides immediate user feedback—images are flagged as soon as their ensemble completes, rather than waiting for the entire job to finish.

### 5.7.5 Device Information Logging

On the first mini-batch of each ensemble, the server logs hardware information to the extension console:

```python
if batch_start == 0 and device_info and not device_logged:
    device_msg = f"[Native Host V2] Processing {total_images} image(s) with {len(classifier_ids)} model(s) | "
    device_msg += f"{device_info['backend']}: {device_info['name']}"
    chunk_errors.append({'type': 'info', 'message': device_msg})
```

This appears in the browser console as:
```
[Native Host V2] Processing 47 image(s) with 1 model(s) | DirectML: AMD Radeon RX 7900 XTX
```

Providing transparency about which hardware is being used for classification.

---

## 5.8 Error Handling and Resilience

The server implements multiple layers of error handling to ensure graceful degradation.

### 5.8.1 Fetch Failures

If an image fails to fetch (CORS error, 404, timeout), the server returns a placeholder result with `score: 0.5` and `label: 'uncertain'`:

```python
if fetch_err:
    chunk_results.append({
        'id': item_id,
        'modality': 'image',
        'label': 'uncertain',
        'score': 0.5,
        'notes': fetch_err
    })
```

This prevents a single broken image from blocking the entire batch.

### 5.8.2 Classifier Failures

If a classifier crashes during inference, the ensemble continues with remaining classifiers:

```python
try:
    scores = classifier.process_batch(valid_inputs, modality)
except Exception as e:
    log(f"[Ensemble] {clf_id} failed during process_batch: {e}")
    scores = [-1.0] * len(valid_inputs)
```

The weighted averaging logic automatically excludes `-1.0` scores and renormalizes weights.

### 5.8.3 Main Loop Exception Handling

The main message loop catches all exceptions to prevent the host from crashing:

```python
while True:
    try:
        msg = read_message()
        # ... process message ...
    except Exception as e:
        log(f"[Main] Exception in main loop: {e}")
        log(traceback.format_exc())
```

This ensures the native host remains responsive even if a single request triggers an unhandled exception.

---

## 5.9 Performance Optimizations

### 5.9.1 Lazy Imports

Classifiers use lazy imports to avoid loading heavy dependencies (PyTorch, NumPy, PIL) during host startup:

```python
def load_model(self):
    if not self._ensure_imports_safely({'torch': 'torch', 'torchvision': 'torchvision'}):
        return False, "torch not available"
    # ... rest of load logic ...
```

This reduces startup time from ~5 seconds to <100ms, improving perceived responsiveness.

### 5.9.2 GPU Memory Management

The lazy loading system (`lazyLoad: true`) unloads models immediately after classification:

```python
if self.lazy_load:
    classifier.unload_model()
```

This allows multiple large models to coexist on GPUs with limited VRAM (e.g., 8GB) by ensuring only one model is loaded at a time.

### 5.9.3 Batch Processing

All classifiers process items in batches rather than one-at-a-time, leveraging GPU parallelism:

```python
# Bad: Sequential processing (slow)
for img in images:
    score = model(preprocess(img))

# Good: Batch processing (fast)
batch_tensor = torch.stack([preprocess(img) for img in images])
scores = model(batch_tensor)
```

This provides a **10-50x speedup** on GPUs compared to sequential processing.

---

## 5.10 Summary

The local classification server implements a sophisticated pipeline for multi-modal AI detection:

1. **Native Messaging Protocol**: Binary stdin/stdout communication with 1MB message limit
2. **Job Reconstruction**: Automatic reassembly of chunked requests with garbage collection
3. **Classifier Registry**: Dynamic discovery and lazy loading of detection models
4. **Base Classifier Interface**: Standardized API for multi-modal detection with helper utilities
5. **Ensemble System**: Weighted averaging of multiple classifiers with automatic failure handling
6. **Data Fetching**: Dual-source image loading (base64/HTTP) with error tracking
7. **Classification Orchestration**: Modality routing, mini-batch processing, and result streaming
8. **Error Handling**: Multi-layer resilience ensuring graceful degradation
9. **Performance Optimizations**: Lazy imports, GPU memory management, and batch processing

The modular architecture allows new classifiers to be added by simply dropping a Python file into the `classifiers/` folder, with automatic discovery, GUI integration, and ensemble support—no code changes required in the orchestrator or extension.
