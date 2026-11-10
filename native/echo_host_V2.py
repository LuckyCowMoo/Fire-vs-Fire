#!.venv311/Scripts/python.exe
"""
PYTHON ENVIRONMENT: .venv311 (Python 3.11 with GPU support via torch-directml)
LOG FILE: ../native_host_v2.log
"""

import sys
import json
import struct
import time
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import base64
import urllib.parse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os

# Setup logging to file
LOG_FILE = Path(__file__).parent.parent / "native_host_v2.log"
def log(msg):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except:
        pass

log(f"[START] Echo Host V2 starting")

# Global set to track cancelled jobs
cancelled_jobs = set()

# Global classification cache (RAM-based, max 5000 items)
# Maps (item_id, ensemble_id) -> {score, label, display_label, classifiers, timestamp}
classification_cache = {}
MAX_CACHE_SIZE = 5000

# Import classifier registry
from model_registry import get_registry

# ============================================================================
# Native Messaging Protocol
# ============================================================================

# Preserve raw stdout binary buffer for Native Messaging protocol
_raw_stdout = sys.stdout.buffer
# Redirect text stdout to stderr so ANY accidental print() never corrupts native messaging!
sys.stdout = sys.stderr

def read_message():
    # Read 4-byte message length
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length or len(raw_length) < 4:
        return None
    message_length = struct.unpack('<I', raw_length)[0]

    # Read message_length bytes in a loop (handles partial reads on Windows pipes)
    chunks = []
    bytes_read = 0
    while bytes_read < message_length:
        to_read = min(message_length - bytes_read, 65536)
        chunk = sys.stdin.buffer.read(to_read)
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)

    if bytes_read < message_length:
        log(f"[Native] Incomplete message read: expected {message_length}, got {bytes_read}")
        return None

    raw_message = b''.join(chunks).decode('utf-8')
    return json.loads(raw_message)

def send_message(message):
    encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    # Firefox strict 1MB native messaging limit: truncate gracefully so port NEVER disconnects
    max_safe_bytes = 750 * 1024
    if len(encoded) >= max_safe_bytes and isinstance(message, dict) and 'results' in message and isinstance(message['results'], list):
        log(f"[Native] Warning: message size ({len(encoded)} bytes) exceeds safe limit; truncating results to protect connection")
        while len(encoded) >= max_safe_bytes and len(message['results']) > 1:
            message['results'] = message['results'][:len(message['results']) // 2]
            encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    _raw_stdout.write(struct.pack('<I', len(encoded)))
    _raw_stdout.write(encoded)
    _raw_stdout.flush()


# ============================================================================
# Job Reconstruction (Chunked Requests)
# ============================================================================

class JobAssembler:
    def __init__(self, ttl_seconds: int = 300, max_bytes: int = 250 * 1024 * 1024):
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.jobs = {}

    def _cleanup(self):
        now = time.time()
        to_delete = []
        for job_id, job in self.jobs.items():
            if now - job['created_at'] > self.ttl_seconds:
                to_delete.append(job_id)
        for job_id in to_delete:
            log(f"[JobAssembler] Expired job {job_id}")
            self.jobs.pop(job_id, None)

    def add_chunk(self, job_id: str, chunk_index: int, total_chunks: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a chunk to a job. Returns assembled envelope when complete.
        """
        self._cleanup()

        if job_id not in self.jobs:
            self.jobs[job_id] = {
                'created_at': time.time(),
                'total_chunks': total_chunks,
                'received': {},
                'bytes': 0,
                'model': None
            }

        job = self.jobs[job_id]

        # Update total_chunks if provided later
        if total_chunks and total_chunks > 0:
            job['total_chunks'] = total_chunks

        # Ignore duplicate chunk
        if chunk_index in job['received']:
            return None

        items = payload.get('items') or []
        model = payload.get('model') or None

        try:
            chunk_bytes = len(json.dumps(items))
        except Exception:
            chunk_bytes = 0

        if job['bytes'] + chunk_bytes > self.max_bytes:
            log(f"[JobAssembler] Job {job_id} exceeded max size; dropping")
            self.jobs.pop(job_id, None)
            return None

        job['received'][chunk_index] = items
        job['bytes'] += chunk_bytes
        if model and job['model'] is None:
            job['model'] = model

        if len(job['received']) >= job['total_chunks']:
            # Assemble in order
            all_items = []
            for idx in range(job['total_chunks']):
                all_items.extend(job['received'].get(idx, []))

            envelope = {
                'version': 2,
                'type': 'classify',
                'requestId': job_id,
                'timestamp': int(time.time() * 1000),
                'payload': {
                    'items': all_items,
                    'model': job['model'] or {}
                }
            }
            self.jobs.pop(job_id, None)
            return envelope

        return None


# ============================================================================
# IMAGE & TEXT FETCHING
# ============================================================================
# Phase 1 of classify_items() pipeline: Prepare input data

def fetch_text_from_items(text_items: List[Dict[str, Any]]) -> List[tuple]:

    # Extract text from content script

    # Text is already in browser (no network fetch needed), just needs extraction
    # Returns 4-tuples: (item_id, text_string, None, 0_ms)
    
    results = []
    for item in text_items:
        item_id = item.get('id')
        text = item.get('text', '')
        results.append((item_id, text, None, 0))
    return results


def _fetch_single_image(item: Dict[str, Any]) -> tuple:
    item_id = item.get('id')
    url = item.get('url')
    t0 = time.time()
    if not url:
        return (item_id, None, 'errorus/no_url', 0)
    try:
        if url.startswith('data:'):
            header, data_part = url.split(',', 1)
            if ';base64' in header:
                raw = base64.b64decode(data_part)
            else:
                raw = urllib.parse.unquote_to_bytes(data_part)
            img = Image.open(BytesIO(raw)).convert('RGB')
            return (item_id, img, None, int((time.time() - t0) * 1000))
        elif url.startswith(('http://', 'https://')):
            req = Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/*,*/*;q=0.8'
            })
            with urlopen(req, timeout=3.5) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data)).convert('RGB')
            return (item_id, img, None, int((time.time() - t0) * 1000))
        else:
            return (item_id, None, 'invalid_url_scheme', 0)
    except Exception as e:
        return (item_id, None, f'fetch_failed: {str(e)[:40]}', int((time.time() - t0) * 1000))

def fetch_images_from_urls(image_items: List[Dict[str, Any]]) -> List[tuple]:
    if not image_items:
        return []
    # If all items are base64, process in-thread directly for zero threading overhead
    all_base64 = all(bool(it.get('url', '').startswith('data:')) for it in image_items)
    if all_base64 or len(image_items) == 1:
        return [_fetch_single_image(it) for it in image_items]
    
    # Otherwise fetch concurrently with ThreadPoolExecutor so network delays never block host
    with ThreadPoolExecutor(max_workers=min(6, len(image_items))) as executor:
        results = list(executor.map(_fetch_single_image, image_items))
    return results


# ----------------------------------------------------------------------------
# Multi-Model Classification with Ensemble Averaging
# ----------------------------------------------------------------------------

class EnsembleClassifier:
    
    def __init__(self, classifier_ids: List[str], weights: Optional[List[float]] = None, lazy_load: bool = True, inactivity_timeout_minutes: int = 15):
        """
        Initialize ensemble.
        
        Args:
            classifier_ids: List of classifier IDs (e.g., ['resnet50_fft'])
            weights: Optional weights for each classifier (must sum to 1.0)
            lazy_load: If True, unload models after use to save VRAM
            inactivity_timeout_minutes: Unload models if unused for this many minutes
        """
        self.registry = get_registry()
        self.classifier_ids = classifier_ids
        self.lazy_load = lazy_load
        self.inactivity_timeout_minutes = inactivity_timeout_minutes
        self.last_activity_time = time.time()  # Track inactivity for timeout
        
        # Normalize weights
        if weights is None:
            self.weights = [1.0 / len(classifier_ids)] * len(classifier_ids)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        log(f"[Ensemble] Initialized with {len(classifier_ids)} classifiers")
        log(f"[Ensemble] Weights: {dict(zip(classifier_ids, self.weights))}")
        log(f"[Ensemble] Inactivity timeout: {inactivity_timeout_minutes} minutes")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_time = time.time()
    
    def check_inactivity_timeout(self):
        """Check if inactivity timeout has been exceeded. Returns True if should unload."""
        elapsed_minutes = (time.time() - self.last_activity_time) / 60
        if elapsed_minutes > self.inactivity_timeout_minutes:
            log(f"[Ensemble] Inactivity timeout triggered: {elapsed_minutes:.1f} minutes > {self.inactivity_timeout_minutes}")
            return True
        return False
    
    def classify_batch(self, inputs: List[Any], modality: str = 'image') -> tuple:
        """
        Classify a batch of items using ensemble.
        
        Args:
            inputs: List of inputs (PIL Images, text strings, etc.)
            modality: Type of input ('image', 'text', 'audio', 'video')
        
        Returns:
            Tuple of (results_list, device_info_dict)
        """
        self.update_activity()  # Update activity timestamp
        
        n_items = len(inputs)
        all_scores = []  # List of (classifier_id, weight, scores_list, label)
        device_info = None
        
        # Run each classifier
        for clf_id, weight in zip(self.classifier_ids, self.weights):
            log(f"[Ensemble] Running classifier: {clf_id} (weight={weight:.3f})")
            
            classifier = self.registry.get_classifier(clf_id, lazy_load=not self.lazy_load)
            if classifier is None:
                log(f"[Ensemble] Failed to load {clf_id}, skipping")
                continue
            
            # Check if classifier supports this modality
            supported = classifier.get_supported_modalities()
            if modality not in supported:
                log(f"[Ensemble] {clf_id} doesn't support modality '{modality}', skipping")
                continue
            
            # Load model if needed
            if not classifier.is_loaded():
                log(f"[Ensemble] Loading model {clf_id}...")
                load_start = time.time()
                success, error = classifier.load_model()
                load_ms = int((time.time() - load_start) * 1000)
                if not success:
                    log(f"[Ensemble] {clf_id} load failed: {error}")
                    continue
                log(f"[Ensemble] {clf_id} loaded in {load_ms}ms")

            try:
                classifier_label = classifier.get_label()
            except Exception:
                classifier_label = 'AI generated'
            if not classifier_label:
                classifier_label = 'AI generated'
            
            # Get device info from first loaded classifier
            if device_info is None:
                device_info = classifier.get_device_info()
            
            # Filter out None/empty inputs (fetch failures)
            if modality == 'image':
                valid_inputs = [inp for inp in inputs if inp is not None]
            else:  # text, audio, video, etc.
                valid_inputs = [inp for inp in inputs if inp is not None and inp != '']
            
            valid_indices = [i for i, inp in enumerate(inputs) if (inp is not None and (modality != 'image' or inp is not None) and (modality != 'text' or inp != ''))]
            
            if not valid_inputs:
                log(f"[Ensemble] No valid inputs for {clf_id} (modality={modality})")
                continue
            
            # Classify batch
            t0 = time.time()
            try:
                # Use process_batch which handles preprocess + classify
                scores = classifier.process_batch(valid_inputs, modality)
                if scores is None:
                    log(f"[Ensemble] {clf_id} returned None scores")
                    scores = [-1.0] * len(valid_inputs)
            except Exception as e:
                log(f"[Ensemble] {clf_id} failed during process_batch: {e}")
                import traceback
                log(traceback.format_exc())
                scores = [-1.0] * len(valid_inputs)
            
            inference_ms = int((time.time() - t0) * 1000)
            
            log(f"[Ensemble] {clf_id} processed {len(valid_inputs)} {modality}(s) in {inference_ms}ms (total including preprocess)")
            
            # Map scores back to full batch (fill errors with -1.0)
            full_scores = [-1.0] * n_items
            for idx, score in zip(valid_indices, scores):
                full_scores[idx] = score
            
            all_scores.append((clf_id, weight, full_scores, classifier_label))
            
            # Unload if lazy loading enabled OR inactivity timeout exceeded
            should_unload = self.lazy_load or self.check_inactivity_timeout()
            if should_unload:
                classifier.unload_model()
                if self.lazy_load:
                    log(f"[Ensemble] Unloaded {clf_id} (lazy load enabled)")
                else:
                    log(f"[Ensemble] Unloaded {clf_id} (inactivity timeout)")
        
        # Compute weighted average
        results = []
        for i in range(n_items):
            # Collect scores from all classifiers for this item
            item_scores = []
            item_weights = []
            classifier_details = {}
            
            classifier_labels = []

            for clf_id, weight, scores, classifier_label in all_scores:
                score = scores[i]
                if score >= 0:  # Valid score
                    item_scores.append(score)
                    item_weights.append(weight)
                    classifier_details[clf_id] = {
                        'score': round(score, 4),
                        'label': classifier_label or 'AI generated'
                    }
                    if classifier_label:
                        classifier_labels.append(classifier_label)
            
            # Calculate weighted average
            if item_scores:
                # Normalize weights (in case some classifiers failed)
                weight_sum = sum(item_weights)
                if weight_sum > 0:
                    normalized_weights = [w / weight_sum for w in item_weights]
                    avg_score = sum(s * w for s, w in zip(item_scores, normalized_weights))
                else:
                    avg_score = 0.5
                
                label = 'ai' if avg_score >= 0.5 else 'real'
                display_label = next(
                    (name for name in classifier_labels if name and name != 'AI generated'),
                    (classifier_labels[0] if classifier_labels else 'AI generated')
                )
                
                results.append({
                    'score': round(avg_score, 4),
                    'label': label,
                    'display_label': display_label,
                    'classifiers': classifier_details,
                    'ensemble_size': len(item_scores)
                })
            else:
                # All classifiers failed
                results.append({
                    'score': 0.5,
                    'label': 'uncertain',
                    'display_label': 'AI generated',
                    'classifiers': {},
                    'ensemble_size': 0
                })
        
        return results, device_info


# ----------------------------------------------------------------------------
# Main Classification Pipeline
# ----------------------------------------------------------------------------

def _normalize_ensembles(model_config: Dict[str, Any], modality: str) -> List[Dict[str, Any]]:
    ensembles_by_modality = model_config.get('ensemblesByModality') or {}
    ensembles = ensembles_by_modality.get(modality) or []

    if not isinstance(ensembles, list):
        ensembles = []

    if ensembles:
        return ensembles

    # Back-compat: single ensemble from legacy classifiers list
    classifier_ids = model_config.get('classifiers', ['res_net50_fft', 'text'])
    if isinstance(classifier_ids, str):
        classifier_ids = [classifier_ids]

    return [{
        'id': f'default_{modality}',
        'classifiers': classifier_ids,
        'weights': model_config.get('weights', None)
    }]


def _manage_cache_size():
    """Remove oldest entries if cache exceeds max size (FIFO eviction)."""
    global classification_cache
    evicted = 0
    while len(classification_cache) > MAX_CACHE_SIZE:
        oldest_key = next(iter(classification_cache))
        classification_cache.pop(oldest_key, None)
        evicted += 1
    if evicted:
        log(f"[Cache] Evicted {evicted} oldest entries, cache size now {len(classification_cache)}")


def classify_items(envelope, job_id: Optional[str] = None, send_chunk=None):
    """
    Main classification orchestrator with mini-batch support.

    If send_chunk is provided and streamResults is enabled, results are streamed
    per-ensemble as soon as each ensemble completes.
    """
    req_id = envelope.get('requestId')
    job_key = job_id or req_id
    
    # Check if job was cancelled before starting
    if job_key in cancelled_jobs:
        log(f"[classify_items] Job {job_key} was cancelled; aborting")
        cancelled_jobs.discard(job_key)
        return {
            'version': 2,
            'type': 'classifyResult',
            'requestId': req_id,
            'timestamp': int(time.time() * 1000),
            'results': [],
            'errors': [{'type': 'cancelled', 'message': 'Classification was cancelled'}]
        }
    
    payload = envelope.get('payload') or {}
    items = payload.get('items') or []
    model_config = payload.get('model') or {}

    weights = model_config.get('weights', None)
    raw_mini_batch = model_config.get('miniBatchSize', 16)
    # Cap image mini-batch size to 16 when streaming so pages with 80+ images
    # stream badges progressively to the browser every ~100ms instead of blocking.
    mini_batch_size = max(1, min(int(raw_mini_batch or 16), 16))
    lazy_load = model_config.get('lazyLoad', True)
    stream_results = model_config.get('streamResults', True)

    log(f"[classify_items] Starting with {len(items)} items")

    # Sanitize first item for logging (remove base64 image data)
    if items:
        sample = items[0].copy()
        if sample.get('modality') == 'image' and sample.get('url'):
            url = sample['url']
            if url.startswith('data:'):
                mime_part = url.split(';')[0]
                sample['url'] = f"{mime_part};base64,<truncated>"
        log(f"[classify_items] First item sample: {sample}")
    else:
        log(f"[classify_items] First item sample: none")

    items_json_size = len(json.dumps(items))
    log(f"[classify_items] Request items JSON size: {items_json_size} bytes ({items_json_size / (1024*1024):.2f} MB)")
    log(f"[classify_items] Mini-batch size: {mini_batch_size}, lazy_load: {lazy_load}, stream: {stream_results}")

    results = []
    errors = []

    # Separate items by modality
    image_items = [it for it in items if it.get('modality') == 'image' and it.get('url')]
    text_items = [it for it in items if it.get('modality') == 'text' and it.get('text')]
    other_items = [it for it in items if it not in image_items and it not in text_items]

    log(f"[classify_items] {len(image_items)} image items, {len(text_items)} text items, {len(other_items)} other items")
    
    # Check cache for already-classified items (right before processing starts)
    # This catches rapid page refreshes where requests queue up
    cached_results = []
    cache_hits = 0
    
    # Get ensemble IDs for cache lookup
    image_ensemble_ids = [e.get('id', 'image_ensemble') for e in _normalize_ensembles(model_config, 'image')]
    text_ensemble_ids = [e.get('id', 'text_ensemble') for e in _normalize_ensembles(model_config, 'text')]
    
    # Filter out cached items from image_items
    uncached_image_items = []
    for item in image_items:
        item_id = item.get('id')
        if not item_id:
            uncached_image_items.append(item)
            continue
        
        # Check if ALL requested ensembles have cached results for this item
        all_ensembles_cached = True
        for ensemble_id in image_ensemble_ids:
            cache_key = (item_id, ensemble_id)
            if cache_key in classification_cache:
                cached_data = classification_cache[cache_key]
                cached_results.append({
                    'id': item_id,
                    'modality': 'image',
                    'ensembleId': ensemble_id,
                    'label': cached_data.get('label', 'uncertain'),
                    'score': cached_data.get('score', 0.5),
                    'display_label': cached_data.get('display_label', 'AI generated'),
                    'classifiers': cached_data.get('classifiers', {}),
                    'model': cached_data.get('model', 'cached'),
                    'durationMs': 0,
                    'cached': True
                })
                cache_hits += 1
            else:
                all_ensembles_cached = False
        
        # Only skip processing if ALL ensembles have cached results
        if not all_ensembles_cached:
            uncached_image_items.append(item)
    
    # Filter out cached items from text_items
    uncached_text_items = []
    for item in text_items:
        item_id = item.get('id')
        if not item_id:
            uncached_text_items.append(item)
            continue
        
        # Check if ALL requested ensembles have cached results for this item
        all_ensembles_cached = True
        for ensemble_id in text_ensemble_ids:
            cache_key = (item_id, ensemble_id)
            if cache_key in classification_cache:
                cached_data = classification_cache[cache_key]
                cached_results.append({
                    'id': item_id,
                    'modality': 'text',
                    'ensembleId': ensemble_id,
                    'label': cached_data.get('label', 'uncertain'),
                    'score': cached_data.get('score', 0.5),
                    'display_label': cached_data.get('display_label', 'AI generated'),
                    'classifiers': cached_data.get('classifiers', {}),
                    'model': cached_data.get('model', 'cached'),
                    'durationMs': 0,
                    'cached': True
                })
                cache_hits += 1
            else:
                all_ensembles_cached = False
        
        # Only skip processing if ALL ensembles have cached results
        if not all_ensembles_cached:
            uncached_text_items.append(item)
    
    # Replace with uncached items only
    image_items = uncached_image_items
    text_items = uncached_text_items
    
    if cache_hits > 0:
        log(f"[Cache] {cache_hits} items found in cache, {len(image_items)} images and {len(text_items)} text items need classification")
        # Send cached results as chunk if streaming, otherwise add to results array
        if send_chunk and stream_results:
            # Send cached results in batches to avoid overwhelming the browser
            batch_size = 20
            for i in range(0, len(cached_results), batch_size):
                batch = cached_results[i:i+batch_size]
                send_chunk({
                    'version': 2,
                    'type': 'classifyResultChunk',
                    'jobId': job_id or req_id,
                    'ensembleId': 'cached',
                    'modality': 'mixed',
                    'timestamp': int(time.time() * 1000),
                    'results': batch,
                    'errors': []
                })
        else:
            results.extend(cached_results)

    if not image_items and not text_items:
        # All items cached - add info message
        errors.append({'type': 'info', 'message': f'All {cache_hits} items returned from cache'})

    image_results = []
    if image_items:
        log(f"[classify_items] Fetching {len(image_items)} images...")
        fetch_start = time.time()
        image_results = fetch_images_from_urls(image_items)
        fetch_ms = int((time.time() - fetch_start) * 1000)

        successful_fetches = sum(1 for _, img, _, _ in image_results if img is not None)
        log(f"[classify_items] Fetched {successful_fetches}/{len(image_items)} images in {fetch_ms}ms")
        errors.append({'type': 'fetch_timing', 'count': len(image_items),
                       'successful': successful_fetches, 'durationMs': fetch_ms})

    text_results = []
    if text_items:
        log(f"[classify_items] Extracting {len(text_items)} text items...")
        text_results = fetch_text_from_items(text_items)
        log(f"[classify_items] Extracted {len(text_results)} text items")

    inactivity_timeout = model_config.get('inactivityTimeout', 15)

    def emit_chunk(ensemble_id: str, modality: str, chunk_results: List[Dict[str, Any]], chunk_errors: List[Dict[str, Any]]):
        if send_chunk and stream_results:
            send_chunk({
                'version': 2,
                'type': 'classifyResultChunk',
                'jobId': job_id or req_id,
                'ensembleId': ensemble_id,
                'modality': modality,
                'timestamp': int(time.time() * 1000),
                'results': chunk_results,
                'errors': chunk_errors
            })
            return True
        return False

    # Process image ensembles
    if image_results:
        image_ensembles = _normalize_ensembles(model_config, 'image')
        total_images = len(image_results)

        for ensemble_cfg in image_ensembles:
            ensemble_id = ensemble_cfg.get('id') or 'image_ensemble'
            classifier_ids = ensemble_cfg.get('classifiers') or model_config.get('classifiers', [])
            if isinstance(classifier_ids, str):
                classifier_ids = [classifier_ids]
            ensemble_weights = ensemble_cfg.get('weights', weights)

            if not classifier_ids:
                log(f"[classify_items] Ensemble '{ensemble_id}' has no classifiers; skipping")
                continue

            log(f"[classify_items] Running image ensemble '{ensemble_id}' with {len(classifier_ids)} classifier(s)")
            ensemble = EnsembleClassifier(classifier_ids, ensemble_weights, lazy_load, inactivity_timeout)

            chunk_errors = []
            device_logged = False

            for batch_start in range(0, total_images, mini_batch_size):
                # Check if job was cancelled before processing this batch
                if job_key in cancelled_jobs:
                    log(f"[classify_items] Image batch cancelled for job {job_key}; stopping")
                    cancelled_jobs.discard(job_key)
                    break
                
                batch_end = min(batch_start + mini_batch_size, total_images)
                batch_slice = image_results[batch_start:batch_end]
                batch_ids = [item_id for item_id, _, _, _ in batch_slice]
                batch_images = [img for _, img, _, _ in batch_slice]
                batch_errors = [err for _, _, err, _ in batch_slice]

                t0 = time.time()
                batch_results, device_info = ensemble.classify_batch(batch_images, modality='image')
                classify_ms = int((time.time() - t0) * 1000)

                if batch_start == 0 and device_info and not device_logged:
                    device_logged = True
                    device_msg = f"[Native Host V2] Processing {total_images} image(s) with {len(classifier_ids)} model(s) | "
                    device_msg += f"{device_info['backend']}: {device_info['name']}"
                    chunk_errors.append({'type': 'info', 'message': device_msg, 'ensembleId': ensemble_id, 'modality': 'image'})

                chunk_results = []
                for idx, (item_id, result_data) in enumerate(zip(batch_ids, batch_results)):
                    fetch_err = batch_errors[idx]
                    if fetch_err:
                        result = {
                            'id': item_id,
                            'modality': 'image',
                            'label': 'uncertain',
                            'score': 0.5,
                            'display_label': 'AI generated',
                            'model': ','.join(classifier_ids),
                            'notes': fetch_err
                        }
                    else:
                        result = {
                            'id': item_id,
                            'modality': 'image',
                            'label': result_data['label'],
                            'score': result_data['score'],
                            'display_label': result_data.get('display_label', 'AI generated'),
                            'classifiers': result_data.get('classifiers', {}),
                            'model': ','.join(classifier_ids),
                            'durationMs': int(classify_ms / max(1, len(batch_slice)))
                        }
                        # Cache successful classification
                        if item_id:
                            cache_key = (item_id, ensemble_id)
                            classification_cache[cache_key] = {
                                'label': result_data['label'],
                                'score': result_data['score'],
                                'display_label': result_data.get('display_label', 'AI generated'),
                                'classifiers': result_data.get('classifiers', {}),
                                'model': ','.join(classifier_ids),
                                'timestamp': time.time()
                            }
                            _manage_cache_size()
                    chunk_results.append(result)

                if not emit_chunk(ensemble_id, 'image', chunk_results, chunk_errors):
                    results.extend(chunk_results)
                    errors.extend(chunk_errors)

    # Process text ensembles
    if text_results:
        text_ensembles = _normalize_ensembles(model_config, 'text')
        total_text = len(text_results)

        for ensemble_cfg in text_ensembles:
            ensemble_id = ensemble_cfg.get('id') or 'text_ensemble'
            classifier_ids = ensemble_cfg.get('classifiers') or model_config.get('classifiers', [])
            if isinstance(classifier_ids, str):
                classifier_ids = [classifier_ids]
            ensemble_weights = ensemble_cfg.get('weights', weights)

            if not classifier_ids:
                log(f"[classify_items] Ensemble '{ensemble_id}' has no classifiers; skipping")
                continue

            log(f"[classify_items] Running text ensemble '{ensemble_id}' with {len(classifier_ids)} classifier(s)")
            ensemble = EnsembleClassifier(classifier_ids, ensemble_weights, lazy_load, inactivity_timeout)

            chunk_errors = []
            device_logged = False

            for batch_start in range(0, total_text, mini_batch_size):
                # Check if job was cancelled before processing this batch
                if job_key in cancelled_jobs:
                    log(f"[classify_items] Text batch cancelled for job {job_key}; stopping")
                    cancelled_jobs.discard(job_key)
                    break
                
                batch_end = min(batch_start + mini_batch_size, total_text)
                batch_slice = text_results[batch_start:batch_end]
                batch_ids = [item_id for item_id, _, _, _ in batch_slice]
                batch_texts = [text for _, text, _, _ in batch_slice]
                batch_errors = [err for _, _, err, _ in batch_slice]

                t0 = time.time()
                batch_results, device_info = ensemble.classify_batch(batch_texts, modality='text')
                classify_ms = int((time.time() - t0) * 1000)

                if batch_start == 0 and device_info and not device_logged:
                    device_logged = True
                    device_msg = f"[Native Host V2] Processing {total_text} text section(s) with {len(classifier_ids)} model(s) | "
                    device_msg += f"{device_info['backend']}: {device_info['name']}"
                    chunk_errors.append({'type': 'info', 'message': device_msg, 'ensembleId': ensemble_id, 'modality': 'text'})

                chunk_results = []
                for idx, (item_id, result_data) in enumerate(zip(batch_ids, batch_results)):
                    fetch_err = batch_errors[idx]
                    if fetch_err:
                        result = {
                            'id': item_id,
                            'modality': 'text',
                            'label': 'uncertain',
                            'score': 0.5,
                            'display_label': 'AI generated',
                            'model': ','.join(classifier_ids),
                            'notes': fetch_err
                        }
                    else:
                        result = {
                            'id': item_id,
                            'modality': 'text',
                            'label': result_data['label'],
                            'score': result_data['score'],
                            'display_label': result_data.get('display_label', 'AI generated'),
                            'classifiers': result_data.get('classifiers', {}),
                            'model': ','.join(classifier_ids),
                            'durationMs': int(classify_ms / max(1, len(batch_slice)))
                        }
                        # Cache successful classification
                        if item_id:
                            cache_key = (item_id, ensemble_id)
                            classification_cache[cache_key] = {
                                'label': result_data['label'],
                                'score': result_data['score'],
                                'display_label': result_data.get('display_label', 'AI generated'),
                                'classifiers': result_data.get('classifiers', {}),
                                'model': ','.join(classifier_ids),
                                'timestamp': time.time()
                            }
                            _manage_cache_size()
                    chunk_results.append(result)

                if not emit_chunk(ensemble_id, 'text', chunk_results, chunk_errors):
                    results.extend(chunk_results)
                    errors.extend(chunk_errors)

    for it in other_items:
        results.append({
            'id': it.get('id'),
            'modality': it.get('modality'),
            'label': 'uncertain',
            'score': 0.5,
            'model': 'none',
            'durationMs': 0,
            'notes': 'unsupported_modality'
        })

    return {
        'version': 2,
        'type': 'classifyResult',
        'requestId': req_id,
        'timestamp': int(time.time() * 1000),
        'results': results,
        'errors': errors
    }


def list_models(request_id: Optional[str] = None):
    """List available classifiers from registry with fresh folder scan."""
    registry = get_registry()
    available = registry.list_available_details(force_rescan=True)
    
    # Also list .pt files in Immage Models for backward compatibility
    repo_root = Path(__file__).parent.parent
    imm_dir = repo_root / 'Immage Models'
    pt_files = []
    if imm_dir.exists():
        pt_files = [p.name for p in imm_dir.glob('*.pt') if p.is_file()]
    
    response = {
        'ok': True,
        'classifiers': available,
        'model_files': sorted(pt_files)
    }
    if request_id:
        response['requestId'] = request_id
    return response


# ----------------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------------

HOST_VERSION = "2.1.0"
OFFICIAL_CATALOG_URL = "https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json"
ALLOWED_GITHUB_PREFIXES = (
    "https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/",
    "https://media.githubusercontent.com/media/LuckyCowMoo/Fire-vs-Fire/",
    "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/download/",
)


def _is_allowed_official_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return any(url.startswith(prefix) for prefix in ALLOWED_GITHUB_PREFIXES)


def _resolve_safe_repo_path(relative_path: str) -> Optional[Path]:
    """Ensure target path stays strictly inside repo_root."""
    if not relative_path or ".." in relative_path:
        return None
    repo_root = Path(__file__).resolve().parent.parent
    target = (repo_root / relative_path).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        return None
    return target


def _load_official_catalog() -> Dict[str, Any]:
    """Load official first-party model catalog from GitHub with local fallback."""
    local_catalog_path = Path(__file__).resolve().parent / "official_models.json"
    catalog = None

    try:
        req = Request(
            OFFICIAL_CATALOG_URL,
            headers={"User-Agent": "FireVsFire-NativeHost/2.0", "Cache-Control": "no-cache"}
        )
        with urlopen(req, timeout=4) as resp:
            raw = resp.read().decode("utf-8")
            candidate = json.loads(raw)
            if isinstance(candidate, dict) and isinstance(candidate.get("models"), list):
                catalog = candidate
                log("[Catalog] Loaded latest official_models.json from GitHub")
    except Exception as exc:
        log(f"[Catalog] Remote GitHub fetch fell back to local catalog: {exc}")

    if local_catalog_path.exists():
        try:
            with open(local_catalog_path, "r", encoding="utf-8") as fh:
                local_data = json.load(fh)
                if isinstance(local_data, dict) and isinstance(local_data.get("models"), list):
                    if catalog is None:
                        catalog = local_data
                    else:
                        remote_ids = {m.get("id") for m in catalog.get("models", [])}
                        for lm in local_data.get("models", []):
                            if lm.get("id") not in remote_ids:
                                catalog["models"].append(lm)
        except Exception as exc:
            log(f"[Catalog] Failed to load local official_models.json: {exc}")

    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        catalog = {"catalog_version": 2, "repository": "LuckyCowMoo/Fire-vs-Fire", "models": []}

    installed_ids = []
    for model in catalog.get("models", []):
        files = model.get("files") or []
        if model.get("status") != "available" or not files:
            model["installed"] = False
            continue
        all_present = True
        for f_entry in files:
            rel_path = f_entry.get("relative_path", "")
            safe_path = _resolve_safe_repo_path(rel_path)
            if safe_path is None or not safe_path.exists() or safe_path.stat().st_size == 0:
                all_present = False
                break
        model["installed"] = all_present
        if all_present:
            installed_ids.append(model.get("id"))

    catalog["installed_models"] = installed_ids
    # Dynamically scan native/classifiers/ so ANY classifier or weights file
    # placed directly in the folder is instantly registered automatically.
    catalog = _sync_catalog_with_local_folder(catalog)
    return catalog


def _sync_catalog_with_local_folder(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically scans native/classifiers/ directly on disk.
    Any classifier Python module or model weights file found in the folder is
    automatically added to the catalog and marked installed, completely
    bypassing the need for manual JSON file edits or remote releases.
    """
    registry = get_registry()
    details = registry.list_available_details(force_rescan=True)
    models = catalog.setdefault("models", [])
    models_by_id = {m.get("id"): m for m in models if isinstance(m, dict) and m.get("id")}
    repo_root = Path(__file__).resolve().parent.parent

    for detail in details:
        cid = detail["id"]
        weights_ready = detail.get("weights_ready", False)

        if cid in models_by_id:
            m = models_by_id[cid]
            if weights_ready:
                m["installed"] = True
            if detail.get("download_size") and detail["download_size"] != "Local":
                m["download_size"] = detail["download_size"]
            if detail.get("size_mb") and detail["size_mb"] > 0:
                m["size_mb"] = detail["size_mb"]
        else:
            modality = (detail.get("modalities") or ["image"])[0]
            label = detail.get("label", "AI generated")
            is_ai = label.lower() not in ("spider", "object", "custom")
            name = detail.get("name") or cid
            is_recommended = ("04" in cid) or ("heimdall" in cid.lower() and "04" in cid)

            files_list = []
            if detail.get("py_file"):
                files_list.append({
                    "type": "classifier_py",
                    "relative_path": detail["py_file"],
                    "required": True
                })
            if detail.get("weight_path"):
                try:
                    rel_w = str(Path(detail["weight_path"]).resolve().relative_to(repo_root)).replace('\\', '/')
                except Exception:
                    rel_w = str(detail["weight_path"]).replace('\\', '/')
                files_list.append({
                    "type": "weights_pt" if rel_w.endswith(".pt") else "weights_onnx",
                    "relative_path": rel_w,
                    "required": True
                })

            new_model = {
                "id": cid,
                "name": name,
                "version": detail.get("version", "1.0.0-local"),
                "modality": modality,
                "status": "available",
                "recommended": is_recommended,
                "is_ai_detector": is_ai,
                "category_name": name,
                "label": label,
                "default_threshold": 0.75,
                "default_color": "#FF1744" if "04" in cid else ("#FF2E00" if "03" in cid else "#FF5722"),
                "size_mb": detail.get("size_mb", 0.0),
                "download_size": detail.get("download_size", "Local"),
                "ram_size": "1.5 GB VRAM / RAM",
                "architecture": name,
                "tagline": detail.get("doc", "").split("\n")[0] if detail.get("doc") else "Local folder auto-discovered classifier",
                "description": detail.get("doc") or f"Locally discovered classifier: {name}. Automatically loaded from native/classifiers/.",
                "files": files_list,
                "installed": weights_ready
            }

            if is_recommended:
                models.insert(0, new_model)
            else:
                models.append(new_model)
            models_by_id[cid] = new_model

    installed_ids = []
    for m in models:
        if m.get("installed") is True:
            installed_ids.append(m.get("id"))
            continue
        files = m.get("files") or []
        if m.get("status") == "available" and files:
            all_present = True
            for f_entry in files:
                rel_path = f_entry.get("relative_path", "")
                safe_path = _resolve_safe_repo_path(rel_path)
                if safe_path is None or not safe_path.exists() or safe_path.stat().st_size == 0:
                    all_present = False
                    break
            if all_present:
                m["installed"] = True
                installed_ids.append(m.get("id"))

    catalog["installed_models"] = sorted(list(set(installed_ids)))
    return catalog


def get_official_catalog_response(request_id: Optional[str] = None) -> Dict[str, Any]:
    catalog = _load_official_catalog()
    registry = get_registry()
    classifiers = registry.list_available_details(force_rescan=True)
    resp = {
        "ok": True,
        "type": "officialCatalog",
        "catalog": catalog,
        "installedModels": catalog.get("installed_models", []),
        "classifiers": classifiers,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def download_official_model(model_id: str, request_id: Optional[str] = None, force_download: bool = False) -> Dict[str, Any]:
    """
    Security-hardened downloader for first-party Fire vs Fire classifiers.
    Accepts ONLY a model_id (never arbitrary URLs/repositories from the browser),
    validates against the official LuckyCowMoo/Fire-vs-Fire catalog, and downloads
    both the .py classifier module(s) and .pt model weights.
    """
    if not model_id or not isinstance(model_id, str):
        resp = {"ok": False, "error": "Invalid modelId"}
        if request_id:
            resp["requestId"] = request_id
        return resp

    catalog = _load_official_catalog()
    target_model = next((m for m in catalog.get("models", []) if m.get("id") == model_id), None)
    if target_model is None:
        resp = {"ok": False, "error": f"Model '{model_id}' is not in the official Fire vs Fire catalog."}
        if request_id:
            resp["requestId"] = request_id
        return resp

    if target_model.get("status") != "available":
        resp = {"ok": False, "error": f"Model '{model_id}' is not yet released ({target_model.get('status')})."}
        if request_id:
            resp["requestId"] = request_id
        return resp

    files = target_model.get("files") or []
    if not files:
        resp = {"ok": False, "error": f"Model '{model_id}' has no downloadable files configured."}
        if request_id:
            resp["requestId"] = request_id
        return resp

    total_files = len(files)
    downloaded_files = []

    for file_idx, f_entry in enumerate(files):
        rel_path = f_entry.get("relative_path", "")
        primary_url = f_entry.get("url", "")
        fallback_url = f_entry.get("fallback_url", "")

        dest_path = _resolve_safe_repo_path(rel_path)
        if dest_path is None:
            resp = {"ok": False, "error": f"Unsafe target path rejected: {rel_path}"}
            if request_id:
                resp["requestId"] = request_id
            return resp

        if not _is_allowed_official_url(primary_url):
            resp = {"ok": False, "error": f"Untrusted download URL rejected: {primary_url}"}
            if request_id:
                resp["requestId"] = request_id
            return resp

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not force_download and dest_path.exists() and dest_path.stat().st_size > 0:
            file_size = dest_path.stat().st_size
            overall_pct = int(((file_idx + 1) / total_files) * 100)
            send_message({
                "version": 2,
                "type": "downloadProgress",
                "requestId": request_id,
                "modelId": model_id,
                "file": dest_path.name,
                "fileIndex": file_idx + 1,
                "totalFiles": total_files,
                "bytesDownloaded": file_size,
                "totalBytes": file_size,
                "percent": overall_pct,
                "status": f"Verified {dest_path.name}"
            })
            downloaded_files.append(rel_path)
            continue

        urls_to_try = [primary_url]
        if fallback_url and _is_allowed_official_url(fallback_url) and fallback_url != primary_url:
            urls_to_try.append(fallback_url)

        last_err = None
        success = False
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        for candidate_url in urls_to_try:
            try:
                log(f"[Downloader] Fetching {model_id} file {rel_path} from {candidate_url}")
                req = Request(candidate_url, headers={"User-Agent": "FireVsFire-NativeHost/2.0"})
                with urlopen(req, timeout=30) as response:
                    total_bytes = int(response.headers.get("Content-Length") or 0)
                    bytes_so_far = 0
                    chunk_size = 256 * 1024
                    last_emit = 0.0

                    with open(tmp_path, "wb") as out_fh:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_fh.write(chunk)
                            bytes_so_far += len(chunk)
                            now = time.time()
                            if now - last_emit >= 0.12:
                                last_emit = now
                                file_frac = (bytes_so_far / total_bytes) if total_bytes > 0 else 0.5
                                overall_pct = int(((file_idx + file_frac) / total_files) * 100)
                                send_message({
                                    "version": 2,
                                    "type": "downloadProgress",
                                    "requestId": request_id,
                                    "modelId": model_id,
                                    "file": dest_path.name,
                                    "fileIndex": file_idx + 1,
                                    "totalFiles": total_files,
                                    "bytesDownloaded": bytes_so_far,
                                    "totalBytes": total_bytes,
                                    "percent": min(99, max(1, overall_pct)),
                                    "status": f"Downloading {dest_path.name}..."
                                })

                if tmp_path.exists() and tmp_path.stat().st_size > 0:
                    os.replace(tmp_path, dest_path)
                    success = True
                    downloaded_files.append(rel_path)
                    send_message({
                        "version": 2,
                        "type": "downloadProgress",
                        "requestId": request_id,
                        "modelId": model_id,
                        "file": dest_path.name,
                        "fileIndex": file_idx + 1,
                        "totalFiles": total_files,
                        "bytesDownloaded": dest_path.stat().st_size,
                        "totalBytes": dest_path.stat().st_size,
                        "percent": int(((file_idx + 1) / total_files) * 100),
                        "status": f"Installed {dest_path.name}"
                    })
                    break
            except Exception as exc:
                last_err = str(exc)
                log(f"[Downloader] Error downloading {candidate_url}: {exc}")
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        if not success:
            if dest_path.exists() and dest_path.stat().st_size > 0:
                downloaded_files.append(rel_path)
                continue
            resp = {
                "ok": False,
                "modelId": model_id,
                "error": f"Failed to download {dest_path.name} from official GitHub: {last_err}"
            }
            if request_id:
                resp["requestId"] = request_id
            return resp

    registry = get_registry()
    registry.discover_classifiers(force=True)
    available = registry.list_available_details()

    resp = {
        "ok": True,
        "type": "downloadComplete",
        "modelId": model_id,
        "files": downloaded_files,
        "classifiers": available,
        "timestamp": int(time.time() * 1000)
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def check_for_host_updates(request_id: Optional[str] = None) -> Dict[str, Any]:
    catalog = _load_official_catalog()
    remote_version = catalog.get("host_version", HOST_VERSION)
    update_available = (remote_version != HOST_VERSION)
    resp = {
        "ok": True,
        "type": "checkUpdateResponse",
        "currentVersion": HOST_VERSION,
        "latestVersion": remote_version,
        "updateAvailable": update_available,
        "releaseNotes": catalog.get("release_notes", "Performance improvements and new model support."),
        "downloadUrl": "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/download/v2.0/FireVsFire-NativeHost.zip",
        "timestamp": int(time.time() * 1000)
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def apply_host_updates(request_id: Optional[str] = None) -> Dict[str, Any]:
    files_to_update = [
        "native/echo_host_V2.py",
        "native/model_registry.py",
        "native/official_models.json",
        "native/classifiers/base_classifier.py",
        "native/classifiers/fvf_heimdall_02_classifier.py",
        "native/classifiers/fvf_heimdall_03_classifier.py",
        "native/classifiers/Resnet50_spider.py"
    ]
    base_raw_url = "https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/"
    updated_files = []
    errors = []

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    for rel_path in files_to_update:
        target_path = repo_root / rel_path
        remote_url = base_raw_url + rel_path
        try:
            req = Request(remote_url, headers={"User-Agent": f"FireVsFire-Updater/{HOST_VERSION}", "Cache-Control": "no-cache"})
            with urlopen(req, timeout=12) as resp_stream:
                content = resp_stream.read()
                if content and len(content) > 50:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(content)
                    updated_files.append(rel_path)
                    log(f"[Updater] Successfully updated {rel_path}")
        except Exception as exc:
            log(f"[Updater] Failed updating {rel_path}: {exc}")
            errors.append(f"{rel_path}: {exc}")

    catalog = _load_official_catalog()
    new_version = catalog.get("host_version", HOST_VERSION)

    resp = {
        "ok": len(updated_files) > 0,
        "type": "applyUpdateResponse",
        "updatedFiles": updated_files,
        "version": new_version,
        "message": f"Successfully updated {len(updated_files)} native host component(s) to v{new_version}. Please reload the extension or restart your browser.",
        "timestamp": int(time.time() * 1000)
    }
    if errors:
        resp["errors"] = errors
    if request_id:
        resp["requestId"] = request_id
    return resp


job_assembler = JobAssembler()

def main():
    """Main message loop for native messaging protocol."""
    log("[Main] Entering main loop (V2)")
    
    while True:
        try:
            msg = read_message()
            if msg is None:
                log("[Main] read_message returned None, exiting")
                break
            
            mtype = msg.get('type')
            log(f"[Main] Received message type: {mtype}")
            
            if mtype == 'ping' or mtype == 'PING':
                req_id = msg.get('requestId')
                resp = {
                    'ok': True,
                    'type': 'pong',
                    'version': HOST_VERSION,
                    'time': int(time.time() * 1000)
                }
                if req_id:
                    resp['requestId'] = req_id
                send_message(resp)
                continue

            elif mtype == 'checkUpdate' or mtype == 'checkNativeHostUpdate':
                req_id = msg.get('requestId')
                resp = check_for_host_updates(req_id)
                send_message(resp)
                continue

            elif mtype == 'applyUpdate' or mtype == 'applyNativeHostUpdate':
                req_id = msg.get('requestId')
                resp = apply_host_updates(req_id)
                send_message(resp)
                continue

            # Ping handled above
            
            elif mtype == 'classifyJobChunk':
                job_id = msg.get('jobId') or msg.get('requestId') or f"job-{int(time.time()*1000)}"
                chunk_index = int(msg.get('chunkIndex', 0))
                total_chunks = int(msg.get('totalChunks', 1))
                payload = msg.get('payload') or {}

                log(f"[Main] Received job chunk {chunk_index+1}/{total_chunks} for job {job_id}")
                envelope = job_assembler.add_chunk(job_id, chunk_index, total_chunks, payload)

                if envelope:
                    log(f"[Main] Job {job_id} reconstructed; starting classification")
                    classify_items(envelope, job_id=job_id, send_chunk=send_message)
                    send_message({
                        'version': 2,
                        'type': 'classifyJobComplete',
                        'jobId': job_id,
                        'timestamp': int(time.time() * 1000)
                    })

            elif mtype == 'classify':
                log(f"[Main] Starting classify with {len(msg.get('payload', {}).get('items', []))} items")
                final_resp = classify_items(msg)
                log(f"[Main] Classify complete, sending final response")
                send_message(final_resp)
            
            elif mtype == 'listModels':
                log("[Main] Listing available models")
                send_message(list_models(msg.get('requestId')))

            elif mtype == 'getOfficialCatalog':
                log("[Main] Getting official first-party catalog")
                send_message(get_official_catalog_response(msg.get('requestId')))

            elif mtype == 'downloadOfficialModel':
                model_id = msg.get('modelId')
                force_dl = bool(msg.get('forceDownload', False))
                log(f"[Main] Downloading official model: {model_id}")
                send_message(download_official_model(model_id, msg.get('requestId'), force_dl))
            
            elif mtype == 'cancelJob':
                job_key = msg.get('jobId') or msg.get('requestId')
                log(f"[Main] Received cancel request for job {job_key}")
                if job_key:
                    cancelled_jobs.add(job_key)
                    send_message({
                        'ok': True,
                        'type': 'jobCancelled',
                        'jobId': job_key,
                        'timestamp': int(time.time() * 1000)
                    })
                else:
                    send_message({'ok': False, 'error': 'No jobId provided'})
            
            elif mtype == 'clearCache':
                log(f"[Main] Clearing classification cache ({len(classification_cache)} items)")
                classification_cache.clear()
                send_message({
                    'ok': True,
                    'type': 'cacheCleared',
                    'timestamp': int(time.time() * 1000)
                })
            
            elif mtype == 'getCacheStats':
                send_message({
                    'ok': True,
                    'type': 'cacheStats',
                    'size': len(classification_cache),
                    'maxSize': MAX_CACHE_SIZE,
                    'timestamp': int(time.time() * 1000)
                })
            
            else:
                log(f"[Main] Unknown message type: {mtype}")
                send_message({'ok': False, 'error': f'Unknown type: {mtype}'})
        
        except Exception as e:
            log(f"[Main] Exception in main loop: {e}")
            import traceback
            log(traceback.format_exc())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"[Error] Uncaught exception: {e}")
        import traceback
        log(traceback.format_exc())
