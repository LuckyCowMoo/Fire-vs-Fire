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
# Maps item_id -> {score, label, display_label, classifiers, timestamp}
classification_cache = {}
MAX_CACHE_SIZE = 5000

# Import classifier registry
from model_registry import get_registry

# ============================================================================
# Native Messaging Protocol
# ============================================================================

def read_message():
    # Read one message from browser via stdin
    raw_length = sys.stdin.buffer.read(4)
    if len(raw_length) == 0:
        return None
    message_length = struct.unpack('<I', raw_length)[0]
    message = sys.stdin.buffer.read(message_length).decode('utf-8')
    return json.loads(message)

def send_message(message):
    # Send one message to browser via stdout
    encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('<I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


# ============================================================================
# Job Reconstruction (Chunked Requests)
# ============================================================================

class JobAssembler:
    def __init__(self, ttl_seconds: int = 120, max_bytes: int = 30 * 1024 * 1024):
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


def fetch_images_from_urls(image_items: List[Dict[str, Any]]) -> List[tuple]:
    
    # Fetch images from URLs or decode base64 data URLs
    
    # Returns 4-tuples per image: (item_id, PIL.Image or None, error_reason or None, fetch_duration_ms)
        
    results = []
    base64_count = 0
    http_count = 0
    
    for item in image_items:
        item_id = item.get('id')
        url = item.get('url')
        t0 = time.time()
        
        try:
            # Handle data URLs (base64-encoded images from browser)
            if url and url.startswith('data:'):
                base64_count += 1
                header, data_part = url.split(',', 1)
                if ';base64' in header:
                    raw = base64.b64decode(data_part)
                else:
                    raw = urllib.parse.unquote_to_bytes(data_part)
                # Optionally dump incoming raw bytes for debugging transport/preproc
                try:
                    if os.environ.get('FIRE_SAVE_INCOMING_IMAGES') == '1':
                        dbg_dir = Path(__file__).resolve().parent / 'incoming_debug'
                        dbg_dir.mkdir(exist_ok=True)
                        # Guess extension from header
                        ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
                        out_path = dbg_dir / f"{item_id}_{int(time.time()*1000)}.{ext}"
                        with open(out_path, 'wb') as fh:
                            fh.write(raw)
                        log(f"[fetch_images] Saved incoming image for {item_id} -> {out_path} ({len(raw)} bytes)")
                except Exception:
                    pass

                img = Image.open(BytesIO(raw)).convert('RGB')
                results.append((item_id, img, None, int((time.time() - t0) * 1000)))
            
            # Handle HTTP/HTTPS URLs (fetch from network)
            elif url and url.startswith(('http://', 'https://')):
                http_count += 1
                req = Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (native-host)',
                    'Accept': 'image/*,*/*;q=0.8'
                })
                with urlopen(req, timeout=15) as resp:
                    data = resp.read()
                try:
                    if os.environ.get('FIRE_SAVE_INCOMING_IMAGES') == '1':
                        dbg_dir = Path(__file__).resolve().parent / 'incoming_debug'
                        dbg_dir.mkdir(exist_ok=True)
                        out_path = dbg_dir / f"{item_id}_{int(time.time()*1000)}.jpg"
                        with open(out_path, 'wb') as fh:
                            fh.write(data)
                        log(f"[fetch_images] Saved fetched image for {item_id} -> {out_path} ({len(data)} bytes)")
                except Exception:
                    pass
                img = Image.open(BytesIO(data)).convert('RGB')
                results.append((item_id, img, None, int((time.time() - t0) * 1000)))
            
            else:
                results.append((item_id, None, 'errorus/no_url', int((time.time() - t0) * 1000)))
        
        except (HTTPError, URLError) as e:
            results.append((item_id, None, 'image_fetch_failed', int((time.time() - t0) * 1000)))
        except Exception as e:
            results.append((item_id, None, 'image_load_error', int((time.time() - t0) * 1000)))
    
    if base64_count > 0 or http_count > 0:
        log(f"[fetch_images] Source breakdown: {base64_count} base64 (cached), {http_count} HTTP (network)")
    
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
    if len(classification_cache) > MAX_CACHE_SIZE:
        # Sort by timestamp and remove oldest entries
        sorted_items = sorted(classification_cache.items(), key=lambda x: x[1].get('timestamp', 0))
        num_to_remove = len(classification_cache) - MAX_CACHE_SIZE
        for i in range(num_to_remove):
            item_id = sorted_items[i][0]
            classification_cache.pop(item_id, None)
        log(f"[Cache] Evicted {num_to_remove} oldest entries, cache size now {len(classification_cache)}")


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
    mini_batch_size = model_config.get('miniBatchSize', 1000)
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
    
    # Filter out cached items from image_items
    uncached_image_items = []
    for item in image_items:
        item_id = item.get('id')
        if item_id and item_id in classification_cache:
            cached_data = classification_cache[item_id]
            cached_results.append({
                'id': item_id,
                'modality': 'image',
                'ensembleId': cached_data.get('ensembleId', 'cached'),
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
            uncached_image_items.append(item)
    
    # Filter out cached items from text_items
    uncached_text_items = []
    for item in text_items:
        item_id = item.get('id')
        if item_id and item_id in classification_cache:
            cached_data = classification_cache[item_id]
            cached_results.append({
                'id': item_id,
                'modality': 'text',
                'ensembleId': cached_data.get('ensembleId', 'cached'),
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
            uncached_text_items.append(item)
    
    # Replace with uncached items only
    image_items = uncached_image_items
    text_items = uncached_text_items
    
    if cache_hits > 0:
        log(f"[Cache] {cache_hits} items found in cache, {len(image_items)} images and {len(text_items)} text items need classification")
        # Send cached results immediately if streaming
        if send_chunk and stream_results and cached_results:
            send_chunk({
                'version': 2,
                'type': 'classifyResultChunk',
                'jobId': job_id or req_id,
                'ensembleId': 'cached',
                'modality': 'mixed',
                'timestamp': int(time.time() * 1000),
                'results': cached_results,
                'errors': [{'type': 'info', 'message': f'Returned {cache_hits} cached results'}]
            })
        else:
            results.extend(cached_results)

    if not image_items and not text_items:
        return {
            'version': 2,
            'type': 'classifyResult',
            'requestId': req_id,
            'timestamp': int(time.time() * 1000),
            'results': [],
            'errors': [{'type': 'info', 'message': 'No images or text to classify'}]
        }

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
                            'ensembleId': ensemble_id,
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
                            'ensembleId': ensemble_id,
                            'label': result_data['label'],
                            'score': result_data['score'],
                            'display_label': result_data.get('display_label', 'AI generated'),
                            'classifiers': result_data.get('classifiers', {}),
                            'model': ','.join(classifier_ids),
                            'durationMs': int(classify_ms / max(1, len(batch_slice)))
                        }
                        # Cache successful classification
                        if item_id:
                            classification_cache[item_id] = {
                                'ensembleId': ensemble_id,
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
                            'ensembleId': ensemble_id,
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
                            'ensembleId': ensemble_id,
                            'label': result_data['label'],
                            'score': result_data['score'],
                            'display_label': result_data.get('display_label', 'AI generated'),
                            'classifiers': result_data.get('classifiers', {}),
                            'model': ','.join(classifier_ids),
                            'durationMs': int(classify_ms / max(1, len(batch_slice)))
                        }
                        # Cache successful classification
                        if item_id:
                            classification_cache[item_id] = {
                                'ensembleId': ensemble_id,
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
    """List available classifiers from registry."""
    registry = get_registry()
    available = registry.list_available_details()
    
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
                send_message({'ok': True, 'time': int(time.time() * 1000)})
            
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
