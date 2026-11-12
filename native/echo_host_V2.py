#!.venv311/Scripts/python.exe
"""
PYTHON ENVIRONMENT: .venv311 (Python 3.11 with GPU support via torch-directml / CUDA / MPS)
LOG FILE: ../native_host_v2.log

Dynamic Prioritisation & Single-Item Streaming Architecture:
- Dedicated I/O Thread + Priority Inference Worker Thread ("Keep the Engine Running").
- Content-Addressed RAM Cache (SHA-256) checked on the I/O thread BEFORE queueing (<1ms cache hits)
  plus in-flight deduplication across tabs, windows, and LAN sources.
- Zero-Internet Native Host: Only decodes base64/data payloads; never fetches external URLs.
- PacketAssembler: Reassembles chunked single items (>512 KB) under the 1 MB Native Messaging limit.
- Dynamic Context & Priority Scheduling:
  * P0: Active tab & in viewport (or boosted by scroll)
  * P1: Active tab & offscreen
  * P2: Background tab / inactive window
  * Instant queue purge on `dropContext` (tab closed or navigated away).
- Multi-Model Hardware Memory Manager: Keeps multiple simultaneous classifiers warm in RAM/VRAM
  when hardware permits, with automatic LRU eviction on low-memory systems and idle timeout unloading.
"""

import sys
import json
import struct
import time
import hashlib
import threading
import gc
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from collections import OrderedDict
import base64
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Set
from PIL import Image
import os
import ssl

# Setup logging to file
LOG_FILE_LOCAL = Path(__file__).resolve().parent / "native_host.log"
LOG_FILE_ROOT = Path(__file__).resolve().parent.parent / "native_host_v2.log"
LOG_FILE_TEMP = Path(os.environ.get("TEMP", os.environ.get("TMP", "."))) / "native_host_v2.log"

_log_lock = threading.Lock()


def log(msg: str) -> None:
    formatted = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with _log_lock:
        for p in (LOG_FILE_LOCAL, LOG_FILE_ROOT, LOG_FILE_TEMP):
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(formatted)
                    f.flush()
            except Exception:
                pass


log("[START] Echo Host V2 (Dynamic Prioritisation Engine) starting")

# Global set to track cancelled jobs (legacy support)
cancelled_jobs: Set[str] = set()

# Legacy classification cache alias (kept for backward compatibility)
classification_cache: Dict[Any, Any] = {}
MAX_CACHE_SIZE = 10000

# Import classifier registry and BaseClassifier
from model_registry import get_registry
from classifiers.base_classifier import BaseClassifier

# ============================================================================
# Native Messaging Protocol (Thread-Safe)
# ============================================================================

# Preserve raw stdout binary buffer for Native Messaging protocol
_raw_stdout = sys.stdout.buffer
# Redirect text stdout to stderr so ANY accidental print() never corrupts native messaging!
sys.stdout = sys.stderr

_stdout_lock = threading.Lock()


def read_message() -> Optional[Dict[str, Any]]:
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length or len(raw_length) < 4:
        return None
    message_length = struct.unpack('<I', raw_length)[0]

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


def send_message(message: Dict[str, Any]) -> None:
    encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    max_safe_bytes = 750 * 1024
    if (
        len(encoded) >= max_safe_bytes
        and isinstance(message, dict)
        and 'results' in message
        and isinstance(message['results'], list)
    ):
        log(f"[Native] Warning: message size ({len(encoded)} bytes) exceeds safe limit; truncating results")
        while len(encoded) >= max_safe_bytes and len(message['results']) > 1:
            message['results'] = message['results'][: len(message['results']) // 2]
            encoded = json.dumps(message, separators=(',', ':')).encode('utf-8')
    with _stdout_lock:
        _raw_stdout.write(struct.pack('<I', len(encoded)))
        _raw_stdout.write(encoded)
        _raw_stdout.flush()


# ============================================================================
# Content-Addressed RAM Cache & In-Flight Deduplicator
# ============================================================================

class ContentCache:
    """
    Thread-safe in-memory LRU cache keyed by (content_hash, classifier_id)
    and fallback (item_id, classifier_id).
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        # (content_hash, classifier_id) -> {"score": float, "label": str, "model_name": str, "timestamp": float}
        self._by_hash: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        # item_id -> content_hash (allows instant lookup by itemId before/after hashing)
        self._item_to_hash: "OrderedDict[str, str]" = OrderedDict()

    @staticmethod
    def compute_content_hash(modality: str, raw_bytes_or_text: Any, fallback_id: str = "") -> str:
        h = hashlib.sha256()
        h.update(modality.encode("utf-8", errors="ignore"))
        h.update(b":")
        if isinstance(raw_bytes_or_text, bytes):
            h.update(raw_bytes_or_text)
        elif isinstance(raw_bytes_or_text, str) and raw_bytes_or_text:
            h.update(raw_bytes_or_text.encode("utf-8", errors="ignore"))
        else:
            h.update(fallback_id.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def register_item_hash(self, item_id: str, content_hash: str) -> None:
        if not item_id or not content_hash:
            return
        with self._lock:
            self._item_to_hash[item_id] = content_hash
            self._item_to_hash.move_to_end(item_id)
            while len(self._item_to_hash) > self.max_entries * 2:
                self._item_to_hash.popitem(last=False)

    def get_classifier_score(self, content_hash: str, classifier_id: str, item_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            key = (content_hash, classifier_id)
            if key in self._by_hash:
                self._by_hash.move_to_end(key)
                return self._by_hash[key]
            if item_id and item_id in self._item_to_hash:
                mapped_hash = self._item_to_hash[item_id]
                alt_key = (mapped_hash, classifier_id)
                if alt_key in self._by_hash:
                    self._by_hash.move_to_end(alt_key)
                    return self._by_hash[alt_key]
        return None

    def set_classifier_score(
        self,
        content_hash: str,
        classifier_id: str,
        score: float,
        label: str,
        model_name: str = "",
        item_id: Optional[str] = None,
    ) -> None:
        if score < 0:
            return
        with self._lock:
            entry = {
                "score": round(float(score), 4),
                "label": label or "AI generated",
                "model_name": model_name or classifier_id,
                "timestamp": time.time(),
            }
            key = (content_hash, classifier_id)
            self._by_hash[key] = entry
            self._by_hash.move_to_end(key)
            if item_id:
                self._item_to_hash[item_id] = content_hash
                self._item_to_hash.move_to_end(item_id)
            while len(self._by_hash) > self.max_entries:
                self._by_hash.popitem(last=False)

    def evaluate_ensembles_from_cache(
        self,
        content_hash: str,
        ensembles: List[Dict[str, Any]],
        item_id: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]], Set[str]]:
        """
        Check if all classifiers required across `ensembles` are already in the cache.
        Returns:
            (all_cached: bool, ensemble_results: List[Dict], missing_classifiers: Set[str])
        """
        missing: Set[str] = set()
        cached_per_clf: Dict[str, Dict[str, Any]] = {}

        for ens in ensembles:
            clf_ids = ens.get("classifiers") or []
            if isinstance(clf_ids, str):
                clf_ids = [clf_ids]
            for cid in clf_ids:
                if cid in cached_per_clf:
                    continue
                hit = self.get_classifier_score(content_hash, cid, item_id=item_id)
                if hit is not None:
                    cached_per_clf[cid] = hit
                else:
                    missing.add(cid)

        if missing:
            return False, [], missing

        ensemble_results = build_ensemble_results(ensembles, cached_per_clf, duration_ms=0, cached=True)
        return True, ensemble_results, set()

    def clear(self) -> None:
        with self._lock:
            self._by_hash.clear()
            self._item_to_hash.clear()
            classification_cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._by_hash)


def build_ensemble_results(
    ensembles: List[Dict[str, Any]],
    clf_scores: Dict[str, Dict[str, Any]],
    duration_ms: int = 0,
    cached: bool = False,
) -> List[Dict[str, Any]]:
    """
    Given per-classifier scores `{clf_id: {'score': float, 'label': str}}`,
    compute the weighted ensemble results for all requested ensembles.
    """
    out: List[Dict[str, Any]] = []
    for ens in ensembles:
        ens_id = ens.get("id") or "default_ensemble"
        clf_ids = ens.get("classifiers") or []
        if isinstance(clf_ids, str):
            clf_ids = [clf_ids]
        raw_weights = ens.get("weights")
        if not raw_weights or not isinstance(raw_weights, list) or len(raw_weights) != len(clf_ids):
            weights = [1.0 / max(1, len(clf_ids))] * len(clf_ids)
        else:
            weights = [float(w) if isinstance(w, (int, float)) and w >= 0 else 1.0 for w in raw_weights]

        valid_scores: List[float] = []
        valid_weights: List[float] = []
        classifier_details: Dict[str, Any] = {}
        classifier_labels: List[str] = []

        for cid, w in zip(clf_ids, weights):
            info = clf_scores.get(cid)
            if not info:
                continue
            sc = float(info.get("score", -1.0))
            lbl = info.get("label") or "AI generated"
            if sc >= 0.0:
                valid_scores.append(sc)
                valid_weights.append(w)
                classifier_details[cid] = {
                    "score": round(sc, 4),
                    "label": lbl,
                }
                if lbl:
                    classifier_labels.append(lbl)

        if valid_scores:
            w_sum = sum(valid_weights)
            if w_sum > 0:
                avg_score = sum(s * (w / w_sum) for s, w in zip(valid_scores, valid_weights))
            else:
                avg_score = sum(valid_scores) / len(valid_scores)
            label = "ai" if avg_score >= 0.5 else "real"
            display_label = next(
                (name for name in classifier_labels if name and name != "AI generated"),
                (classifier_labels[0] if classifier_labels else "AI generated"),
            )
            out.append({
                "ensembleId": ens_id,
                "score": round(avg_score, 4),
                "label": label,
                "display_label": display_label,
                "classifiers": classifier_details,
                "model": ",".join(clf_ids),
                "durationMs": duration_ms,
                "cached": cached,
            })
        else:
            out.append({
                "ensembleId": ens_id,
                "score": 0.5,
                "label": "uncertain",
                "display_label": "AI generated",
                "classifiers": {},
                "model": ",".join(clf_ids),
                "durationMs": duration_ms,
                "cached": cached,
            })
    return out


# ============================================================================
# PacketAssembler (Single-Item Chunk Reassembly Under 1 MB Limit)
# ============================================================================

class PacketAssembler:
    """
    Reassembles single items split across multiple packets (`submitItemPacket`)
    so large base64 images/audio/video/text never exceed browser 1 MB limits.
    """

    def __init__(self, ttl_seconds: int = 120, max_item_bytes: int = 50 * 1024 * 1024):
        self.ttl_seconds = ttl_seconds
        self.max_item_bytes = max_item_bytes
        self._lock = threading.Lock()
        self._assemblies: Dict[str, Dict[str, Any]] = {}

    def _cleanup_unlocked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._assemblies.items() if now - v["created_at"] > self.ttl_seconds]
        for k in expired:
            self._assemblies.pop(k, None)

    def add_packet(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a packet from a `submitItemPacket` message.
        Returns the complete item request dict when all packets have arrived, else None.
        """
        packet_id = msg.get("packetGroupId") or f"{msg.get('sourceId', 'local')}:{msg.get('pageId', '')}:{msg.get('itemId', '')}"
        packet_index = int(msg.get("packetIndex", 0))
        total_packets = int(msg.get("totalPackets", 1))
        data_chunk = msg.get("dataChunk") or ""

        if total_packets <= 1:
            assembled = dict(msg)
            assembled["data"] = data_chunk or msg.get("data") or msg.get("url") or msg.get("text") or ""
            return assembled

        with self._lock:
            self._cleanup_unlocked()
            entry = self._assemblies.get(packet_id)
            if entry is None:
                entry = {
                    "created_at": time.time(),
                    "total_packets": total_packets,
                    "chunks": {},
                    "bytes": 0,
                    "meta": None,
                }
                self._assemblies[packet_id] = entry

            if total_packets > 0:
                entry["total_packets"] = total_packets

            if packet_index not in entry["chunks"]:
                chunk_len = len(data_chunk)
                if entry["bytes"] + chunk_len > self.max_item_bytes:
                    log(f"[PacketAssembler] Item {packet_id} exceeded max size ({self.max_item_bytes} bytes); dropping")
                    self._assemblies.pop(packet_id, None)
                    return None
                entry["chunks"][packet_index] = data_chunk
                entry["bytes"] += chunk_len

            if entry["meta"] is None and (msg.get("itemId") or msg.get("modality") or msg.get("model")):
                meta = dict(msg)
                meta.pop("dataChunk", None)
                entry["meta"] = meta
            elif entry["meta"] is not None and msg.get("model") and not entry["meta"].get("model"):
                entry["meta"]["model"] = msg.get("model")

            if len(entry["chunks"]) >= entry["total_packets"]:
                full_data = "".join(entry["chunks"].get(i, "") for i in range(entry["total_packets"]))
                result = dict(entry["meta"] or msg)
                result["data"] = full_data
                self._assemblies.pop(packet_id, None)
                return result

        return None

    def drop_context(self, source_id: Optional[str] = None, tab_id: Optional[Any] = None, page_id: Optional[str] = None) -> int:
        dropped = 0
        with self._lock:
            to_remove = []
            for k, v in self._assemblies.items():
                meta = v.get("meta") or {}
                if page_id and meta.get("pageId") == page_id:
                    to_remove.append(k)
                elif tab_id is not None and meta.get("tabId") == tab_id and (not source_id or meta.get("sourceId") == source_id):
                    to_remove.append(k)
            for k in to_remove:
                self._assemblies.pop(k, None)
                dropped += 1
        return dropped


# ============================================================================
# Legacy JobAssembler (Kept for Backward Compatibility with Old Clients)
# ============================================================================

class JobAssembler:
    def __init__(self, ttl_seconds: int = 300, max_bytes: int = 250 * 1024 * 1024):
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def _cleanup(self) -> None:
        now = time.time()
        to_delete = [jid for jid, job in self.jobs.items() if now - job['created_at'] > self.ttl_seconds]
        for job_id in to_delete:
            self.jobs.pop(job_id, None)

    def add_chunk(self, job_id: str, chunk_index: int, total_chunks: int, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._cleanup()
        if job_id not in self.jobs:
            self.jobs[job_id] = {
                'created_at': time.time(),
                'total_chunks': total_chunks,
                'received': {},
                'bytes': 0,
                'model': None,
            }
        job = self.jobs[job_id]
        if total_chunks and total_chunks > 0:
            job['total_chunks'] = total_chunks
        if chunk_index in job['received']:
            return None

        items = payload.get('items') or []
        model = payload.get('model') or None
        try:
            chunk_bytes = len(json.dumps(items))
        except Exception:
            chunk_bytes = 0

        if job['bytes'] + chunk_bytes > self.max_bytes:
            self.jobs.pop(job_id, None)
            return None

        job['received'][chunk_index] = items
        job['bytes'] += chunk_bytes
        if model and job['model'] is None:
            job['model'] = model

        if len(job['received']) >= job['total_chunks']:
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
                    'model': job['model'] or {},
                },
            }
            self.jobs.pop(job_id, None)
            return envelope
        return None


# ============================================================================
# Zero-Internet Payload Decoding (Base64 / Text Only)
# ============================================================================

def decode_image_payload(data_or_url: str) -> Tuple[Optional[Image.Image], Optional[bytes], Optional[str]]:
    """
    Decode an image strictly from a `data:` URI or raw base64 string.
    External HTTP/HTTPS URLs are rejected for security (zero Native Host internet fetching).
    Returns:
        (pil_image, raw_bytes, error_reason)
    """
    if not data_or_url or not isinstance(data_or_url, str):
        return None, None, "empty_image_payload"
    try:
        if data_or_url.startswith(("http://", "https://")):
            return None, None, "remote_url_fetching_disabled_in_native_host"
        if data_or_url.startswith("data:"):
            header, data_part = data_or_url.split(",", 1)
            if ";base64" in header:
                raw = base64.b64decode(data_part)
            else:
                raw = urllib.parse.unquote_to_bytes(data_part)
        else:
            raw = base64.b64decode(data_or_url)
        img = Image.open(BytesIO(raw)).convert("RGB")
        return img, raw, None
    except Exception as exc:
        return None, None, f"decode_failed: {str(exc)[:60]}"


def _fetch_single_image(item: Dict[str, Any]) -> tuple:
    item_id = item.get('id')
    url = item.get('data') or item.get('url') or ''
    t0 = time.time()
    img, _raw, err = decode_image_payload(url)
    return (item_id, img, err, int((time.time() - t0) * 1000))


def fetch_images_from_urls(image_items: List[Dict[str, Any]]) -> List[tuple]:
    if not image_items:
        return []
    return [_fetch_single_image(it) for it in image_items]


def fetch_text_from_items(text_items: List[Dict[str, Any]]) -> List[tuple]:
    results = []
    for item in text_items:
        item_id = item.get('id')
        text = item.get('text') or item.get('data') or ''
        results.append((item_id, text, None, 0))
    return results


# ============================================================================
# Multi-Model Hardware Memory Manager ("Keep the Engine Running")
# ============================================================================

class ModelMemoryManager:
    """
    Manages loaded BaseClassifier instances across modalities and categories
    (e.g., AI image + Spider image + AI text running simultaneously).
    - Keeps active models warm in RAM/VRAM so single-item inference has 0ms spool-up.
    - Probes hardware RAM/VRAM via `BaseClassifier.get_system_memory_info()` before loading
      a new model; on memory-constrained systems, evicts the least-recently-used (LRU) model.
    - Periodically unloads all models after `inactivity_timeout_minutes` of total queue silence.
    """

    # Reserve at least 1100 MB of system RAM (and 450 MB VRAM when on CUDA) as safety headroom
    MIN_FREE_RAM_MB = 1100.0
    MIN_FREE_VRAM_MB = 450.0

    def __init__(self, inactivity_timeout_minutes: float = 15.0):
        self.registry = get_registry()
        self.inactivity_timeout_minutes = inactivity_timeout_minutes
        self.last_activity_time = time.time()
        self._lock = threading.Lock()
        # clf_id -> last_used_timestamp (in LRU order)
        self._active_lru: "OrderedDict[str, float]" = OrderedDict()

    def touch_activity(self, clf_id: Optional[str] = None) -> None:
        now = time.time()
        self.last_activity_time = now
        if clf_id:
            with self._lock:
                self._active_lru[clf_id] = now
                self._active_lru.move_to_end(clf_id)

    def acquire_classifier(self, clf_id: str, lazy_load: bool = False) -> Optional[BaseClassifier]:
        """
        Return a loaded classifier ready for inference, managing multi-model memory pressure.
        """
        classifier = self.registry.get_classifier(clf_id, lazy_load=True)
        if classifier is None:
            return None

        if classifier.is_loaded():
            self.touch_activity(clf_id)
            return classifier

        # Need to load `classifier`. Check if hardware memory requires evicting an older loaded model.
        needed_mb = classifier.get_estimated_memory_mb()
        mem_info = BaseClassifier.get_system_memory_info()
        ram_avail = mem_info.get("ram_avail_mb", 4096.0)
        vram_avail = mem_info.get("vram_avail_mb", 0.0)
        vram_total = mem_info.get("vram_total_mb", 0.0)

        with self._lock:
            loaded_ids = [
                cid for cid in list(self._active_lru.keys())
                if cid != clf_id and self._is_clf_loaded(cid)
            ]

        # Evict LRU models if:
        # 1) User explicitly requested strict lazyLoad (only 1 model in memory at a time), OR
        # 2) Free RAM or CUDA VRAM would drop below safety headroom
        need_evict = bool(lazy_load and len(loaded_ids) >= 1)
        if not need_evict and loaded_ids:
            if (ram_avail - needed_mb) < self.MIN_FREE_RAM_MB:
                need_evict = True
            elif vram_total > 0 and (vram_avail - needed_mb) < self.MIN_FREE_VRAM_MB:
                need_evict = True

        if need_evict and loaded_ids:
            for victim_id in loaded_ids:
                log(
                    f"[MemoryManager] Evicting LRU classifier '{victim_id}' to make room for '{clf_id}' "
                    f"(avail RAM={ram_avail:.0f}MB, needed={needed_mb:.0f}MB, lazy_load={lazy_load})"
                )
                self.unload_classifier(victim_id)
                mem_info = BaseClassifier.get_system_memory_info()
                ram_avail = mem_info.get("ram_avail_mb", 4096.0)
                vram_avail = mem_info.get("vram_avail_mb", 0.0)
                if not lazy_load:
                    ram_ok = (ram_avail - needed_mb) >= self.MIN_FREE_RAM_MB
                    vram_ok = (vram_total == 0) or ((vram_avail - needed_mb) >= self.MIN_FREE_VRAM_MB)
                    if ram_ok and vram_ok:
                        break

        t0 = time.time()
        success, err = classifier.load_model()
        load_ms = int((time.time() - t0) * 1000)
        if not success:
            log(f"[MemoryManager] Failed to load '{clf_id}': {err}")
            return None

        self.touch_activity(clf_id)
        log(f"[MemoryManager] Loaded '{clf_id}' in {load_ms}ms (active models: {self.get_loaded_models()})")
        return classifier

    def _is_clf_loaded(self, clf_id: str) -> bool:
        inst = self.registry.get_classifier(clf_id, lazy_load=True)
        return bool(inst and inst.is_loaded())

    def unload_classifier(self, clf_id: str) -> None:
        with self._lock:
            self._active_lru.pop(clf_id, None)
        try:
            inst = self.registry.get_classifier(clf_id, lazy_load=True)
            if inst and inst.is_loaded():
                inst.unload_model()
            gc.collect()
        except Exception as exc:
            log(f"[MemoryManager] Error unloading {clf_id}: {exc}")

    def check_idle_unload(self, queue_empty: bool) -> None:
        if not queue_empty:
            self.last_activity_time = time.time()
            return
        with self._lock:
            if not self._active_lru:
                return
            elapsed_min = (time.time() - self.last_activity_time) / 60.0
            if elapsed_min < self.inactivity_timeout_minutes:
                return
            to_unload = list(self._active_lru.keys())

        for cid in to_unload:
            log(f"[MemoryManager] Unloading '{cid}' after {elapsed_min:.1f}m of inactivity")
            self.unload_classifier(cid)

    def get_loaded_models(self) -> List[str]:
        with self._lock:
            return [cid for cid in self._active_lru.keys() if self._is_clf_loaded(cid)]


# ============================================================================
# Dynamic Priority Queue & Continuous Single-Item Inference Engine
# ============================================================================

class PriorityClassificationEngine:
    """
    Manages the priority task queue, browser tab/viewport context, in-flight
    deduplication, and the background inference worker thread.
    """

    def __init__(self, cache: ContentCache, memory_manager: ModelMemoryManager):
        self.cache = cache
        self.memory_manager = memory_manager
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq_counter = 0

        # Active tab tracking per sourceId (and windowId -> activeTabId)
        # sourceId -> {"activeTabId": Any, "activeWindowId": Any, "activeTabsByWindow": Dict[Any, Any]}
        self._source_context: Dict[str, Dict[str, Any]] = {}

        # Pending tasks keyed by content_hash for automatic deduplication!
        # content_hash -> task_dict
        self._pending_by_hash: Dict[str, Dict[str, Any]] = {}

        # Currently executing task (0 or 1)
        self._in_flight_task: Optional[Dict[str, Any]] = None
        self._running = True

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="FvF-InferenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _compute_subscriber_priority(self, sub: Dict[str, Any]) -> int:
        """
        Compute priority tier for a subscriber:
        - 0 (P0): Active Tab + inViewport
        - 1 (P1): Active Tab + offscreen
        - 2 (P2): Background Tab / Inactive Window
        """
        source_id = sub.get("sourceId") or "local"
        tab_id = sub.get("tabId")
        win_id = sub.get("windowId")
        in_viewport = bool(sub.get("inViewport", False))

        ctx = self._source_context.get(source_id)
        if not ctx:
            # If no tab context received yet, treat tab as active
            is_active_tab = True
        else:
            active_tab = ctx.get("activeTabId")
            active_tabs_by_win = ctx.get("activeTabsByWindow") or {}
            if tab_id is None:
                is_active_tab = True
            elif active_tab is not None and tab_id == active_tab:
                is_active_tab = True
            elif win_id is not None and active_tabs_by_win.get(win_id) == tab_id:
                # Active tab in an open browser window (multi-window support)
                is_active_tab = True
            else:
                is_active_tab = False

        if is_active_tab:
            return 0 if in_viewport else 1
        return 2

    def _recompute_task_priority_unlocked(self, task: Dict[str, Any]) -> None:
        subs = task.get("subscribers") or []
        if not subs:
            task["priority"] = 99
            return
        best = min(self._compute_subscriber_priority(s) for s in subs)
        task["priority"] = best

    def _get_remaining_count_unlocked(self) -> int:
        """
        Return total number of subscriber items remaining (queued + in-flight).
        """
        queued_subs = sum(len(t.get("subscribers", [])) for t in self._pending_by_hash.values())
        inflight_subs = len(self._in_flight_task.get("subscribers", [])) if self._in_flight_task else 0
        return queued_subs + inflight_subs

    def emit_queue_status(self) -> None:
        with self._lock:
            remaining = self._get_remaining_count_unlocked()
            queued_unique = len(self._pending_by_hash)
            in_flight = 1 if self._in_flight_task is not None else 0
        send_message({
            "version": 2,
            "type": "queueStatus",
            "remainingCount": remaining,
            "queuedUniqueCount": queued_unique,
            "inFlightCount": in_flight,
            "timestamp": int(time.time() * 1000),
        })

    def update_tab_context(self, msg: Dict[str, Any]) -> None:
        source_id = msg.get("sourceId") or "local"
        active_tab_id = msg.get("activeTabId") if "activeTabId" in msg else msg.get("tabId")
        window_id = msg.get("windowId")

        with self._cond:
            ctx = self._source_context.setdefault(
                source_id,
                {"activeTabId": None, "activeWindowId": None, "activeTabsByWindow": {}},
            )
            if active_tab_id is not None:
                ctx["activeTabId"] = active_tab_id
            if window_id is not None:
                ctx["activeWindowId"] = window_id
                if active_tab_id is not None:
                    ctx["activeTabsByWindow"][window_id] = active_tab_id

            for task in self._pending_by_hash.values():
                self._recompute_task_priority_unlocked(task)
            self._cond.notify_all()

        log(f"[PriorityEngine] Updated tab context for source={source_id}: activeTab={active_tab_id}, window={window_id}")

    def boost_items(self, msg: Dict[str, Any]) -> None:
        source_id = msg.get("sourceId") or "local"
        tab_id = msg.get("tabId")
        page_id = msg.get("pageId")
        item_ids = set(msg.get("itemIds") or [])
        if not item_ids:
            return

        boosted = 0
        with self._cond:
            for task in self._pending_by_hash.values():
                changed = False
                for sub in task.get("subscribers", []):
                    if sub.get("itemId") in item_ids:
                        if (page_id and sub.get("pageId") == page_id) or (tab_id is not None and sub.get("tabId") == tab_id) or (not page_id and tab_id is None):
                            if not sub.get("inViewport"):
                                sub["inViewport"] = True
                                changed = True
                                boosted += 1
                if changed:
                    self._recompute_task_priority_unlocked(task)
            if boosted > 0:
                self._cond.notify_all()

        if boosted > 0:
            log(f"[PriorityEngine] Boosted {boosted} item(s) into viewport priority (P0) for tab={tab_id}")

    def drop_context(self, msg: Dict[str, Any]) -> int:
        source_id = msg.get("sourceId")
        tab_id = msg.get("tabId")
        page_id = msg.get("pageId")

        removed_subs = 0
        with self._cond:
            to_delete_hashes = []
            for chash, task in self._pending_by_hash.items():
                subs = task.get("subscribers") or []
                kept = []
                for sub in subs:
                    match_page = bool(page_id and sub.get("pageId") == page_id)
                    match_tab = bool(
                        tab_id is not None
                        and sub.get("tabId") == tab_id
                        and (not source_id or sub.get("sourceId") == source_id)
                        and not page_id
                    )
                    if match_page or match_tab:
                        removed_subs += 1
                    else:
                        kept.append(sub)
                if not kept:
                    to_delete_hashes.append(chash)
                elif len(kept) != len(subs):
                    task["subscribers"] = kept
                    self._recompute_task_priority_unlocked(task)

            for chash in to_delete_hashes:
                self._pending_by_hash.pop(chash, None)

            # Also prune subscribers on the currently in-flight task if their page/tab closed
            if self._in_flight_task:
                subs = self._in_flight_task.get("subscribers") or []
                kept = []
                for sub in subs:
                    match_page = bool(page_id and sub.get("pageId") == page_id)
                    match_tab = bool(
                        tab_id is not None
                        and sub.get("tabId") == tab_id
                        and (not source_id or sub.get("sourceId") == source_id)
                        and not page_id
                    )
                    if match_page or match_tab:
                        removed_subs += 1
                    else:
                        kept.append(sub)
                self._in_flight_task["subscribers"] = kept

            self._cond.notify_all()

        if removed_subs > 0:
            log(f"[PriorityEngine] Dropped {removed_subs} pending item(s) for closed/navigated context (tab={tab_id}, page={page_id})")
        self.emit_queue_status()
        return removed_subs

    def cancel_all(self) -> None:
        with self._cond:
            count = len(self._pending_by_hash)
            self._pending_by_hash.clear()
            if self._in_flight_task:
                self._in_flight_task["subscribers"] = []
            self._cond.notify_all()
        log(f"[PriorityEngine] Cleared all {count} queued tasks")
        self.emit_queue_status()

    def submit_item(self, item_msg: Dict[str, Any]) -> None:
        """
        Handle a reassembled single item on the I/O thread:
        1. Compute content SHA-256 hash.
        2. Check RAM `ContentCache` immediately before queueing.
           If all required classifiers for the item's ensembles are cached, emit
           `classifyItemResult` immediately (<1ms) and skip queueing.
        3. If already in-flight or queued with the same `content_hash`, attach as a
           subscriber and promote task priority if needed.
        4. Otherwise enqueue into the priority queue and wake the inference worker.
        """
        item_id = item_msg.get("itemId") or item_msg.get("id")
        if not item_id:
            return

        modality = item_msg.get("modality") or "image"
        source_id = item_msg.get("sourceId") or "local"
        window_id = item_msg.get("windowId")
        tab_id = item_msg.get("tabId")
        page_id = item_msg.get("pageId")
        in_viewport = bool(item_msg.get("inViewport", True))
        request_id = item_msg.get("requestId")
        job_id = item_msg.get("jobId")

        model_config = item_msg.get("model") or {}
        ensembles = _normalize_ensembles(model_config, modality)
        if not ensembles:
            return
        lazy_load = bool(model_config.get("lazyLoad", False))
        inactivity_timeout = float(model_config.get("inactivityTimeout", 15))
        self.memory_manager.inactivity_timeout_minutes = max(1.0, inactivity_timeout)

        raw_payload = item_msg.get("data") or item_msg.get("url") or item_msg.get("text") or ""

        # Decode or extract content and compute SHA-256 content hash
        decoded_input: Any = None
        decode_error: Optional[str] = None
        if modality == "image":
            pil_img, raw_bytes, err = decode_image_payload(raw_payload)
            decoded_input = pil_img
            decode_error = err
            content_hash = ContentCache.compute_content_hash("image", raw_bytes if raw_bytes else raw_payload, fallback_id=item_id)
        elif modality == "text":
            text_val = raw_payload if isinstance(raw_payload, str) else str(raw_payload)
            decoded_input = text_val
            if not text_val.strip():
                decode_error = "empty_text"
            content_hash = ContentCache.compute_content_hash("text", text_val, fallback_id=item_id)
        else:
            decoded_input = raw_payload
            content_hash = ContentCache.compute_content_hash(modality, str(raw_payload), fallback_id=item_id)

        self.cache.register_item_hash(item_id, content_hash)

        # Fast-path 1: Check RAM cache BEFORE touching the queue!
        all_cached, cached_ens_results, _missing = self.cache.evaluate_ensembles_from_cache(
            content_hash, ensembles, item_id=item_id
        )
        if all_cached and cached_ens_results:
            log(f"[Cache] Instant RAM cache hit (<1ms) for {modality} item: {str(item_id)[:65]}")
            self._emit_item_results(
                subscriber={
                    "sourceId": source_id,
                    "windowId": window_id,
                    "tabId": tab_id,
                    "pageId": page_id,
                    "itemId": item_id,
                    "requestId": request_id,
                    "jobId": job_id,
                    "modality": modality,
                    "ensembles": ensembles,
                },
                ensemble_results=cached_ens_results,
            )
            self.emit_queue_status()
            return

        # Fast-path 2: If decoding failed (e.g. corrupt image or rejected remote URL), return error result immediately
        if decode_error is not None or decoded_input is None:
            log(f"[PriorityEngine] Skipping queue for {str(item_id)[:65]} due to decode error: {decode_error}")
            err_results = [
                {
                    "ensembleId": ens.get("id") or f"default_{modality}",
                    "score": 0.5,
                    "label": "uncertain",
                    "display_label": "AI generated",
                    "classifiers": {},
                    "model": ",".join(ens.get("classifiers") or []),
                    "durationMs": 0,
                    "notes": decode_error or "invalid_input",
                }
                for ens in ensembles
            ]
            self._emit_item_results(
                subscriber={
                    "sourceId": source_id,
                    "windowId": window_id,
                    "tabId": tab_id,
                    "pageId": page_id,
                    "itemId": item_id,
                    "requestId": request_id,
                    "jobId": job_id,
                    "modality": modality,
                    "ensembles": ensembles,
                },
                ensemble_results=err_results,
            )
            self.emit_queue_status()
            return

        subscriber = {
            "sourceId": source_id,
            "windowId": window_id,
            "tabId": tab_id,
            "pageId": page_id,
            "itemId": item_id,
            "inViewport": in_viewport,
            "requestId": request_id,
            "jobId": job_id,
            "modality": modality,
            "ensembles": ensembles,
            "lazyLoad": lazy_load,
        }

        with self._cond:
            # Check if identical content is currently in-flight
            if self._in_flight_task and self._in_flight_task.get("content_hash") == content_hash:
                self._in_flight_task["subscribers"].append(subscriber)
                self._merge_ensembles_into_task_unlocked(self._in_flight_task, ensembles)
                log(f"[PriorityEngine] Attached duplicate item {str(item_id)[:50]} to in-flight task {content_hash[:10]}")
                remaining = self._get_remaining_count_unlocked()
            elif content_hash in self._pending_by_hash:
                task = self._pending_by_hash[content_hash]
                task["subscribers"].append(subscriber)
                self._merge_ensembles_into_task_unlocked(task, ensembles)
                self._recompute_task_priority_unlocked(task)
                self._cond.notify_all()
                log(f"[PriorityEngine] Deduplicated queued item {str(item_id)[:50]} (hash={content_hash[:10]}, priority=P{task['priority']})")
                remaining = self._get_remaining_count_unlocked()
            else:
                self._seq_counter += 1
                task = {
                    "content_hash": content_hash,
                    "modality": modality,
                    "decoded_input": decoded_input,
                    "subscribers": [subscriber],
                    "classifier_ids": self._collect_classifier_ids(ensembles),
                    "lazy_load": lazy_load,
                    "seq": self._seq_counter,
                    "priority": 1,
                    "created_at": time.time(),
                }
                self._recompute_task_priority_unlocked(task)
                self._pending_by_hash[content_hash] = task
                self._cond.notify_all()
                remaining = self._get_remaining_count_unlocked()

        send_message({
            "version": 2,
            "type": "queueStatus",
            "remainingCount": remaining,
            "queuedUniqueCount": len(self._pending_by_hash),
            "inFlightCount": 1 if self._in_flight_task is not None else 0,
            "timestamp": int(time.time() * 1000),
        })

    @staticmethod
    def _collect_classifier_ids(ensembles: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for ens in ensembles:
            cids = ens.get("classifiers") or []
            if isinstance(cids, str):
                cids = [cids]
            for c in cids:
                if c and c not in out:
                    out.append(c)
        return out

    def _merge_ensembles_into_task_unlocked(self, task: Dict[str, Any], ensembles: List[Dict[str, Any]]) -> None:
        existing = task.setdefault("classifier_ids", [])
        for cid in self._collect_classifier_ids(ensembles):
            if cid not in existing:
                existing.append(cid)

    def _pop_next_task_unlocked(self) -> Optional[Dict[str, Any]]:
        if not self._pending_by_hash:
            return None
        # Select task with lowest (priority, seq)
        best_hash = min(
            self._pending_by_hash.keys(),
            key=lambda h: (self._pending_by_hash[h]["priority"], self._pending_by_hash[h]["seq"]),
        )
        return self._pending_by_hash.pop(best_hash, None)

    def _worker_loop(self) -> None:
        log("[InferenceWorker] Background single-item inference thread started")
        while self._running:
            task: Optional[Dict[str, Any]] = None
            with self._cond:
                while self._running and not self._pending_by_hash:
                    self._cond.wait(timeout=15.0)
                    if not self._pending_by_hash:
                        # Check if warm models should be unloaded due to prolonged inactivity
                        self.memory_manager.check_idle_unload(queue_empty=True)
                if not self._running:
                    break
                task = self._pop_next_task_unlocked()
                self._in_flight_task = task

            if task is None:
                continue

            try:
                self._execute_single_task(task)
            except Exception as exc:
                log(f"[InferenceWorker] Error executing task: {exc}")
                import traceback
                log(traceback.format_exc())
            finally:
                with self._cond:
                    self._in_flight_task = None
                self.emit_queue_status()

    def _execute_single_task(self, task: Dict[str, Any]) -> None:
        content_hash = task["content_hash"]
        modality = task["modality"]
        decoded_input = task["decoded_input"]
        lazy_load = bool(task.get("lazy_load", False))

        # Snapshot classifier IDs needed
        with self._lock:
            classifier_ids = list(task.get("classifier_ids") or [])
            initial_subs = list(task.get("subscribers") or [])

        if not initial_subs:
            return

        first_item_id = initial_subs[0].get("itemId")
        clf_scores: Dict[str, Dict[str, Any]] = {}
        t0 = time.time()

        for cid in classifier_ids:
            # Check if this specific classifier is already cached for this content_hash
            cached_hit = self.cache.get_classifier_score(content_hash, cid, item_id=first_item_id)
            if cached_hit is not None:
                clf_scores[cid] = cached_hit
                continue

            # If all subscribers were dropped while we were running previous classifiers, abort early!
            with self._lock:
                if not task.get("subscribers"):
                    log(f"[InferenceWorker] Aborting in-flight task {content_hash[:10]} (all subscribers dropped)")
                    return

            classifier = self.memory_manager.acquire_classifier(cid, lazy_load=lazy_load)
            if classifier is None:
                log(f"[InferenceWorker] Classifier '{cid}' unavailable; skipping")
                continue

            supported = classifier.get_supported_modalities()
            if modality not in supported:
                log(f"[InferenceWorker] Classifier '{cid}' does not support modality '{modality}'")
                continue

            try:
                clf_label = classifier.get_label() or "AI generated"
            except Exception:
                clf_label = "AI generated"

            try:
                clf_name = classifier.get_model_name() or cid
            except Exception:
                clf_name = cid

            c_t0 = time.time()
            try:
                score = classifier.process_item(decoded_input, modality)
            except Exception as exc:
                log(f"[InferenceWorker] '{cid}' failed during process_item: {exc}")
                score = -1.0
            c_ms = int((time.time() - c_t0) * 1000)

            if score >= 0.0:
                clf_scores[cid] = {
                    "score": round(float(score), 4),
                    "label": clf_label,
                    "model_name": clf_name,
                    "durationMs": c_ms,
                }
                # Store in RAM cache for content_hash and all subscriber itemIds
                with self._lock:
                    current_subs = list(task.get("subscribers") or [])
                self.cache.set_classifier_score(
                    content_hash=content_hash,
                    classifier_id=cid,
                    score=score,
                    label=clf_label,
                    model_name=clf_name,
                    item_id=first_item_id,
                )
                for s in current_subs:
                    sid = s.get("itemId")
                    if sid:
                        self.cache.register_item_hash(sid, content_hash)

            if lazy_load:
                self.memory_manager.unload_classifier(cid)

        total_ms = int((time.time() - t0) * 1000)

        # Take final subscriber snapshot (including any subscribers that attached while in-flight)
        with self._lock:
            final_subscribers = list(task.get("subscribers") or [])

        for sub in final_subscribers:
            sub_ensembles = sub.get("ensembles") or []
            ens_results = build_ensemble_results(sub_ensembles, clf_scores, duration_ms=total_ms, cached=False)
            # Also populate legacy classification_cache for backward compatibility
            sub_item_id = sub.get("itemId")
            if sub_item_id:
                for er in ens_results:
                    classification_cache[(sub_item_id, er["ensembleId"])] = {
                        "label": er["label"],
                        "score": er["score"],
                        "display_label": er["display_label"],
                        "classifiers": er["classifiers"],
                        "model": er["model"],
                        "timestamp": time.time(),
                    }
            self._emit_item_results(sub, ens_results)

    def _emit_item_results(self, subscriber: Dict[str, Any], ensemble_results: List[Dict[str, Any]]) -> None:
        item_id = subscriber.get("itemId")
        modality = subscriber.get("modality") or "image"
        source_id = subscriber.get("sourceId") or "local"
        window_id = subscriber.get("windowId")
        tab_id = subscriber.get("tabId")
        page_id = subscriber.get("pageId")
        job_id = subscriber.get("jobId")
        request_id = subscriber.get("requestId")

        with self._lock:
            remaining = self._get_remaining_count_unlocked()

        # Emit single-item streaming result message
        results_payload = []
        for er in ensemble_results:
            res_obj = {
                "id": item_id,
                "modality": modality,
                "ensembleId": er["ensembleId"],
                "label": er["label"],
                "score": er["score"],
                "display_label": er["display_label"],
                "classifiers": er["classifiers"],
                "model": er["model"],
                "durationMs": er.get("durationMs", 0),
                "cached": er.get("cached", False),
            }
            if er.get("notes"):
                res_obj["notes"] = er["notes"]
            results_payload.append(res_obj)

        send_message({
            "version": 2,
            "type": "classifyItemResult",
            "sourceId": source_id,
            "windowId": window_id,
            "tabId": tab_id,
            "pageId": page_id,
            "itemId": item_id,
            "modality": modality,
            "jobId": job_id,
            "requestId": request_id,
            "remainingCount": remaining,
            "results": results_payload,
            "timestamp": int(time.time() * 1000),
        })


# ============================================================================
# Legacy EnsembleClassifier & classify_items (Backward Compatibility)
# ============================================================================

class EnsembleClassifier:
    def __init__(
        self,
        classifier_ids: List[str],
        weights: Optional[List[float]] = None,
        lazy_load: bool = False,
        inactivity_timeout_minutes: int = 15,
    ):
        self.registry = get_registry()
        self.classifier_ids = classifier_ids
        self.lazy_load = lazy_load
        self.inactivity_timeout_minutes = inactivity_timeout_minutes
        self.last_activity_time = time.time()

        if weights is None:
            self.weights = [1.0 / max(1, len(classifier_ids))] * len(classifier_ids)
        else:
            total = sum(weights) or 1.0
            self.weights = [w / total for w in weights]

    def classify_batch(self, inputs: List[Any], modality: str = 'image') -> tuple:
        self.last_activity_time = time.time()
        n_items = len(inputs)
        all_scores = []
        device_info = None

        for clf_id, weight in zip(self.classifier_ids, self.weights):
            classifier = memory_manager.acquire_classifier(clf_id, lazy_load=self.lazy_load)
            if classifier is None:
                continue
            supported = classifier.get_supported_modalities()
            if modality not in supported:
                continue

            try:
                classifier_label = classifier.get_label() or 'AI generated'
            except Exception:
                classifier_label = 'AI generated'

            if device_info is None:
                device_info = classifier.get_device_info()

            valid_indices = [
                i for i, inp in enumerate(inputs)
                if inp is not None and (modality != 'text' or inp != '')
            ]
            valid_inputs = [inputs[i] for i in valid_indices]
            if not valid_inputs:
                continue

            try:
                scores = classifier.process_batch(valid_inputs, modality)
                if scores is None:
                    scores = [-1.0] * len(valid_inputs)
            except Exception as e:
                log(f"[Ensemble] {clf_id} failed during process_batch: {e}")
                scores = [-1.0] * len(valid_inputs)

            full_scores = [-1.0] * n_items
            for idx, score in zip(valid_indices, scores):
                full_scores[idx] = score

            all_scores.append((clf_id, weight, full_scores, classifier_label))
            if self.lazy_load:
                memory_manager.unload_classifier(clf_id)

        results = []
        for i in range(n_items):
            item_scores = []
            item_weights = []
            classifier_details = {}
            classifier_labels = []

            for clf_id, weight, scores, classifier_label in all_scores:
                score = scores[i]
                if score >= 0:
                    item_scores.append(score)
                    item_weights.append(weight)
                    classifier_details[clf_id] = {
                        'score': round(score, 4),
                        'label': classifier_label or 'AI generated',
                    }
                    if classifier_label:
                        classifier_labels.append(classifier_label)

            if item_scores:
                weight_sum = sum(item_weights)
                if weight_sum > 0:
                    normalized_weights = [w / weight_sum for w in item_weights]
                    avg_score = sum(s * w for s, w in zip(item_scores, normalized_weights))
                else:
                    avg_score = 0.5

                label = 'ai' if avg_score >= 0.5 else 'real'
                display_label = next(
                    (name for name in classifier_labels if name and name != 'AI generated'),
                    (classifier_labels[0] if classifier_labels else 'AI generated'),
                )
                results.append({
                    'score': round(avg_score, 4),
                    'label': label,
                    'display_label': display_label,
                    'classifiers': classifier_details,
                    'ensemble_size': len(item_scores),
                })
            else:
                results.append({
                    'score': 0.5,
                    'label': 'uncertain',
                    'display_label': 'AI generated',
                    'classifiers': {},
                    'ensemble_size': 0,
                })

        return results, device_info


def _normalize_ensembles(model_config: Dict[str, Any], modality: str) -> List[Dict[str, Any]]:
    ensembles_by_modality = model_config.get('ensemblesByModality')
    if isinstance(ensembles_by_modality, dict) and modality in ensembles_by_modality:
        ensembles = ensembles_by_modality.get(modality)
        return ensembles if isinstance(ensembles, list) else []

    classifier_ids = model_config.get('classifiers')
    if classifier_ids:
        if isinstance(classifier_ids, str):
            classifier_ids = [classifier_ids]
        return [{
            'id': f'default_{modality}',
            'classifiers': classifier_ids,
            'weights': model_config.get('weights', None),
        }]

    default_clfs = ['fvf_heimdall04'] if modality == 'image' else ['text']
    return [{
        'id': f'default_{modality}',
        'classifiers': default_clfs,
        'weights': model_config.get('weights', None),
    }]


def classify_items(envelope: Dict[str, Any], job_id: Optional[str] = None, send_chunk=None) -> Dict[str, Any]:
    """
    Legacy batch classification handler (kept for backward compatibility with
    tests and legacy callers). Uses 1-by-1 streaming chunks and the shared
    RAM ContentCache & ModelMemoryManager.
    """
    req_id = envelope.get('requestId')
    job_key = job_id or req_id

    if job_key in cancelled_jobs:
        cancelled_jobs.discard(job_key)
        return {
            'version': 2,
            'type': 'classifyResult',
            'requestId': req_id,
            'timestamp': int(time.time() * 1000),
            'results': [],
            'errors': [{'type': 'cancelled', 'message': 'Classification was cancelled'}],
        }

    payload = envelope.get('payload') or {}
    items = payload.get('items') or []
    model_config = payload.get('model') or {}
    weights = model_config.get('weights', None)
    lazy_load = bool(model_config.get('lazyLoad', False))
    stream_results = bool(model_config.get('streamResults', True))
    inactivity_timeout = int(model_config.get('inactivityTimeout', 15))

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    image_items = [it for it in items if it.get('modality') == 'image' and (it.get('url') or it.get('data'))]
    text_items = [it for it in items if it.get('modality') == 'text' and (it.get('text') or it.get('data'))]
    other_items = [it for it in items if it not in image_items and it not in text_items]

    def emit_chunk(ensemble_id: str, modality: str, chunk_results: List[Dict[str, Any]], chunk_errors: List[Dict[str, Any]]) -> bool:
        if send_chunk and stream_results:
            send_chunk({
                'version': 2,
                'type': 'classifyResultChunk',
                'jobId': job_id or req_id,
                'ensembleId': ensemble_id,
                'modality': modality,
                'timestamp': int(time.time() * 1000),
                'results': chunk_results,
                'errors': chunk_errors,
            })
            return True
        return False

    # Process image items 1-by-1 with ContentCache lookup
    if image_items:
        image_ensembles = _normalize_ensembles(model_config, 'image')
        for item in image_items:
            if job_key in cancelled_jobs:
                cancelled_jobs.discard(job_key)
                break
            item_id = item.get('id')
            raw_url = item.get('data') or item.get('url') or ''
            img, raw_bytes, fetch_err = decode_image_payload(raw_url)
            chash = ContentCache.compute_content_hash('image', raw_bytes if raw_bytes else raw_url, fallback_id=item_id or '')
            if item_id:
                content_cache.register_item_hash(item_id, chash)

            all_cached, cached_ens, _ = content_cache.evaluate_ensembles_from_cache(chash, image_ensembles, item_id=item_id)
            if all_cached and cached_ens:
                for er in cached_ens:
                    r = {
                        'id': item_id,
                        'modality': 'image',
                        'ensembleId': er['ensembleId'],
                        'label': er['label'],
                        'score': er['score'],
                        'display_label': er['display_label'],
                        'classifiers': er['classifiers'],
                        'model': er['model'],
                        'durationMs': 0,
                        'cached': True,
                    }
                    if not emit_chunk(er['ensembleId'], 'image', [r], []):
                        results.append(r)
                continue

            for ensemble_cfg in image_ensembles:
                ensemble_id = ensemble_cfg.get('id') or 'image_ensemble'
                classifier_ids = ensemble_cfg.get('classifiers') or model_config.get('classifiers', [])
                if isinstance(classifier_ids, str):
                    classifier_ids = [classifier_ids]
                ensemble_weights = ensemble_cfg.get('weights', weights)
                if not classifier_ids:
                    continue

                if fetch_err or img is None:
                    r = {
                        'id': item_id,
                        'modality': 'image',
                        'ensembleId': ensemble_id,
                        'label': 'uncertain',
                        'score': 0.5,
                        'display_label': 'AI generated',
                        'model': ','.join(classifier_ids),
                        'notes': fetch_err or 'decode_failed',
                    }
                    if not emit_chunk(ensemble_id, 'image', [r], []):
                        results.append(r)
                    continue

                ensemble = EnsembleClassifier(classifier_ids, ensemble_weights, lazy_load, inactivity_timeout)
                t0 = time.time()
                batch_results, _ = ensemble.classify_batch([img], modality='image')
                classify_ms = int((time.time() - t0) * 1000)
                result_data = batch_results[0]
                for cid, cdetail in (result_data.get('classifiers') or {}).items():
                    content_cache.set_classifier_score(
                        chash, cid, cdetail.get('score', -1.0), cdetail.get('label', 'AI generated'), item_id=item_id
                    )
                r = {
                    'id': item_id,
                    'modality': 'image',
                    'ensembleId': ensemble_id,
                    'label': result_data['label'],
                    'score': result_data['score'],
                    'display_label': result_data.get('display_label', 'AI generated'),
                    'classifiers': result_data.get('classifiers', {}),
                    'model': ','.join(classifier_ids),
                    'durationMs': classify_ms,
                }
                if not emit_chunk(ensemble_id, 'image', [r], []):
                    results.append(r)

    if text_items:
        text_ensembles = _normalize_ensembles(model_config, 'text')
        for item in text_items:
            if job_key in cancelled_jobs:
                cancelled_jobs.discard(job_key)
                break
            item_id = item.get('id')
            text_val = item.get('text') or item.get('data') or ''
            chash = ContentCache.compute_content_hash('text', text_val, fallback_id=item_id or '')
            if item_id:
                content_cache.register_item_hash(item_id, chash)

            all_cached, cached_ens, _ = content_cache.evaluate_ensembles_from_cache(chash, text_ensembles, item_id=item_id)
            if all_cached and cached_ens:
                for er in cached_ens:
                    r = {
                        'id': item_id,
                        'modality': 'text',
                        'ensembleId': er['ensembleId'],
                        'label': er['label'],
                        'score': er['score'],
                        'display_label': er['display_label'],
                        'classifiers': er['classifiers'],
                        'model': er['model'],
                        'durationMs': 0,
                        'cached': True,
                    }
                    if not emit_chunk(er['ensembleId'], 'text', [r], []):
                        results.append(r)
                continue

            for ensemble_cfg in text_ensembles:
                ensemble_id = ensemble_cfg.get('id') or 'text_ensemble'
                classifier_ids = ensemble_cfg.get('classifiers') or model_config.get('classifiers', [])
                if isinstance(classifier_ids, str):
                    classifier_ids = [classifier_ids]
                ensemble_weights = ensemble_cfg.get('weights', weights)
                if not classifier_ids:
                    continue

                ensemble = EnsembleClassifier(classifier_ids, ensemble_weights, lazy_load, inactivity_timeout)
                t0 = time.time()
                batch_results, _ = ensemble.classify_batch([text_val], modality='text')
                classify_ms = int((time.time() - t0) * 1000)
                result_data = batch_results[0]
                for cid, cdetail in (result_data.get('classifiers') or {}).items():
                    content_cache.set_classifier_score(
                        chash, cid, cdetail.get('score', -1.0), cdetail.get('label', 'AI generated'), item_id=item_id
                    )
                r = {
                    'id': item_id,
                    'modality': 'text',
                    'ensembleId': ensemble_id,
                    'label': result_data['label'],
                    'score': result_data['score'],
                    'display_label': result_data.get('display_label', 'AI generated'),
                    'classifiers': result_data.get('classifiers', {}),
                    'model': ','.join(classifier_ids),
                    'durationMs': classify_ms,
                }
                if not emit_chunk(ensemble_id, 'text', [r], []):
                    results.append(r)

    for it in other_items:
        results.append({
            'id': it.get('id'),
            'modality': it.get('modality'),
            'label': 'uncertain',
            'score': 0.5,
            'model': 'none',
            'durationMs': 0,
            'notes': 'unsupported_modality',
        })

    return {
        'version': 2,
        'type': 'classifyResult',
        'requestId': req_id,
        'timestamp': int(time.time() * 1000),
        'results': results,
        'errors': errors,
    }


def list_models(request_id: Optional[str] = None) -> Dict[str, Any]:
    """List available classifiers from registry with fresh folder scan."""
    registry = get_registry()
    available = registry.list_available_details(force_rescan=True)

    repo_root = Path(__file__).parent.parent
    imm_dir = repo_root / 'Immage Models'
    pt_files = []
    if imm_dir.exists():
        pt_files = [p.name for p in imm_dir.glob('*.pt') if p.is_file()]

    response = {
        'ok': True,
        'classifiers': available,
        'model_files': sorted(pt_files),
    }
    if request_id:
        response['requestId'] = request_id
    return response


# ============================================================================
# Catalog & Updater Helpers
# ============================================================================

HOST_VERSION = "2.2.0"
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
    local_catalog_path = Path(__file__).resolve().parent / "official_models.json"
    catalog = None

    try:
        req = Request(
            OFFICIAL_CATALOG_URL,
            headers={"User-Agent": "FireVsFire-NativeHost/2.2", "Cache-Control": "no-cache"},
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
            if not f_entry.get("required", True):
                continue
            rel_path = f_entry.get("relative_path", "")
            safe_path = _resolve_safe_repo_path(rel_path)
            if safe_path is None or not safe_path.exists() or safe_path.stat().st_size == 0:
                all_present = False
                break
        model["installed"] = all_present
        if all_present:
            installed_ids.append(model.get("id"))

    catalog["installed_models"] = installed_ids
    catalog = _sync_catalog_with_local_folder(catalog)
    return catalog


def _sync_catalog_with_local_folder(catalog: Dict[str, Any]) -> Dict[str, Any]:
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
                    "required": True,
                })
            if detail.get("weight_path"):
                try:
                    rel_w = str(Path(detail["weight_path"]).resolve().relative_to(repo_root)).replace('\\', '/')
                except Exception:
                    rel_w = str(detail["weight_path"]).replace('\\', '/')
                files_list.append({
                    "type": "weights_pt" if rel_w.endswith(".pt") else "weights_onnx",
                    "relative_path": rel_w,
                    "required": True,
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
                "installed": weights_ready,
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
                if not f_entry.get("required", True):
                    continue
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
                "status": f"Verified {dest_path.name}",
            })
            downloaded_files.append(rel_path)
            continue

        urls_to_try = [primary_url]
        if fallback_url and _is_allowed_official_url(fallback_url) and fallback_url != primary_url:
            urls_to_try.append(fallback_url)

        last_err = None
        success = False
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        }

        def _get_ssl_contexts():
            contexts = []
            try:
                import certifi
                contexts.append(ssl.create_default_context(cafile=certifi.where()))
            except Exception:
                pass
            try:
                contexts.append(ssl.create_default_context())
            except Exception:
                pass
            try:
                contexts.append(ssl._create_unverified_context())
            except Exception:
                pass
            return contexts

        ssl_contexts = _get_ssl_contexts()

        for candidate_url in urls_to_try:
            if success:
                break
            for ctx_idx, ctx in enumerate(ssl_contexts):
                try:
                    log(f"[Downloader] Fetching {model_id} file {rel_path} from {candidate_url} (ssl_attempt={ctx_idx+1})")
                    req = Request(candidate_url, headers=browser_headers)
                    with urlopen(req, context=ctx, timeout=300) as response:
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
                                if now - last_emit >= 0.5:
                                    last_emit = now
                                    file_frac = (bytes_so_far / total_bytes) if total_bytes > 0 else 0.5
                                    overall_pct = int(((file_idx + file_frac) / total_files) * 100)
                                    try:
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
                                            "status": f"Downloading {dest_path.name}...",
                                        })
                                    except Exception:
                                        pass

                    if tmp_path.exists() and tmp_path.stat().st_size > 0:
                        os.replace(tmp_path, dest_path)
                        success = True
                        downloaded_files.append(rel_path)
                        log(f"[Downloader] Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
                        try:
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
                                "status": f"Installed {dest_path.name}",
                            })
                        except Exception:
                            pass
                        break
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    log(f"[Downloader] Error downloading {candidate_url} with ssl context #{ctx_idx+1}: {last_err}")
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    if "SSL" not in last_err and "CERTIFICATE" not in last_err:
                        break

        if not success:
            if dest_path.exists() and dest_path.stat().st_size > 0:
                downloaded_files.append(rel_path)
                continue
            if not f_entry.get("required", True):
                log(f"[Downloader] Optional file {rel_path} skipped: {last_err}")
                continue
            resp = {
                "ok": False,
                "modelId": model_id,
                "error": f"Failed to download {dest_path.name}: {last_err}",
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
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def check_for_host_updates(request_id: Optional[str] = None) -> Dict[str, Any]:
    catalog = _load_official_catalog()
    remote_version = catalog.get("host_version", HOST_VERSION)
    update_available = remote_version != HOST_VERSION
    resp = {
        "ok": True,
        "type": "checkUpdateResponse",
        "currentVersion": HOST_VERSION,
        "latestVersion": remote_version,
        "updateAvailable": update_available,
        "releaseNotes": catalog.get("release_notes", "Performance improvements and new model support."),
        "downloadUrl": "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/download/v2.0/FireVsFire-NativeHost.zip",
        "timestamp": int(time.time() * 1000),
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
        "native/classifiers/fvf_heimdall_04_classifier.py",
        "native/classifiers/fvf_heimdall_04_b_classifier.py",
        "native/classifiers/Resnet50_spider.py",
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
        "timestamp": int(time.time() * 1000),
    }
    if errors:
        resp["errors"] = errors
    if request_id:
        resp["requestId"] = request_id
    return resp


def get_system_power_status(request_id: Optional[str] = None) -> Dict[str, Any]:
    """Detect whether host is plugged in, battery percentage, and charging status."""
    is_plugged = True
    has_battery = False
    pct: Optional[int] = 100
    charging = False
    source = 'unknown'

    try:
        if sys.platform == 'win32':
            import ctypes
            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ('ACLineStatus', ctypes.c_byte),
                    ('BatteryFlag', ctypes.c_byte),
                    ('BatteryLifePercent', ctypes.c_byte),
                    ('SystemStatusFlag', ctypes.c_byte),
                    ('BatteryLifeTime', ctypes.c_ulong),
                    ('BatteryFullLifeTime', ctypes.c_ulong),
                ]
            s = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                is_plugged = (s.ACLineStatus == 1)
                has_battery = not bool(s.BatteryFlag & 128)
                pct = int(s.BatteryLifePercent) if s.BatteryLifePercent <= 100 else None
                charging = bool(s.BatteryFlag & 8)
                source = 'win32_api'
        elif sys.platform.startswith('linux'):
            power_dir = Path('/sys/class/power_supply')
            if power_dir.is_dir():
                for supply in power_dir.iterdir():
                    type_file = supply / 'type'
                    if type_file.is_file():
                        stype = type_file.read_text(encoding='utf-8').strip().lower()
                        if stype == 'mains':
                            online_file = supply / 'online'
                            if online_file.is_file():
                                is_plugged = (online_file.read_text(encoding='utf-8').strip() == '1')
                                source = 'linux_sysfs'
                        elif stype == 'battery':
                            has_battery = True
                            cap_file = supply / 'capacity'
                            if cap_file.is_file():
                                try:
                                    pct = int(cap_file.read_text(encoding='utf-8').strip())
                                except ValueError:
                                    pass
                            status_file = supply / 'status'
                            if status_file.is_file():
                                charging = (status_file.read_text(encoding='utf-8').strip().lower() == 'charging')
                            source = 'linux_sysfs'
    except Exception as e:
        safe_log(f"Error querying power status: {e}")

    resp: Dict[str, Any] = {
        'ok': True,
        'type': 'powerStatus',
        'is_plugged_in': is_plugged,
        'has_battery': has_battery,
        'battery_percent': pct,
        'charging': charging,
        'source': source
    }
    if request_id:
        resp['requestId'] = request_id
    return resp


# ============================================================================
# Global Singletons & Main I/O Loop
# ============================================================================

content_cache = ContentCache(max_entries=MAX_CACHE_SIZE)
memory_manager = ModelMemoryManager(inactivity_timeout_minutes=15.0)
packet_assembler = PacketAssembler()
job_assembler = JobAssembler()
priority_engine = PriorityClassificationEngine(cache=content_cache, memory_manager=memory_manager)


def main() -> None:
    """Main non-blocking I/O message loop for native messaging protocol."""
    log("[Main] Entering non-blocking I/O loop (V2.2 Dynamic Prioritisation)")

    while True:
        try:
            msg = read_message()
            if msg is None:
                log("[Main] read_message returned None, exiting")
                break

            mtype = msg.get('type')

            if mtype == 'ping' or mtype == 'PING':
                req_id = msg.get('requestId')
                resp = {
                    'ok': True,
                    'type': 'pong',
                    'version': HOST_VERSION,
                    'time': int(time.time() * 1000),
                }
                if req_id:
                    resp['requestId'] = req_id
                send_message(resp)
                continue

            # ------------------------------------------------------------------
            # Real-Time Single-Item Streaming & Context Priority Control Messages
            # ------------------------------------------------------------------
            elif mtype in ('submitItem', 'submitItemPacket'):
                assembled = packet_assembler.add_packet(msg)
                if assembled is not None:
                    priority_engine.submit_item(assembled)
                continue

            elif mtype == 'updateTabContext':
                priority_engine.update_tab_context(msg)
                continue

            elif mtype == 'boostItems':
                priority_engine.boost_items(msg)
                continue

            elif mtype == 'dropContext':
                packet_assembler.drop_context(
                    source_id=msg.get('sourceId'),
                    tab_id=msg.get('tabId'),
                    page_id=msg.get('pageId'),
                )
                priority_engine.drop_context(msg)
                continue

            elif mtype == 'getQueueStatus':
                priority_engine.emit_queue_status()
                continue

            # ------------------------------------------------------------------
            # Updater, Catalog, and Model Management Messages
            # ------------------------------------------------------------------
            elif mtype == 'checkUpdate' or mtype == 'checkNativeHostUpdate':
                req_id = msg.get('requestId')
                send_message(check_for_host_updates(req_id))
                continue

            elif mtype == 'applyUpdate' or mtype == 'applyNativeHostUpdate':
                req_id = msg.get('requestId')
                send_message(apply_host_updates(req_id))
                continue

            elif mtype == 'listModels':
                send_message(list_models(msg.get('requestId')))
                continue

            elif mtype == 'getOfficialCatalog':
                send_message(get_official_catalog_response(msg.get('requestId')))
                continue

            elif mtype == 'getPowerStatus':
                req_id = msg.get('requestId')
                send_message(get_system_power_status(req_id))
                continue

            elif mtype == 'downloadOfficialModel':
                model_id = msg.get('modelId')
                force_dl = bool(msg.get('forceDownload', False))
                req_id = msg.get('requestId')
                # Run download in a separate thread so we don't block the I/O loop
                threading.Thread(
                    target=lambda: send_message(download_official_model(model_id, req_id, force_dl)),
                    daemon=True,
                ).start()
                continue

            # ------------------------------------------------------------------
            # Legacy Batch / Job Compatibility
            # ------------------------------------------------------------------
            elif mtype == 'classifyJobChunk':
                job_id = msg.get('jobId') or msg.get('requestId') or f"job-{int(time.time()*1000)}"
                chunk_index = int(msg.get('chunkIndex', 0))
                total_chunks = int(msg.get('totalChunks', 1))
                payload = msg.get('payload') or {}

                envelope = job_assembler.add_chunk(job_id, chunk_index, total_chunks, payload)
                if envelope:
                    # Feed items from legacy job directly into priority engine or legacy runner
                    items = (envelope.get('payload') or {}).get('items') or []
                    model_cfg = (envelope.get('payload') or {}).get('model') or {}
                    tab_id = msg.get('tabId')
                    page_id = msg.get('pageId')
                    if msg.get('usePriorityEngine', False):
                        for it in items:
                            priority_engine.submit_item({
                                'itemId': it.get('id'),
                                'modality': it.get('modality', 'image'),
                                'data': it.get('data') or it.get('url') or it.get('text') or '',
                                'tabId': tab_id,
                                'pageId': page_id,
                                'jobId': job_id,
                                'inViewport': bool(it.get('inViewport', True)),
                                'model': model_cfg,
                            })
                    else:
                        classify_items(envelope, job_id=job_id, send_chunk=send_message)
                        send_message({
                            'version': 2,
                            'type': 'classifyJobComplete',
                            'jobId': job_id,
                            'timestamp': int(time.time() * 1000),
                        })
                continue

            elif mtype == 'classify':
                final_resp = classify_items(msg)
                send_message(final_resp)
                continue

            elif mtype == 'cancelJob':
                job_key = msg.get('jobId') or msg.get('requestId')
                if job_key == 'ALL':
                    priority_engine.cancel_all()
                elif job_key:
                    cancelled_jobs.add(job_key)
                send_message({
                    'ok': True,
                    'type': 'jobCancelled',
                    'jobId': job_key or 'ALL',
                    'timestamp': int(time.time() * 1000),
                })
                continue

            elif mtype == 'clearCache':
                content_cache.clear()
                send_message({
                    'ok': True,
                    'type': 'cacheCleared',
                    'timestamp': int(time.time() * 1000),
                })
                continue

            elif mtype == 'getCacheStats':
                send_message({
                    'ok': True,
                    'type': 'cacheStats',
                    'size': content_cache.size(),
                    'maxSize': MAX_CACHE_SIZE,
                    'loadedModels': memory_manager.get_loaded_models(),
                    'systemMemory': BaseClassifier.get_system_memory_info(),
                    'timestamp': int(time.time() * 1000),
                })
                continue

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
