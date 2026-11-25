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
  * P0: Hovered item under user pointer in active tab
  * P1: Active tab & in viewport (or boosted by scroll)
  * P2: Active tab & offscreen
  * P3: Background tab / inactive window
  * Instant queue purge on `dropContext` (tab closed or navigated away).
- Multi-Model Hardware Memory Manager: Keeps multiple simultaneous classifiers warm in RAM/VRAM
  when hardware permits, with automatic LRU eviction on low-memory systems and idle timeout unloading.
"""

import sys
import json
import copy
import struct
import time
import hashlib
import threading
import gc
import zipfile
from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from collections import OrderedDict
import base64
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Set
from PIL import Image
import os
import re
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

    def clear_classifier(self, classifier_id: str) -> None:
        if not classifier_id:
            return
        norm_target = classifier_id.replace("_", "")
        with self._lock:
            to_delete = [
                k for k in self._by_hash.keys()
                if k[1] == classifier_id or k[1].replace("_", "") == norm_target
            ]
            for k in to_delete:
                self._by_hash.pop(k, None)

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

        # Check checkpoint availability BEFORE evicting warm models
        find_ckpt = getattr(classifier, "_find_checkpoint", None)
        if callable(find_ckpt) and find_ckpt() is None:
            return None

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
        norm_target = clf_id.replace("_", "") if clf_id else ""
        with self._lock:
            keys_to_pop = [
                k for k in self._active_lru.keys()
                if k == clf_id or (norm_target and k.replace("_", "") == norm_target)
            ]
            for k in keys_to_pop:
                self._active_lru.pop(k, None)
        try:
            self.registry.unload_classifier(clf_id)
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

        # Currently hovered pending item per sourceId -> {"tabId": Any, "pageId": Any, "itemId": Any}
        self._hover_by_source: Dict[str, Dict[str, Any]] = {}

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
        - 0 (P0): Hovered item in active tab
        - 1 (P1): Active Tab + inViewport
        - 2 (P2): Active Tab + offscreen
        - 3 (P3): Background Tab / Inactive Window
        """
        source_id = sub.get("sourceId") or "local"
        tab_id = sub.get("tabId")
        win_id = sub.get("windowId")
        page_id = sub.get("pageId")
        item_id = sub.get("itemId")
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

        hover = self._hover_by_source.get(source_id)
        if (
            is_active_tab
            and hover
            and hover.get("itemId") is not None
            and item_id == hover["itemId"]
        ):
            hover_page = hover.get("pageId")
            hover_tab = hover.get("tabId")
            if hover_page:
                if page_id == hover_page:
                    return 0
            elif hover_tab is not None:
                if tab_id == hover_tab:
                    return 0
            else:
                return 0

        if is_active_tab:
            return 1 if in_viewport else 2
        return 3

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

    def set_hover_item(self, msg: Dict[str, Any]) -> None:
        source_id = msg.get("sourceId") or "local"
        item_id = msg.get("itemId")
        tab_id = msg.get("tabId")
        page_id = msg.get("pageId")

        got_p0 = False
        with self._cond:
            if not item_id:
                self._hover_by_source.pop(source_id, None)
            else:
                self._hover_by_source[source_id] = {
                    "tabId": tab_id,
                    "pageId": page_id,
                    "itemId": item_id,
                }

            for task in self._pending_by_hash.values():
                self._recompute_task_priority_unlocked(task)
                if item_id and task.get("priority") == 0:
                    if any(s.get("itemId") == item_id for s in (task.get("subscribers") or [])):
                        got_p0 = True

            self._cond.notify_all()

        if item_id and got_p0:
            log(f"[PriorityEngine] Hover boost (P0) for item={str(item_id)[:65]} tab={tab_id}")

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
            log(f"[PriorityEngine] Boosted {boosted} item(s) into viewport priority (P1) for tab={tab_id}")

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

            for sid in list(self._hover_by_source.keys()):
                if source_id and sid != source_id:
                    continue
                hover = self._hover_by_source[sid]
                match_page = bool(page_id and hover.get("pageId") == page_id)
                match_tab = bool(tab_id is not None and hover.get("tabId") == tab_id and not page_id)
                if match_page or match_tab:
                    self._hover_by_source.pop(sid, None)

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
                    "priority": 2,
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

    default_clfs = ['fvf_heimdall04'] if modality == 'image' else ['fvf_mimir_02']
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


def list_models(
    request_id: Optional[str] = None,
    check_updates: bool = False,
    force_update_check: bool = False,
) -> Dict[str, Any]:
    """List available classifiers from registry with fresh folder scan and optional update check."""
    registry = get_registry()
    available = [c for c in registry.list_available_details(force_rescan=True) if c.get("weights_ready")]

    repo_root = Path(__file__).parent.parent
    imm_dir = repo_root / 'Immage Models'
    pt_files = []
    if imm_dir.exists():
        pt_files = [p.name for p in imm_dir.glob('*.pt') if p.is_file()]

    response: Dict[str, Any] = {
        'ok': True,
        'classifiers': available,
        'model_files': sorted(pt_files),
    }
    if check_updates:
        try:
            response['updates'] = _compute_update_status(force_remote=force_update_check, timeout=2.5)
        except Exception as exc:
            log(f"[Updater] Failed piggybacked update check in list_models: {exc}")
    if request_id:
        response['requestId'] = request_id
    return response


# ============================================================================
# Catalog & Release-Diff Updater Helpers
# ============================================================================

HOST_VERSION = "2.2.0"
GITHUB_RELEASE_LATEST_API = "https://api.github.com/repos/LuckyCowMoo/Fire-vs-Fire/releases/latest"
GITHUB_RELEASES_API = "https://api.github.com/repos/LuckyCowMoo/Fire-vs-Fire/releases"
OFFICIAL_CATALOG_URL = "https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/FVF-clean/native/official_models.json"
DEFAULT_HOST_FILES = (
    "native/echo_host_V2.py",
    "native/model_registry.py",
    "native/official_models.json",
    "native/classifiers/base_classifier.py",
    "native/classifiers/_heimdall_arch.py",
    "native/classifiers/mimir_extractor.py",
)
ALLOWED_GITHUB_PREFIXES = (
    "https://raw.githubusercontent.com/LuckyCowMoo/Fire-vs-Fire/",
    "https://media.githubusercontent.com/media/LuckyCowMoo/Fire-vs-Fire/",
    "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/",
)
WEIGHT_EXTENSIONS = (".pt", ".onnx", ".joblib", ".bin", ".pth", ".safetensors")
_TEXT_FILE_EXTENSIONS = frozenset({
    ".py", ".json", ".bat", ".ps1", ".reg", ".txt", ".md",
    ".js", ".html", ".css", ".svg", ".yaml", ".yml",
})
_MACHINE_LOCAL_HOST_PATHS = frozenset({
    "manifest.json",
    "native/manifest.chrome.json",
    "native/manifest.firefox.json",
    "native/register-chrome-host.reg",
    "native/register-firefox-host.reg",
})
_HASH_CACHE_SIZE_THRESHOLD = 2 * 1024 * 1024
_CATALOG_REMOTE_CACHE_TTL = 600.0
_RELEASE_CACHE_TTL = 600.0


def _get_ssl_contexts() -> List[Any]:
    contexts: List[Any] = []
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


def _urlopen_response_with_ssl_fallback(req: Request, timeout: float = 5.0) -> Tuple[bytes, Dict[str, str]]:
    last_exc: Optional[Exception] = None
    for ctx in _get_ssl_contexts():
        try:
            with urlopen(req, context=ctx, timeout=timeout) as resp:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return resp.read(), hdrs
        except Exception as exc:
            last_exc = exc
            err_msg = f"{type(exc).__name__}: {exc}"
            if "SSL" not in err_msg and "CERTIFICATE" not in err_msg:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No SSL contexts available")


def _urlopen_with_ssl_fallback(req: Request, timeout: float = 5.0) -> bytes:
    body, _ = _urlopen_response_with_ssl_fallback(req, timeout=timeout)
    return body


def _normalize_repo_rel_path(rel_path: str) -> str:
    return str(rel_path or "").strip().lstrip("/").replace("\\", "/")


def _is_text_rel_path(rel_path: str) -> bool:
    norm = _normalize_repo_rel_path(rel_path)
    p = Path(norm)
    return p.suffix.lower() in _TEXT_FILE_EXTENSIONS or p.name.lower() == "license"


def _is_machine_local_host_path(rel_path: str) -> bool:
    norm = _normalize_repo_rel_path(rel_path).lower()
    if not norm or norm.endswith("/"):
        return True
    if norm in _MACHINE_LOCAL_HOST_PATHS:
        return True
    parts = [part for part in norm.split("/") if part]
    if any(part in (".venv", ".git", "__pycache__", "dist") or part.startswith(".") for part in parts):
        return True
    if norm.endswith(".log"):
        return True
    return False


def _compute_sha256_bytes(data: bytes, normalize_eol: bool = False) -> str:
    if normalize_eol:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest().lower()


def _normalize_digest_hex(raw_digest: Any) -> Optional[str]:
    if not isinstance(raw_digest, str):
        return None
    val = raw_digest.strip().lower()
    if val.startswith("sha256:"):
        val = val[7:]
    if len(val) == 64 and all(c in "0123456789abcdef" for c in val):
        return val
    return None


def _read_json_file_safe(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file_safe(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        log(f"[Updater] Failed writing {path.name}: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


_local_hash_mem_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _get_local_file_sha256(path: Path, normalize_eol: Optional[bool] = None) -> Optional[str]:
    global _local_hash_mem_cache
    if not path.is_file():
        return None
    if normalize_eol is None:
        normalize_eol = _is_text_rel_path(path.name)

    try:
        st = path.stat()
    except OSError:
        return None

    size = int(st.st_size)
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))

    if not normalize_eol and size >= _HASH_CACHE_SIZE_THRESHOLD:
        cache_path = Path(__file__).resolve().parent / ".local_file_hashes.json"
        if _local_hash_mem_cache is None:
            raw_cache = _read_json_file_safe(cache_path)
            _local_hash_mem_cache = {
                str(k): v for k, v in raw_cache.items() if isinstance(v, dict)
            }
        cache_key = str(path.resolve())
        cached = _local_hash_mem_cache.get(cache_key)
        if (
            cached
            and cached.get("size") == size
            and cached.get("mtime_ns") == mtime_ns
            and _normalize_digest_hex(cached.get("sha256"))
        ):
            return _normalize_digest_hex(cached.get("sha256"))

    try:
        if normalize_eol:
            digest = _compute_sha256_bytes(path.read_bytes(), normalize_eol=True)
        else:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            digest = h.hexdigest().lower()
    except OSError:
        return None

    if not normalize_eol and size >= _HASH_CACHE_SIZE_THRESHOLD and _local_hash_mem_cache is not None:
        cache_path = Path(__file__).resolve().parent / ".local_file_hashes.json"
        _local_hash_mem_cache[str(path.resolve())] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "sha256": digest,
        }
        _write_json_file_safe(cache_path, _local_hash_mem_cache)

    return digest


_release_remote_cache: Optional[Dict[str, Any]] = None
_release_remote_cache_time: float = 0.0


def _release_cache_path() -> Path:
    return Path(__file__).resolve().parent / ".release_cache.json"


def _normalize_github_release(raw_rel: Dict[str, Any], etag: Optional[str] = None) -> Dict[str, Any]:
    assets_by_name: Dict[str, Dict[str, Any]] = {}
    for asset in raw_rel.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").strip()
        if not name:
            continue
        assets_by_name[name] = {
            "name": name,
            "size": int(asset.get("size") or 0),
            "digest": _normalize_digest_hex(asset.get("digest")),
            "updated_at": str(asset.get("updated_at") or ""),
            "browser_download_url": str(asset.get("browser_download_url") or ""),
        }

    return {
        "tag_name": str(raw_rel.get("tag_name") or ""),
        "name": str(raw_rel.get("name") or raw_rel.get("tag_name") or ""),
        "html_url": str(raw_rel.get("html_url") or "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases"),
        "target_commitish": str(raw_rel.get("target_commitish") or ""),
        "published_at": str(raw_rel.get("published_at") or ""),
        "body": str(raw_rel.get("body") or ""),
        "assets_by_name": assets_by_name,
        "etag": etag or str(raw_rel.get("etag") or ""),
    }


def _find_native_host_zip_asset(release: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(release, dict):
        return None
    assets = release.get("assets_by_name") or {}
    if "FireVsFire-NativeHost.zip" in assets:
        return assets["FireVsFire-NativeHost.zip"]
    for name, info in assets.items():
        low = name.lower()
        if low.startswith("firevsfire-nativehost") and low.endswith(".zip"):
            return info
    return None


def _fetch_latest_github_release(force: bool = False, timeout: float = 4.0) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Fetch current GitHub release metadata with ETag conditional validation."""
    global _release_remote_cache, _release_remote_cache_time
    now = time.time()
    if (
        not force
        and _release_remote_cache is not None
        and (now - _release_remote_cache_time) < _RELEASE_CACHE_TTL
    ):
        return copy.deepcopy(_release_remote_cache), True

    disk_cache = _read_json_file_safe(_release_cache_path())
    cached_rel = disk_cache.get("release") if isinstance(disk_cache.get("release"), dict) else None
    cached_etag = str(disk_cache.get("etag") or "")

    headers = {
        "User-Agent": f"FireVsFire-NativeHost/{HOST_VERSION}",
        "Accept": "application/vnd.github+json",
    }
    if cached_etag and cached_rel:
        headers["If-None-Match"] = cached_etag

    try:
        req = Request(GITHUB_RELEASE_LATEST_API, headers=headers)
        try:
            raw_bytes, resp_headers = _urlopen_response_with_ssl_fallback(req, timeout=timeout)
            raw_rel = json.loads(raw_bytes.decode("utf-8"))
            new_etag = resp_headers.get("etag", "")
        except HTTPError as http_err:
            if http_err.code == 304 and cached_rel is not None:
                _release_remote_cache = copy.deepcopy(cached_rel)
                _release_remote_cache_time = now
                log("[ReleaseCheck] GitHub release metadata unchanged (304 Not Modified)")
                return copy.deepcopy(cached_rel), True
            if http_err.code == 404:
                req_all = Request(GITHUB_RELEASES_API, headers={
                    "User-Agent": f"FireVsFire-NativeHost/{HOST_VERSION}",
                    "Accept": "application/vnd.github+json",
                })
                raw_bytes, resp_headers = _urlopen_response_with_ssl_fallback(req_all, timeout=timeout)
                rel_list = json.loads(raw_bytes.decode("utf-8"))
                non_drafts = [r for r in (rel_list or []) if isinstance(r, dict) and not r.get("draft")]
                if not non_drafts:
                    raise RuntimeError("No published releases found on GitHub")
                raw_rel = non_drafts[0]
                new_etag = resp_headers.get("etag", "")
            else:
                raise

        if isinstance(raw_rel, dict) and raw_rel.get("tag_name"):
            normalized = _normalize_github_release(raw_rel, etag=new_etag)
            _release_remote_cache = normalized
            _release_remote_cache_time = now
            disk_cache["etag"] = new_etag
            disk_cache["release"] = normalized
            _write_json_file_safe(_release_cache_path(), disk_cache)
            log(f"[ReleaseCheck] Loaded GitHub release '{normalized.get('tag_name')}' metadata")
            return copy.deepcopy(normalized), True
    except Exception as exc:
        log(f"[ReleaseCheck] Failed fetching GitHub release metadata: {exc}")

    if _release_remote_cache is not None:
        return copy.deepcopy(_release_remote_cache), False
    if cached_rel is not None:
        return copy.deepcopy(cached_rel), False
    return None, False


def _asset_cache_key(asset: Dict[str, Any]) -> str:
    return str(
        asset.get("digest")
        or f"size:{asset.get('size', 0)}|at:{asset.get('updated_at', '')}"
    )


def _manifest_from_zip_bytes(raw_bytes: bytes, asset: Dict[str, Any], release: Dict[str, Any]) -> Dict[str, Any]:
    files_hashes: Dict[str, str] = {}
    with zipfile.ZipFile(BytesIO(raw_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            norm = _normalize_repo_rel_path(info.filename)
            if _is_machine_local_host_path(norm):
                continue
            entry_bytes = zf.read(info.filename)
            files_hashes[norm] = _compute_sha256_bytes(
                entry_bytes,
                normalize_eol=_is_text_rel_path(norm),
            )
    return {
        "asset_name": asset.get("name") or "FireVsFire-NativeHost.zip",
        "asset_key": _asset_cache_key(asset),
        "asset_digest": asset.get("digest") or "",
        "asset_size": int(asset.get("size") or len(raw_bytes)),
        "asset_updated_at": asset.get("updated_at") or "",
        "tag_name": release.get("tag_name") or "",
        "files": files_hashes,
    }


def _get_release_zip_manifest(release: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    asset = _find_native_host_zip_asset(release)
    if not asset:
        return None
    target_key = _asset_cache_key(asset)
    disk_cache = _read_json_file_safe(_release_cache_path())
    cached_manifest = disk_cache.get("zip_manifest")
    if (
        isinstance(cached_manifest, dict)
        and cached_manifest.get("asset_key") == target_key
        and isinstance(cached_manifest.get("files"), dict)
        and cached_manifest.get("files")
    ):
        return cached_manifest

    dl_url = str(asset.get("browser_download_url") or "")
    if not dl_url or not _is_allowed_official_url(dl_url):
        return None

    try:
        req = Request(
            dl_url,
            headers={"User-Agent": f"FireVsFire-NativeHost/{HOST_VERSION}", "Cache-Control": "no-cache"},
        )
        raw_bytes = _urlopen_with_ssl_fallback(req, timeout=timeout)
        expected_digest = asset.get("digest")
        if expected_digest:
            actual_digest = hashlib.sha256(raw_bytes).hexdigest().lower()
            if actual_digest != expected_digest:
                raise RuntimeError(f"Release zip SHA-256 mismatch ({actual_digest[:12]} != {expected_digest[:12]})")
        manifest = _manifest_from_zip_bytes(raw_bytes, asset, release)
        disk_cache["zip_manifest"] = manifest
        _write_json_file_safe(_release_cache_path(), disk_cache)
        log(f"[ReleaseCheck] Indexed {len(manifest['files'])} files from release archive {asset.get('name')}")
        return manifest
    except Exception as exc:
        log(f"[ReleaseCheck] Failed fetching release zip manifest: {exc}")
        return None


def _parse_model_family_and_rank(model: Dict[str, Any]) -> Tuple[Optional[str], int]:
    if not isinstance(model, dict):
        return None, -1
    raw_name = str(model.get("name") or model.get("id") or "")
    m = re.search(
        r"(heimdall|mimir|mímir|huginn|bragi)[\s_v-]*0*(\d+)(?:[\s_-]*([a-z])(?:\b|$))?",
        raw_name,
        re.IGNORECASE,
    )
    if not m:
        return None, -1
    family = m.group(1).lower()
    if family == "mímir":
        family = "mimir"
    major = int(m.group(2))
    suffix = (ord(m.group(3).lower()) - 96) if m.group(3) else 0
    return family, major * 100 + suffix


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


_catalog_remote_cache: Optional[Dict[str, Any]] = None
_catalog_remote_cache_time: float = 0.0


def _load_local_catalog_raw() -> Optional[Dict[str, Any]]:
    local_catalog_path = Path(__file__).resolve().parent / "official_models.json"
    if not local_catalog_path.exists():
        return None
    try:
        with open(local_catalog_path, "r", encoding="utf-8") as fh:
            local_data = json.load(fh)
            if isinstance(local_data, dict) and isinstance(local_data.get("models"), list):
                return copy.deepcopy(local_data)
    except Exception as exc:
        log(f"[Catalog] Failed to load local official_models.json: {exc}")
    return None


def _fetch_remote_catalog(force: bool = False, timeout: float = 2.5) -> Tuple[Optional[Dict[str, Any]], bool]:
    global _catalog_remote_cache, _catalog_remote_cache_time
    now = time.time()
    if (
        not force
        and _catalog_remote_cache is not None
        and (now - _catalog_remote_cache_time) < _CATALOG_REMOTE_CACHE_TTL
    ):
        return copy.deepcopy(_catalog_remote_cache), True

    try:
        req = Request(
            OFFICIAL_CATALOG_URL,
            headers={"User-Agent": f"FireVsFire-NativeHost/{HOST_VERSION}", "Cache-Control": "no-cache"},
        )
        raw = _urlopen_with_ssl_fallback(req, timeout=timeout).decode("utf-8")
        candidate = json.loads(raw)
        if isinstance(candidate, dict) and isinstance(candidate.get("models"), list):
            _catalog_remote_cache = candidate
            _catalog_remote_cache_time = now
            log("[Catalog] Loaded latest official_models.json from GitHub")
            return copy.deepcopy(candidate), True
    except Exception as exc:
        log(f"[Catalog] Remote GitHub fetch fell back to cache/local: {exc}")

    if _catalog_remote_cache is not None:
        return copy.deepcopy(_catalog_remote_cache), False
    return None, False


def _installed_versions_path() -> Path:
    return Path(__file__).resolve().parent / ".installed_versions.json"


def _read_installed_versions() -> Dict[str, str]:
    path = _installed_versions_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if k and v}
    except Exception as exc:
        log(f"[Catalog] Failed to read .installed_versions.json: {exc}")
    return {}


def _write_installed_versions(versions: Dict[str, str]) -> None:
    path = _installed_versions_path()
    tmp_path = path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(versions, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception as exc:
        log(f"[Catalog] Failed to write .installed_versions.json: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _record_installed_model_version(model_id: str, version: str) -> None:
    if not model_id or not version:
        return
    versions = _read_installed_versions()
    versions[model_id] = version
    _write_installed_versions(versions)


def _remove_installed_model_version(model_id: str) -> None:
    if not model_id:
        return
    versions = _read_installed_versions()
    if model_id in versions:
        versions.pop(model_id, None)
        _write_installed_versions(versions)


def _merge_remote_into_catalog(
    local_catalog: Optional[Dict[str, Any]],
    remote_catalog: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if local_catalog is None and remote_catalog is None:
        return {"catalog_version": 2, "repository": "LuckyCowMoo/Fire-vs-Fire", "models": []}
    if local_catalog is None:
        return copy.deepcopy(remote_catalog)  # type: ignore[arg-type]
    if remote_catalog is None:
        return copy.deepcopy(local_catalog)

    merged = copy.deepcopy(local_catalog)
    for key in ("catalog_version", "repository", "branch", "author", "updated_at", "host_version", "release_notes", "host_files"):
        if key in remote_catalog:
            merged[key] = copy.deepcopy(remote_catalog[key])

    remote_by_id = {
        m.get("id"): m for m in remote_catalog.get("models", []) if isinstance(m, dict) and m.get("id")
    }
    updated_models: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for m in merged.get("models", []):
        if not isinstance(m, dict) or not m.get("id"):
            continue
        mid = m["id"]
        seen_ids.add(mid)
        updated_models.append(copy.deepcopy(remote_by_id[mid]) if mid in remote_by_id else m)

    for rm in remote_catalog.get("models", []):
        if isinstance(rm, dict) and rm.get("id") and rm["id"] not in seen_ids:
            updated_models.append(copy.deepcopy(rm))
            seen_ids.add(rm["id"])

    merged["models"] = updated_models
    return merged


def _load_official_catalog(force_remote: bool = False) -> Dict[str, Any]:
    catalog = _load_local_catalog_raw()
    remote_catalog: Optional[Dict[str, Any]] = None
    if force_remote or catalog is None:
        remote_catalog, _ = _fetch_remote_catalog(force=force_remote, timeout=3.0)
    elif _catalog_remote_cache is not None:
        remote_catalog = copy.deepcopy(_catalog_remote_cache)

    merged = _merge_remote_into_catalog(catalog, remote_catalog)
    return _sync_catalog_with_local_folder(merged)


def _evaluate_model_disk_status(model: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return (is_installed, formatted_disk_size) for a catalog model entry."""
    files = model.get("files") or []
    if model.get("status") != "available" or not files:
        return False, None

    has_weight_entry = any(
        Path(f_entry.get("relative_path", "")).suffix.lower() in WEIGHT_EXTENSIONS
        for f_entry in files
    )
    if not has_weight_entry:
        return False, None

    for f_entry in files:
        if not f_entry.get("required", True):
            continue
        rel_path = f_entry.get("relative_path", "")
        safe_path = _resolve_safe_repo_path(rel_path)
        if safe_path is None or not safe_path.exists() or safe_path.stat().st_size == 0:
            return False, None

    total_weight_bytes = 0
    for f_entry in files:
        rel_path = f_entry.get("relative_path", "")
        if not rel_path or Path(rel_path).suffix.lower() not in WEIGHT_EXTENSIONS:
            continue
        safe_path = _resolve_safe_repo_path(rel_path)
        if safe_path is not None and safe_path.is_file() and safe_path.stat().st_size > 0:
            total_weight_bytes += safe_path.stat().st_size

    disk_size = f"{round(total_weight_bytes / (1024 * 1024))} MB" if total_weight_bytes > 0 else None
    return True, disk_size


def _sync_catalog_with_local_folder(catalog: Dict[str, Any]) -> Dict[str, Any]:
    registry = get_registry()
    details = registry.list_available_details(force_rescan=True)
    models = catalog.setdefault("models", [])
    models_by_id = {m.get("id"): m for m in models if isinstance(m, dict) and m.get("id")}
    repo_root = Path(__file__).resolve().parent.parent

    for detail in details:
        cid = detail["id"]
        weights_ready = bool(detail.get("weights_ready", False))

        matched_id = cid if cid in models_by_id else None
        if not matched_id:
            for mid in models_by_id:
                if mid == cid or mid.replace("_", "") == cid.replace("_", ""):
                    matched_id = mid
                    break
                if cid in ("fvf_mimir", "mimir") and "mimir" in mid:
                    matched_id = mid
                    break

        if matched_id:
            m = models_by_id[matched_id]
            m["installed"] = weights_ready
        elif weights_ready and detail.get("weight_path"):
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
                "installed": True,
            }

            if is_recommended:
                models.insert(0, new_model)
            else:
                models.append(new_model)
            models_by_id[cid] = new_model

    installed_ids = []
    for m in models:
        all_present, disk_size = _evaluate_model_disk_status(m)
        m["installed"] = all_present
        if all_present:
            if disk_size:
                m["disk_size"] = disk_size
            installed_ids.append(m.get("id"))
        else:
            m.pop("disk_size", None)

    catalog["installed_models"] = sorted(list(set(installed_ids)))
    return catalog


def _resolve_release_asset_for_file(f_entry: Dict[str, Any], release: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return matching GitHub release asset metadata and asset filename for a catalog file entry."""
    rel_path = str(f_entry.get("relative_path") or "")
    primary_url = str(f_entry.get("url") or "")
    parsed_name = Path(urllib.parse.urlparse(primary_url).path).name if primary_url else ""
    asset_name = parsed_name or Path(rel_path).name
    assets = (release or {}).get("assets_by_name") or {}
    return assets.get(asset_name), asset_name


def get_official_catalog_response(request_id: Optional[str] = None) -> Dict[str, Any]:
    catalog = _load_official_catalog()
    try:
        status = _compute_update_status(force_remote=False, timeout=2.5)
        release_outdated_ids = {
            item.get("installedId")
            for item in (status.get("classifiers") or {}).get("outdated") or []
            if item.get("reason") == "release_updated" and item.get("installedId")
        }
        for m in catalog.get("models") or []:
            if isinstance(m, dict) and m.get("installed") and m.get("id") in release_outdated_ids:
                m["update_available"] = True
                m["update_reason"] = "release_updated"
    except Exception as exc:
        log(f"[Catalog] Skipped annotating classifier updates in getOfficialCatalog: {exc}")

    registry = get_registry()
    classifiers = [c for c in registry.list_available_details(force_rescan=False) if c.get("weights_ready")]
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

    # The total download size is already known and displayed on the model tile
    catalog_mb = float(target_model.get("size_mb") or 0.0)
    known_total_bytes = int(catalog_mb * 1024 * 1024) if catalog_mb > 0 else 0

    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }

    ssl_contexts = _get_ssl_contexts()
    total_files = len(files)
    downloaded_files = []

    cumulative_bytes = 0
    effective_total_bytes = known_total_bytes

    def _emit_progress(done_bytes: int, total_bytes: int, pct_override: Optional[int] = None) -> None:
        tot = max(done_bytes, total_bytes)
        if pct_override is not None:
            pct = pct_override
        elif tot > 0:
            pct = int((done_bytes / tot) * 100)
        else:
            pct = 0
        pct = max(0, min(100, pct))
        mb_done = done_bytes / (1024 * 1024)
        mb_tot = tot / (1024 * 1024)
        try:
            send_message({
                "version": 2,
                "type": "downloadProgress",
                "requestId": request_id,
                "modelId": model_id,
                "bytesDownloaded": done_bytes,
                "totalBytes": tot,
                "percent": pct,
                "status": f"{mb_done:.1f} / {mb_tot:.1f} MB",
            })
        except Exception:
            pass

    latest_release, _ = _fetch_latest_github_release(force=False, timeout=3.0)

    # Immediately emit 0% with known total size on tile
    _emit_progress(0, effective_total_bytes, pct_override=0)

    for file_idx, f_entry in enumerate(files):
        rel_path = f_entry.get("relative_path", "")
        primary_url = f_entry.get("url", "")
        fallback_url = f_entry.get("fallback_url", "")
        release_asset, _ = _resolve_release_asset_for_file(f_entry, latest_release)
        expected_sha256 = (release_asset or {}).get("digest")

        if release_asset and release_asset.get("browser_download_url"):
            rel_dl_url = str(release_asset["browser_download_url"])
            if _is_allowed_official_url(rel_dl_url):
                if primary_url and primary_url != rel_dl_url and not fallback_url:
                    fallback_url = primary_url
                primary_url = rel_dl_url

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

        local_matches_release = True
        if dest_path.exists() and dest_path.stat().st_size > 0 and expected_sha256:
            local_digest = _get_local_file_sha256(dest_path, normalize_eol=False)
            local_matches_release = (local_digest == expected_sha256)

        if not force_download and dest_path.exists() and dest_path.stat().st_size > 0 and local_matches_release:
            file_size = dest_path.stat().st_size
            cumulative_bytes += file_size
            if cumulative_bytes > effective_total_bytes:
                effective_total_bytes = cumulative_bytes
            is_last = (file_idx == total_files - 1)
            _emit_progress(cumulative_bytes, effective_total_bytes, pct_override=100 if is_last else None)
            downloaded_files.append(rel_path)
            continue

        urls_to_try = [primary_url]
        if fallback_url and _is_allowed_official_url(fallback_url) and fallback_url != primary_url:
            urls_to_try.append(fallback_url)

        last_err = None
        success = False
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        for candidate_url in urls_to_try:
            if success:
                break
            for ctx_idx, ctx in enumerate(ssl_contexts):
                try:
                    log(f"[Downloader] Fetching {model_id} file {rel_path} from {candidate_url} (ssl_attempt={ctx_idx+1})")
                    req = Request(candidate_url, headers=browser_headers)
                    with urlopen(req, context=ctx, timeout=300) as response:
                        file_bytes_so_far = 0
                        last_emit = time.time()
                        with open(tmp_path, "wb") as out_f:
                            while True:
                                chunk = response.read(65536)
                                if not chunk:
                                    break
                                out_f.write(chunk)
                                chunk_len = len(chunk)
                                file_bytes_so_far += chunk_len
                                cumulative_bytes += chunk_len
                                if cumulative_bytes > effective_total_bytes:
                                    effective_total_bytes = cumulative_bytes
                                now = time.time()
                                if now - last_emit >= 0.15:
                                    last_emit = now
                                    raw_pct = int((cumulative_bytes / effective_total_bytes) * 100) if effective_total_bytes > 0 else 0
                                    _emit_progress(cumulative_bytes, effective_total_bytes, pct_override=min(99, max(0, raw_pct)))

                    if tmp_path.exists() and tmp_path.stat().st_size > 0:
                        os.replace(tmp_path, dest_path)
                        success = True
                        downloaded_files.append(rel_path)
                        actual_size = dest_path.stat().st_size
                        drift = actual_size - file_bytes_so_far
                        if drift != 0:
                            cumulative_bytes += drift
                            if cumulative_bytes > effective_total_bytes:
                                effective_total_bytes = cumulative_bytes
                        log(f"[Downloader] Successfully downloaded {dest_path.name} ({actual_size} bytes)")
                        is_last = (file_idx == total_files - 1)
                        raw_pct = int((cumulative_bytes / effective_total_bytes) * 100) if effective_total_bytes > 0 else (100 if is_last else 0)
                        _emit_progress(cumulative_bytes, effective_total_bytes, pct_override=100 if is_last else min(99, raw_pct))
                        break
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    log(f"[Downloader] Error downloading {candidate_url} with ssl context #{ctx_idx+1}: {last_err}")
                    cumulative_bytes -= file_bytes_so_far
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    if "SSL" not in last_err and "CERTIFICATE" not in last_err:
                        break

        if not success:
            if dest_path.exists() and dest_path.stat().st_size > 0:
                actual_size = dest_path.stat().st_size
                cumulative_bytes += actual_size
                if cumulative_bytes > effective_total_bytes:
                    effective_total_bytes = cumulative_bytes
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

    final_total = max(cumulative_bytes, effective_total_bytes)
    _emit_progress(final_total, final_total, pct_override=100)

    _record_installed_model_version(model_id, str(target_model.get("version") or ""))

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


def uninstall_official_model(model_id: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Uninstall a downloaded model by unloading it from memory and removing its local weight files."""
    if not model_id or not isinstance(model_id, str):
        resp = {"ok": False, "error": "Invalid modelId"}
        if request_id:
            resp["requestId"] = request_id
        return resp

    log(f"[Uninstaller] Request to uninstall model '{model_id}'")
    catalog = _load_official_catalog()
    models = catalog.get("models", [])
    target_model = next((m for m in models if m.get("id") == model_id), None)
    if target_model is None:
        for m in models:
            mid = m.get("id")
            if mid and (mid == model_id or mid.replace("_", "") == model_id.replace("_", "")):
                target_model = m
                break

    canonical_id = target_model.get("id") if target_model else model_id

    # Step 1: Unload from memory_manager and registry BEFORE unlinking files so Windows file locks are released
    registry = get_registry()
    try:
        memory_manager.unload_classifier(model_id)
        if canonical_id and canonical_id != model_id:
            memory_manager.unload_classifier(canonical_id)
        registry.unload_classifier(model_id)
        if canonical_id and canonical_id != model_id:
            registry.unload_classifier(canonical_id)
        if target_model and target_model.get("category_name"):
            memory_manager.unload_classifier(target_model["category_name"])
            registry.unload_classifier(target_model["category_name"])
        content_cache.clear_classifier(model_id)
        if canonical_id and canonical_id != model_id:
            content_cache.clear_classifier(canonical_id)
        gc.collect()
    except Exception as exc:
        log(f"[Uninstaller] Note on unloading classifier: {exc}")

    # Step 2: Collect all candidate weight files to remove (from catalog definition and local registry weight_path)
    repo_root = Path(__file__).resolve().parent.parent
    weight_exts = WEIGHT_EXTENSIONS + (".part",)
    candidate_paths: Dict[str, Path] = {}

    if target_model and target_model.get("files"):
        for f_entry in target_model["files"]:
            rel_path = f_entry.get("relative_path", "")
            if not rel_path or Path(rel_path).suffix.lower() not in weight_exts:
                continue
            dest_path = _resolve_safe_repo_path(rel_path)
            if dest_path is not None:
                candidate_paths[rel_path.replace("\\", "/").lower()] = dest_path

    norm_target = model_id.replace("_", "")
    for detail in registry.list_available_details(force_rescan=False):
        cid = detail.get("id", "")
        if cid == model_id or cid == canonical_id or cid.replace("_", "") == norm_target:
            wpath = detail.get("weight_path")
            if wpath:
                wp = Path(wpath).resolve()
                if wp.suffix.lower() in weight_exts:
                    try:
                        rel_w = str(wp.relative_to(repo_root)).replace("\\", "/").lower()
                    except Exception:
                        rel_w = str(wp).replace("\\", "/").lower()
                    candidate_paths[rel_w] = wp

    # Preserve files shared with another currently installed model
    other_installed_files = set()
    for m in models:
        mid = m.get("id")
        if mid not in (model_id, canonical_id) and m.get("installed") is True:
            for f in m.get("files", []):
                rp = f.get("relative_path")
                if rp:
                    other_installed_files.add(rp.replace("\\", "/").lower())

    removed_files = []
    failed_files = []
    for norm_rp, dest_path in candidate_paths.items():
        if norm_rp in other_installed_files:
            log(f"[Uninstaller] Preserving {norm_rp} because it is shared with another installed model.")
            continue
        if not dest_path.exists():
            continue
        for attempt in range(2):
            try:
                dest_path.unlink()
                removed_files.append(norm_rp)
                log(f"[Uninstaller] Deleted weight file {dest_path}")
                break
            except PermissionError as exc:
                if attempt == 0:
                    gc.collect()
                    time.sleep(0.15)
                else:
                    failed_files.append(f"{norm_rp}: {exc}")
                    log(f"[Uninstaller] PermissionError deleting {dest_path}: {exc}")
            except Exception as exc:
                failed_files.append(f"{norm_rp}: {exc}")
                log(f"[Uninstaller] Error deleting {dest_path}: {exc}")
                break

    # Step 3: Force rediscovery and sync catalog from actual disk state
    if canonical_id:
        _remove_installed_model_version(canonical_id)
    if model_id and model_id != canonical_id:
        _remove_installed_model_version(model_id)

    registry.discover_classifiers(force=True)
    available = [c for c in registry.list_available_details(force_rescan=True) if c.get("weights_ready")]

    updated_catalog = _sync_catalog_with_local_folder(catalog)
    installed_ids = updated_catalog.get("installed_models", [])

    resp = {
        "ok": len(failed_files) == 0,
        "type": "uninstallComplete",
        "modelId": model_id,
        "removedFiles": removed_files,
        "installedModels": installed_ids,
        "classifiers": available,
        "timestamp": int(time.time() * 1000),
    }
    if failed_files:
        resp["error"] = f"Failed to delete locked file(s): {'; '.join(failed_files)}"
    if request_id:
        resp["requestId"] = request_id
    log(f"[Uninstaller] Uninstall '{model_id}' finished (removed {len(removed_files)} file(s), failed {len(failed_files)})")
    return resp


def _compare_host_with_release(
    release: Optional[Dict[str, Any]],
    zip_manifest: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Return (update_available, changed_rel_paths) by comparing local host files to GitHub release archive."""
    if not release or not zip_manifest:
        return False, []
    manifest_files = zip_manifest.get("files")
    if not isinstance(manifest_files, dict) or not manifest_files:
        return False, []

    changed: List[str] = []
    for rel_path, remote_hash in manifest_files.items():
        if _is_machine_local_host_path(rel_path):
            continue
        local_path = _resolve_safe_repo_path(rel_path)
        if local_path is None:
            continue
        local_hash = _get_local_file_sha256(local_path, normalize_eol=_is_text_rel_path(rel_path))
        if local_hash != remote_hash:
            changed.append(rel_path)
    return len(changed) > 0, sorted(changed)


def _compare_classifiers_with_release(
    synced_catalog: Dict[str, Any],
    release: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect outdated installed classifiers via release asset SHA-256 digests and newer family models."""
    models = [m for m in (synced_catalog.get("models") or []) if isinstance(m, dict) and m.get("id")]
    installed_ids = set(synced_catalog.get("installed_models") or [])
    assets_by_name = (release or {}).get("assets_by_name") or {}
    release_tag = str((release or {}).get("tag_name") or "")

    available_models = [m for m in models if m.get("status") == "available"]
    outdated: List[Dict[str, Any]] = []
    flagged_installed_ids: Set[str] = set()

    # Case A: Check every installed classifier's release asset files against current GitHub release hashes/sizes
    for m in available_models:
        mid = str(m.get("id"))
        if mid not in installed_ids:
            continue
        changed_files: List[str] = []
        for f_entry in m.get("files") or []:
            rel_path = str(f_entry.get("relative_path") or "")
            if not rel_path:
                continue
            asset, asset_name = _resolve_release_asset_for_file(f_entry, release)
            if not asset:
                continue
            local_path = _resolve_safe_repo_path(rel_path)
            if local_path is None or not local_path.is_file():
                changed_files.append(asset_name)
                continue
            expected_digest = asset.get("digest")
            if expected_digest:
                local_digest = _get_local_file_sha256(local_path, normalize_eol=False)
                if local_digest != expected_digest:
                    changed_files.append(asset_name)
            elif asset.get("size") and local_path.stat().st_size != int(asset["size"]):
                changed_files.append(asset_name)

        if changed_files:
            fam, _ = _parse_model_family_and_rank(m)
            outdated.append({
                "family": fam or mid,
                "modality": m.get("modality") or "image",
                "reason": "release_updated",
                "installedId": mid,
                "installedName": m.get("name") or mid,
                "installedVersion": str(m.get("version") or ""),
                "latestId": mid,
                "latestName": m.get("name") or mid,
                "latestVersion": release_tag or str(m.get("version") or ""),
                "releaseTag": release_tag,
                "downloadSize": str(m.get("download_size") or ""),
                "changedFiles": changed_files,
            })
            flagged_installed_ids.add(mid)

    # Case B: Newer generation exists in an installed model family (e.g. Heimdall 03/04 -> Heimdall 04 B)
    family_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for m in available_models:
        fam, rank = _parse_model_family_and_rank(m)
        if fam is not None and rank >= 0:
            family_groups.setdefault(fam, []).append((rank, m))

    for fam, ranked_models in family_groups.items():
        ranked_models.sort(key=lambda pair: pair[0], reverse=True)
        best_remote_rank, best_remote = ranked_models[0]
        installed_in_fam = [(rank, m) for rank, m in ranked_models if m.get("id") in installed_ids]
        if not installed_in_fam:
            continue
        best_installed_rank, best_installed = installed_in_fam[0]
        best_inst_id = str(best_installed.get("id"))
        if best_remote.get("id") != best_inst_id and best_remote_rank > best_installed_rank:
            outdated.append({
                "family": fam,
                "modality": best_remote.get("modality") or best_installed.get("modality") or "image",
                "reason": "newer_generation",
                "installedId": best_inst_id,
                "installedName": best_installed.get("name") or best_inst_id,
                "installedVersion": str(best_installed.get("version") or ""),
                "latestId": best_remote.get("id"),
                "latestName": best_remote.get("name") or best_remote.get("id"),
                "latestVersion": str(best_remote.get("version") or release_tag or ""),
                "releaseTag": release_tag,
                "downloadSize": str(best_remote.get("download_size") or ""),
            })

    return outdated


def _compute_update_status(force_remote: bool = False, timeout: float = 4.0) -> Dict[str, Any]:
    """Compare local native host and installed classifiers against the current GitHub release."""
    local_catalog = _load_local_catalog_raw()
    remote_catalog, _ = _fetch_remote_catalog(force=force_remote, timeout=timeout)
    release, rel_fetched = _fetch_latest_github_release(force=force_remote, timeout=timeout)
    now_ms = int(time.time() * 1000)

    zip_asset = _find_native_host_zip_asset(release)
    zip_manifest = _get_release_zip_manifest(release, timeout=max(timeout, 8.0)) if (release and zip_asset) else None
    default_dl_url = "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/latest/download/FireVsFire-NativeHost.zip"
    download_url = str((zip_asset or {}).get("browser_download_url") or (release or {}).get("html_url") or default_dl_url)
    release_tag = str((release or {}).get("tag_name") or "")
    release_name = str((release or {}).get("name") or release_tag or "Latest Release")

    effective_catalog = _merge_remote_into_catalog(local_catalog, remote_catalog) if remote_catalog else copy.deepcopy(local_catalog or {"models": []})
    synced = _sync_catalog_with_local_folder(effective_catalog)

    host_update_available, diff_files = _compare_host_with_release(release, zip_manifest)
    outdated = _compare_classifiers_with_release(synced, release) if release else []

    release_notes = str(
        (remote_catalog or {}).get("release_notes")
        or (local_catalog or {}).get("release_notes")
        or (release or {}).get("body")
        or "Includes updates from the latest GitHub release."
    )
    checked = bool(release is not None and (zip_manifest is not None or len(outdated) > 0 or rel_fetched))

    host_info = {
        "currentVersion": HOST_VERSION,
        "latestVersion": release_tag or HOST_VERSION,
        "releaseTag": release_tag,
        "releaseName": release_name,
        "assetDigest": str((zip_asset or {}).get("digest") or ""),
        "assetUpdatedAt": str((zip_asset or {}).get("updated_at") or ""),
        "diffCount": len(diff_files),
        "diffFiles": diff_files[:25],
        "updateAvailable": host_update_available,
        "releaseNotes": release_notes,
        "downloadUrl": download_url,
    }
    classifiers_info = {
        "updateAvailable": len(outdated) > 0,
        "outdated": outdated,
    }
    return {
        "checked": checked,
        "checkedAt": now_ms,
        "host": host_info,
        "classifiers": classifiers_info,
        "anyUpdateAvailable": bool(host_update_available or outdated),
    }


def check_for_host_updates(request_id: Optional[str] = None, force: bool = True) -> Dict[str, Any]:
    status = _compute_update_status(force_remote=force, timeout=5.0)
    host_info = status.get("host") or {}
    resp = {
        "ok": True,
        "type": "checkUpdateResponse",
        "checked": bool(status.get("checked", False)),
        "currentVersion": host_info.get("currentVersion", HOST_VERSION),
        "latestVersion": host_info.get("latestVersion", ""),
        "releaseTag": host_info.get("releaseTag", ""),
        "releaseName": host_info.get("releaseName", ""),
        "diffCount": int(host_info.get("diffCount", 0)),
        "diffFiles": host_info.get("diffFiles", []),
        "updateAvailable": bool(host_info.get("updateAvailable", False)),
        "releaseNotes": host_info.get("releaseNotes", ""),
        "downloadUrl": host_info.get(
            "downloadUrl",
            "https://github.com/LuckyCowMoo/Fire-vs-Fire/releases/latest/download/FireVsFire-NativeHost.zip",
        ),
        "classifierUpdates": status.get("classifiers") or {"updateAvailable": False, "outdated": []},
        "updates": status,
        "timestamp": int(time.time() * 1000),
    }
    if request_id:
        resp["requestId"] = request_id
    return resp


def apply_host_updates(request_id: Optional[str] = None) -> Dict[str, Any]:
    global _catalog_remote_cache, _catalog_remote_cache_time, _release_remote_cache, _release_remote_cache_time, _local_hash_mem_cache
    release, _ = _fetch_latest_github_release(force=True, timeout=6.0)
    zip_asset = _find_native_host_zip_asset(release)
    if not release or not zip_asset or not zip_asset.get("browser_download_url"):
        resp = {
            "ok": False,
            "type": "applyUpdateResponse",
            "updatedFiles": [],
            "errors": ["Could not locate FireVsFire-NativeHost.zip on the latest GitHub release."],
            "timestamp": int(time.time() * 1000),
        }
        if request_id:
            resp["requestId"] = request_id
        return resp

    dl_url = str(zip_asset["browser_download_url"])
    if not _is_allowed_official_url(dl_url):
        resp = {
            "ok": False,
            "type": "applyUpdateResponse",
            "updatedFiles": [],
            "errors": [f"Rejected untrusted release URL: {dl_url}"],
            "timestamp": int(time.time() * 1000),
        }
        if request_id:
            resp["requestId"] = request_id
        return resp

    updated_files: List[str] = []
    errors: List[str] = []
    already_up_to_date = False

    try:
        req = Request(
            dl_url,
            headers={"User-Agent": f"FireVsFire-Updater/{HOST_VERSION}", "Cache-Control": "no-cache"},
        )
        raw_bytes = _urlopen_with_ssl_fallback(req, timeout=30.0)
        expected_digest = zip_asset.get("digest")
        if expected_digest:
            actual_digest = hashlib.sha256(raw_bytes).hexdigest().lower()
            if actual_digest != expected_digest:
                raise RuntimeError(f"Release zip SHA-256 verification failed ({actual_digest[:12]} != {expected_digest[:12]})")

        manifest = _manifest_from_zip_bytes(raw_bytes, zip_asset, release)
        manifest_files = manifest.get("files") or {}

        with zipfile.ZipFile(BytesIO(raw_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel_path = _normalize_repo_rel_path(info.filename)
                if _is_machine_local_host_path(rel_path):
                    continue
                target_path = _resolve_safe_repo_path(rel_path)
                if target_path is None:
                    errors.append(f"Rejected unsafe path: {rel_path}")
                    continue

                entry_bytes = zf.read(info.filename)
                remote_hash = manifest_files.get(rel_path)
                local_hash = _get_local_file_sha256(target_path, normalize_eol=_is_text_rel_path(rel_path))
                if remote_hash and local_hash == remote_hash:
                    continue

                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(entry_bytes)
                    os.replace(tmp_path, target_path)
                    updated_files.append(rel_path)
                    log(f"[Updater] Updated {rel_path} from release {release.get('tag_name')}")
                except Exception as file_exc:
                    errors.append(f"{rel_path}: {file_exc}")
                    log(f"[Updater] Failed updating {rel_path}: {file_exc}")
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

        disk_cache = _read_json_file_safe(_release_cache_path())
        disk_cache["release"] = release
        disk_cache["zip_manifest"] = manifest
        _write_json_file_safe(_release_cache_path(), disk_cache)

        if not updated_files and not errors:
            already_up_to_date = True
    except Exception as exc:
        log(f"[Updater] Failed applying release update: {exc}")
        errors.append(str(exc))

    _catalog_remote_cache = None
    _catalog_remote_cache_time = 0.0
    _release_remote_cache = None
    _release_remote_cache_time = 0.0
    _local_hash_mem_cache = None

    release_tag = str(release.get("tag_name") or "latest")
    catalog = _load_official_catalog(force_remote=False)
    new_version = catalog.get("host_version", HOST_VERSION)

    if already_up_to_date:
        msg = f"Native Host already matches GitHub release {release_tag}."
    else:
        msg = f"Updated {len(updated_files)} native host file(s) from GitHub release {release_tag}."

    resp = {
        "ok": len(errors) == 0 and (len(updated_files) > 0 or already_up_to_date),
        "type": "applyUpdateResponse",
        "updatedFiles": updated_files,
        "alreadyUpToDate": already_up_to_date,
        "releaseTag": release_tag,
        "assetDigest": str(zip_asset.get("digest") or ""),
        "version": new_version,
        "restartRecommended": len(updated_files) > 0,
        "message": msg,
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

            elif mtype == 'hoverItem':
                priority_engine.set_hover_item(msg)
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
                send_message(check_for_host_updates(req_id, force=bool(msg.get('force', True))))
                continue

            elif mtype == 'applyUpdate' or mtype == 'applyNativeHostUpdate':
                req_id = msg.get('requestId')
                send_message(apply_host_updates(req_id))
                continue

            elif mtype == 'listModels':
                send_message(list_models(
                    msg.get('requestId'),
                    check_updates=bool(msg.get('checkUpdates', False)),
                    force_update_check=bool(msg.get('forceUpdateCheck', False)),
                ))
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

            elif mtype == 'uninstallOfficialModel':
                model_id = msg.get('modelId')
                req_id = msg.get('requestId')
                threading.Thread(
                    target=lambda: send_message(uninstall_official_model(model_id, req_id)),
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
