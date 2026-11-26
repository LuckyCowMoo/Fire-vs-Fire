"""FvF Mímir 03 B AI Text Classifier.

Unified 3-Branch Neural AI Text Detector with a Single End-to-End Classification Head:
  - Branch 1: ELECTRA-base Disentangled Syntax/Discourse Backbone (masked mean+max -> 256-d)
  - Branch 2: 22 Vocabulary-Rarity-Residualized Causal LM Curvature Features (distilgpt2 ONNX INT8)
  - Branch 3: 32 Length- & Bias-Normalized Mindner & Discourse Stylometrics
  - Single Unified Gated Head: 384 -> 128 -> 1 trained with AFPL ("B") + GroupDRO
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .base_classifier import BaseClassifier
    from .mimir_extractor import (
        MIMIR03_TAB_COLS,
        RuntimeLMScorer,
        create_ort_session_with_warmup,
        extract_mimir03_tab_vector,
        provider_to_hardware_info,
        sanitize_text,
    )
except Exception:
    import importlib.util

    _parent = Path(__file__).resolve().parent
    _base_spec = importlib.util.spec_from_file_location("base_classifier", str(_parent / "base_classifier.py"))
    _base_mod = importlib.util.module_from_spec(_base_spec)
    _base_spec.loader.exec_module(_base_mod)
    BaseClassifier = _base_mod.BaseClassifier

    _ext_spec = importlib.util.spec_from_file_location("mimir_extractor", str(_parent / "mimir_extractor.py"))
    _ext_mod = importlib.util.module_from_spec(_ext_spec)
    _ext_spec.loader.exec_module(_ext_mod)
    MIMIR03_TAB_COLS = _ext_mod.MIMIR03_TAB_COLS
    RuntimeLMScorer = _ext_mod.RuntimeLMScorer
    create_ort_session_with_warmup = _ext_mod.create_ort_session_with_warmup
    extract_mimir03_tab_vector = _ext_mod.extract_mimir03_tab_vector
    provider_to_hardware_info = _ext_mod.provider_to_hardware_info
    sanitize_text = _ext_mod.sanitize_text


class FvfMimir03BClassifier(BaseClassifier):
    """FvF Mímir 03 B Unified Multi-Branch AI Text Classifier (INT8 ONNX)."""

    CLASSIFIER_ID: str = "fvf_mimir_03_b"
    REQUIRED_PACKAGES: List[str] = ["transformers", "onnxruntime", "numpy"]

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path=model_path)
        resolved = self._find_checkpoint()
        self.model_path = Path(model_path) if model_path is not None else resolved
        self._tokenizer = None
        self._ort_session = None
        self._lm_scorer: Optional[RuntimeLMScorer] = None
        self._meta: Dict[str, Any] = {}
        self._max_length: int = 384
        self._temperature: float = 1.0
        self._logit_bias: float = 0.0
        self._tab_mean: np.ndarray = np.zeros(len(MIMIR03_TAB_COLS), dtype=np.float32)
        self._tab_std: np.ndarray = np.ones(len(MIMIR03_TAB_COLS), dtype=np.float32)
        self._device_info: Dict[str, Any] = {"device": "cpu", "name": "CPU", "backend": "CPU"}

    def _find_checkpoint(self) -> Optional[Path]:
        classifiers_dir = Path(__file__).resolve().parent
        onnx_path = Path(self.model_path) if self.model_path is not None else (classifiers_dir / "fvf_mimir_03_b_int8.onnx")
        meta_path = onnx_path.with_name("fvf_mimir_03_b_meta.json")
        if onnx_path.is_file() and meta_path.is_file():
            return onnx_path
        alt_meta = classifiers_dir / "fvf_mimir_03_b_meta.json"
        if onnx_path.is_file() and alt_meta.is_file():
            return onnx_path
        return None

    def get_supported_modalities(self) -> Set[str]:
        return {"text"}

    def get_model_name(self) -> str:
        return "FvF Mímir 03 B"

    def get_label(self) -> str:
        return "AI text"

    def get_suggested_threshold(self) -> Optional[float]:
        return 0.50

    def get_device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def get_estimated_memory_mb(self) -> float:
        return 380.0

    def load_model(self) -> Tuple[bool, Optional[str]]:
        ckpt_path = self._find_checkpoint()
        if ckpt_path is None:
            return False, "Missing fvf_mimir_03_b_int8.onnx or fvf_mimir_03_b_meta.json in native/classifiers/"

        classifiers_dir = Path(__file__).resolve().parent
        meta_path = ckpt_path.with_name("fvf_mimir_03_b_meta.json")
        if not meta_path.is_file():
            meta_path = classifiers_dir / "fvf_mimir_03_b_meta.json"

        try:
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._max_length = int(self._meta.get("max_length", 384))
            self._temperature = float(self._meta.get("temperature", 1.0))
            self._logit_bias = float(self._meta.get("logit_bias", 0.0))
            mean_list = self._meta.get("tab_mean")
            std_list = self._meta.get("tab_std")
            if isinstance(mean_list, list) and len(mean_list) == len(MIMIR03_TAB_COLS):
                self._tab_mean = np.asarray(mean_list, dtype=np.float32)
            if isinstance(std_list, list) and len(std_list) == len(MIMIR03_TAB_COLS):
                self._tab_std = np.maximum(np.asarray(std_list, dtype=np.float32), 1e-4)
        except Exception as exc:
            return False, f"Failed reading Mímir 03 B metadata: {exc}"

        try:
            from transformers import AutoTokenizer

            warmup_feed = {
                "input_ids": np.ones((1, 8), dtype=np.int64),
                "attention_mask": np.ones((1, 8), dtype=np.int64),
                "tab_features": np.zeros((1, len(MIMIR03_TAB_COLS)), dtype=np.float32),
            }
            self._ort_session, active_prov = create_ort_session_with_warmup(
                ckpt_path,
                warmup_feed,
                log_prefix="[FvfMimir03B]",
            )
            dev_code, dev_name, dev_backend = provider_to_hardware_info(active_prov)
            self._device_info = {"device": dev_code, "name": dev_name, "backend": dev_backend}

            tok_dir = classifiers_dir / "mimir_03_tokenizer"
            backbone_id = str(self._meta.get("backbone", "google/electra-base-discriminator"))
            if tok_dir.is_dir() and (tok_dir / "tokenizer.json").is_file():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(backbone_id)

            self._lm_scorer = RuntimeLMScorer(
                onnx_model_path=classifiers_dir / "fvf_mimir_backbone_int8.onnx",
                max_length=512,
            )
            if not self._lm_scorer.probe_hardware_and_load():
                self._log_to_stderr("[FvfMimir03B] Notice: Causal LM scorer unavailable; zeroing LM channels.")
                self._lm_scorer = None

            self.model_path = ckpt_path
            self._is_loaded = True
            self._log_to_stderr(
                f"[FvfMimir03B] Loaded unified Weight-Only INT8 graph on {self._device_info['name']} "
                f"(T={self._temperature:.4f}, features={len(MIMIR03_TAB_COLS)})"
            )
            return True, None
        except Exception as exc:
            self._is_loaded = False
            return False, f"Failed loading FvF Mímir 03 B: {exc}"

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "text" or not self._is_loaded or self._tokenizer is None:
            return None, []

        valid_indices: List[int] = []
        valid_texts: List[str] = []
        tab_rows: List[np.ndarray] = []

        for idx, item in enumerate(inputs):
            if not isinstance(item, str):
                continue
            sanitized = sanitize_text(item)
            if len(sanitized) < 150:
                continue

            valid_indices.append(idx)
            valid_texts.append(sanitized)
            raw_vec = extract_mimir03_tab_vector(sanitized, lm_scorer=self._lm_scorer)
            norm_vec = np.clip((raw_vec - self._tab_mean) / self._tab_std, -10.0, 10.0).astype(np.float32)
            tab_rows.append(norm_vec)

        if not valid_texts:
            return None, []

        tok = self._tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        tab_mat = np.stack(tab_rows, axis=0).astype(np.float32)

        long_positions = [i for i, t in enumerate(valid_texts) if len(t) > 1600]
        w2_payload = None
        if long_positions:
            w2_texts = [valid_texts[i][850:2500].strip() for i in long_positions]
            keep = [(pos, txt) for pos, txt in zip(long_positions, w2_texts) if len(txt) >= 120]
            if keep:
                k_pos = [p for p, _ in keep]
                k_txt = [t for _, t in keep]
                tok2 = self._tokenizer(
                    k_txt,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="np",
                )
                w2_payload = {
                    "positions": k_pos,
                    "input_ids": tok2["input_ids"].astype(np.int64),
                    "attention_mask": tok2["attention_mask"].astype(np.int64),
                    "tab_features": tab_mat[k_pos],
                }

        return {
            "input_ids": tok["input_ids"].astype(np.int64),
            "attention_mask": tok["attention_mask"].astype(np.int64),
            "tab_features": tab_mat,
            "w2": w2_payload,
        }, valid_indices

    def _run_ort(self, feeds: Dict[str, Any]) -> np.ndarray:
        try:
            return np.asarray(self._ort_session.run(["logits"], feeds)[0])
        except Exception as run_err:
            if self._ort_session.get_providers()[0] == "CPUExecutionProvider":
                raise
            self._log_to_stderr(f"[FvfMimir03B] Runtime EP failed ({run_err}); switching to CPUExecutionProvider")
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self._ort_session = ort.InferenceSession(
                str(self.model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
            )
            dev_code, dev_name, dev_backend = provider_to_hardware_info("CPUExecutionProvider")
            self._device_info = {"device": dev_code, "name": dev_name, "backend": dev_backend}
            return np.asarray(self._ort_session.run(["logits"], feeds)[0])

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not batch_tensor or self._ort_session is None:
            return []

        logits = self._run_ort(
            {
                "input_ids": batch_tensor["input_ids"],
                "attention_mask": batch_tensor["attention_mask"],
                "tab_features": batch_tensor["tab_features"],
            }
        ).reshape(-1).astype(np.float64)

        w2 = batch_tensor.get("w2")
        if w2 is not None:
            logits2 = self._run_ort(
                {
                    "input_ids": w2["input_ids"],
                    "attention_mask": w2["attention_mask"],
                    "tab_features": w2["tab_features"],
                }
            ).reshape(-1).astype(np.float64)
            for pos, val in zip(w2["positions"], logits2):
                logits[pos] = 0.5 * (logits[pos] + val)

        scaled = np.clip((logits + self._logit_bias) / max(self._temperature, 1e-4), -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        return [float(p) for p in probs]

    def unload_model(self) -> None:
        self._ort_session = None
        self._tokenizer = None
        self._lm_scorer = None
        self._is_loaded = False
        gc.collect()
