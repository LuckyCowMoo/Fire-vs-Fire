"""FvF Mímir V2 AI Text Classifier.

Official text classification model for Fire vs Fire.
Combines causal LM perplexity curvature (Binoculars/Fast-DetectGPT) and
Mindner et al. (2023) stylometrics with an anti-false-positive calibrated head.

Universal Hardware Support:
- NVIDIA RTX (CUDA)
- AMD Radeon RX (DirectML / ROCm)
- Intel ARC / Core Ultra (DirectML / OpenVINO)
- Apple Silicon M1-M4 (Metal / MPS / CoreML)
- Qualcomm Snapdragon X Elite (Hexagon NPU / DirectML / QNN)
- Universal CPU Fallback
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .base_classifier import BaseClassifier
    from .mimir_extractor import (
        sanitize_text,
        extract_stylometric_features,
        RuntimeLMScorer,
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
    sanitize_text = _ext_mod.sanitize_text
    extract_stylometric_features = _ext_mod.extract_stylometric_features
    RuntimeLMScorer = _ext_mod.RuntimeLMScorer


class FvfMimirClassifier(BaseClassifier):
    """FvF Mímir V2 AI Text Classifier."""

    CLASSIFIER_ID: str = "fvf_mimir_02"
    REQUIRED_PACKAGES: List[str] = ["lightgbm", "joblib"]

    FEATURE_COLS: Tuple[str, ...] = (
        "lm_ppl_global", "lm_mean_nll", "lm_entropy_mean", "lm_ppl_norm",
        "lm_logprob_mean", "lm_logprob_std", "lm_logprob_skew", "lm_logprob_kurtosis",
        "lm_top1_frac", "lm_top5_frac", "lm_top10_frac", "lm_top100_frac",
        "lm_tail_frac", "lm_sent_ppl_mean", "lm_sent_ppl_std", "lm_sent_ppl_range_ratio",
        "lm_curvature_discrepancy", "stylo_word_count", "stylo_sentence_count",
        "stylo_words_per_sentence_mean", "stylo_words_per_sentence_std",
        "stylo_sentence_length_cv", "stylo_ttr", "stylo_yules_k", "stylo_simpson_d",
        "stylo_flesch_reading_ease", "stylo_flesch_kincaid_grade", "stylo_ari",
        "stylo_syllables_per_word", "stylo_determiner_density", "stylo_preposition_density",
        "stylo_conjunction_density", "stylo_pronoun_1st_density", "stylo_pronoun_2nd_density",
        "stylo_pronoun_3rd_density", "stylo_all_pronoun_density", "stylo_verb_density",
        "stylo_adverb_density", "stylo_adjective_density", "stylo_comma_density",
        "stylo_punct_density", "stylo_pacing_std",
    )

    def __init__(self, model_path: Optional[Path] = None):
        classifiers_dir = Path(__file__).resolve().parent
        candidate_paths = [
            classifiers_dir / "fvf_mimir_head.joblib",
            classifiers_dir.parent.parent / "models" / "fvf_mimir_head.joblib",
        ]
        resolved_path = None
        for p in candidate_paths:
            if p.is_file():
                resolved_path = p
                break
        super().__init__(model_path=model_path or resolved_path or candidate_paths[0])
        self._lm_scorer: Optional[RuntimeLMScorer] = None
        self._head_bundle: Optional[Any] = None
        self._classifier_head: Optional[Any] = None
        self._device_info: Dict[str, Any] = {"device": "cpu", "name": "CPU", "backend": "CPU"}

    def get_supported_modalities(self) -> Set[str]:
        return {"text"}

    def get_model_name(self) -> str:
        return "FvF Mímir V2"

    def get_label(self) -> str:
        return "AI text"

    def get_device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def load_model(self) -> Tuple[bool, Optional[str]]:
        # Universal hardware probe: CUDA -> Apple MPS -> DirectML -> CPU
        device = self._try_cuda_then_directml()
        self._log_to_stderr(f"[FvfMimirClassifier] Probing hardware accelerators (Active: {self._device_info.get('name', 'CPU')})...")

        # Initialize Universal LM Scorer
        self._lm_scorer = RuntimeLMScorer(device=device)
        if not self._lm_scorer.probe_hardware_and_load():
            self._log_to_stderr("[FvfMimirClassifier] Notice: LM scorer using fallback.")
        else:
            hw = self._lm_scorer.get_hardware_info()
            self._device_info = hw

        # Load calibrated classifier head bundle
        if self.model_path and Path(self.model_path).exists():
            try:
                import joblib
                self._head_bundle = joblib.load(self.model_path)
                self._classifier_head = self._head_bundle.get("classifier")
                self._log_to_stderr(f"[FvfMimirClassifier] Loaded calibrated classifier head from {self.model_path}")
            except Exception as e:
                self._log_to_stderr(f"[FvfMimirClassifier] Failed loading head from {self.model_path}: {e}")
                self._classifier_head = None
        else:
            self._log_to_stderr(f"[FvfMimirClassifier] Head bundle not found at {self.model_path}; baseline fallback active.")

        self._is_loaded = True
        return True, None

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "text":
            return {"features": np.empty((0, len(self.FEATURE_COLS)))}, []

        valid_indices: List[int] = []
        feature_rows: List[List[float]] = []

        for idx, item in enumerate(inputs):
            if not isinstance(item, str):
                continue
            sanitized = sanitize_text(item)
            # Browser extension minimum text length is 150 chars
            if len(sanitized) < 150:
                continue

            valid_indices.append(idx)

            # Stream 1: LM Perplexity & Curvature
            lm_feats = self._lm_scorer.score(sanitized) if self._lm_scorer else {}
            # Stream 2: Mindner Stylometrics
            sty_feats = extract_stylometric_features(sanitized)

            row_dict = {**lm_feats, **sty_feats}
            feature_vector = [
                float(row_dict[col]) if (col in row_dict and row_dict[col] is not None) else np.nan
                for col in self.FEATURE_COLS
            ]
            feature_rows.append(feature_vector)

        if not feature_rows:
            return {"features": np.empty((0, len(self.FEATURE_COLS)))}, []

        matrix = np.array(feature_rows, dtype=np.float32)
        return {"features": matrix}, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        matrix = batch_tensor.get("features")
        if matrix is None or len(matrix) == 0:
            return []

        # If trained head is available, run calibrated probabilistic prediction
        if self._classifier_head is not None:
            try:
                probs = self._classifier_head.predict_proba(matrix)[:, 1]
                return [float(p) for p in probs]
            except Exception as exc:
                self._log_to_stderr(f"[FvfMimirClassifier] Inference error: {exc}")

        # Baseline heuristic fallback if head weights are not yet generated
        scores: List[float] = []
        col_to_idx = {c: i for i, c in enumerate(self.FEATURE_COLS)}
        for row in matrix:
            ppl_norm = row[col_to_idx.get("lm_ppl_norm", 0)]
            burstiness = row[col_to_idx.get("lm_sent_ppl_std", 0)]
            cv_len = row[col_to_idx.get("stylo_sentence_length_cv", 0)]
            top1 = row[col_to_idx.get("lm_top1_frac", 0)]

            logit = 1.5 * (0.88 - ppl_norm) + 0.05 * (12.0 - burstiness) + 1.2 * (0.40 - cv_len) + 1.5 * (top1 - 0.90)
            score = float(1.0 / (1.0 + math.exp(-max(min(logit * 3.0, 20.0), -20.0))))
            scores.append(score)

        return scores

    def unload_model(self) -> None:
        self._lm_scorer = None
        self._classifier_head = None
        self._is_loaded = False
