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
        RuntimeTextEncoder,
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
    RuntimeTextEncoder = _ext_mod.RuntimeTextEncoder


class FvfMimirClassifier(BaseClassifier):
    """FvF Mímir V2 AI Text Classifier."""

    CLASSIFIER_ID: str = "fvf_mimir_02"
    REQUIRED_PACKAGES: List[str] = ["transformers", "onnxruntime", "lightgbm", "scikit-learn", "joblib"]

    FEATURE_COLS: Tuple[str, ...] = (
        # Channel 1: Semantic & Discourse Forensic Encoder (2 features)
        "enc_logit", "enc_prob",
        # Channel 2A: LM Perplexity & Curvature (17 features)
        "lm_ppl_global", "lm_mean_nll", "lm_entropy_mean", "lm_ppl_norm",
        "lm_logprob_mean", "lm_logprob_std", "lm_logprob_skew", "lm_logprob_kurtosis",
        "lm_top1_frac", "lm_top5_frac", "lm_top10_frac", "lm_top100_frac",
        "lm_tail_frac", "lm_sent_ppl_mean", "lm_sent_ppl_std", "lm_sent_ppl_range_ratio",
        "lm_curvature_discrepancy",
        # Channel 2B: Length-Normalized Mindner Stylometrics (19 features, no raw length/sentence count)
        "stylo_words_per_sentence_mean", "stylo_ttr", "stylo_yules_k", "stylo_simpson_d",
        "stylo_flesch_reading_ease", "stylo_flesch_kincaid_grade", "stylo_ari",
        "stylo_syllables_per_word", "stylo_determiner_density", "stylo_preposition_density",
        "stylo_conjunction_density", "stylo_pronoun_3rd_density", "stylo_all_pronoun_density",
        "stylo_verb_density", "stylo_adverb_density", "stylo_adjective_density",
        "stylo_comma_density", "stylo_punct_density", "stylo_pacing_std",
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
        self._text_encoder: Optional[RuntimeTextEncoder] = None
        self._lm_scorer: Optional[RuntimeLMScorer] = None
        self._head_bundle: Optional[Any] = None
        self._classifier_head: Optional[Any] = None
        self._active_features: Tuple[str, ...] = self.FEATURE_COLS
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

        # Channel 1: Semantic & Discourse Forensic Encoder (DistilRoBERTa-AFPL INT8)
        self._text_encoder = RuntimeTextEncoder(device=device)
        if not self._text_encoder.probe_hardware_and_load():
            self._log_to_stderr("[FvfMimirClassifier] Notice: Text encoder unavailable; using Channel 2 fallback.")
        else:
            enc_hw = self._text_encoder.get_hardware_info()
            self._device_info = enc_hw

        # Channel 2: Causal LM Perplexity & Curvature Scorer
        self._lm_scorer = RuntimeLMScorer(device=device)
        if not self._lm_scorer.probe_hardware_and_load():
            self._log_to_stderr("[FvfMimirClassifier] Notice: LM scorer using fallback.")
        elif not self._text_encoder or self._device_info.get("backend") == "CPU":
            self._device_info = self._lm_scorer.get_hardware_info()

        # Load calibrated fusion head bundle
        if self.model_path and Path(self.model_path).exists():
            try:
                import joblib
                self._head_bundle = joblib.load(self.model_path)
                self._classifier_head = self._head_bundle.get("model") or self._head_bundle.get("classifier")
                bundle_feats = self._head_bundle.get("features")
                if bundle_feats:
                    self._active_features = tuple(bundle_feats)
                self._log_to_stderr(
                    f"[FvfMimirClassifier] Loaded Dual-Channel head v{self._head_bundle.get('version', '2.2')} "
                    f"({len(self._active_features)} features) from {self.model_path}"
                )
            except Exception as e:
                self._log_to_stderr(f"[FvfMimirClassifier] Failed loading head from {self.model_path}: {e}")
                self._classifier_head = None
        else:
            self._log_to_stderr(f"[FvfMimirClassifier] Head bundle not found at {self.model_path}; baseline fallback active.")

        self._is_loaded = True
        return True, None

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "text":
            return {"features": np.empty((0, len(self._active_features)))}, []

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

            # Channel 1: Semantic & Discourse Forensic Encoder
            enc_feats = self._text_encoder.score(sanitized) if self._text_encoder else {"enc_logit": 0.0, "enc_prob": 0.5}
            # Channel 2A: LM Perplexity & Curvature
            lm_feats = self._lm_scorer.score(sanitized) if self._lm_scorer else {}
            # Channel 2B: Length-Normalized Mindner Stylometrics
            sty_feats = extract_stylometric_features(sanitized)

            row_dict = {**enc_feats, **lm_feats, **sty_feats}
            feature_vector = [
                float(row_dict[col]) if (col in row_dict and row_dict[col] is not None) else np.nan
                for col in self._active_features
            ]
            feature_rows.append(feature_vector)

        if not feature_rows:
            return {"features": np.empty((0, len(self._active_features)))}, []

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
        col_to_idx = {c: i for i, c in enumerate(self._active_features)}
        for row in matrix:
            if "enc_prob" in col_to_idx:
                scores.append(float(row[col_to_idx["enc_prob"]]))
            else:
                ppl_norm = row[col_to_idx["lm_ppl_norm"]] if "lm_ppl_norm" in col_to_idx else 1.0
                top1 = row[col_to_idx["lm_top1_frac"]] if "lm_top1_frac" in col_to_idx else 0.35
                tail = row[col_to_idx["lm_tail_frac"]] if "lm_tail_frac" in col_to_idx else 0.15

                logit = 1.2 * (0.86 - ppl_norm) + 1.5 * (top1 - 0.38) - 1.2 * (tail - 0.10)
                score = float(1.0 / (1.0 + math.exp(-max(min(logit * 2.5, 20.0), -20.0))))
                scores.append(score)

        return scores

    def unload_model(self) -> None:
        self._text_encoder = None
        self._lm_scorer = None
        self._classifier_head = None
        self._is_loaded = False
