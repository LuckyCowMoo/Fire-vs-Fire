"""Mímir Runtime Feature Extractor for Native Host.

Self-contained module providing text sanitization, fast stylometrics,
and causal LM perplexity & curvature scoring for live browser text.
Supports universal hardware acceleration across:
- NVIDIA RTX (CUDA)
- AMD Radeon RX (DirectML / ROCm)
- Intel ARC / Core Ultra (DirectML / OpenVINO)
- Apple Silicon M1-M4 (Metal / MPS / CoreML)
- Qualcomm Snapdragon X Elite (Hexagon NPU / DirectML / QNN)
- Universal CPU Fallback
"""

from __future__ import annotations
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


# Detokenizer normalization
_RE_PUNCT_SPACING = re.compile(r"\s+([,.;:!?%])")
_RE_NOT_CONTRACTION = re.compile(r"\b([A-Za-z]+)\s+n't\b", re.IGNORECASE)
_RE_APOSTROPHE_CONTRACTION = re.compile(r"\b([A-Za-z]+)\s+'(s|re|ve|d|ll|m)\b", re.IGNORECASE)
_RE_HYPHEN_SPACING = re.compile(r"\b([A-Za-z0-9]+)\s+-\s+([A-Za-z0-9]+)\b")
_RE_PAREN_OPEN = re.compile(r"\(\s+")
_RE_PAREN_CLOSE = re.compile(r"\s+\)")
_RE_BRACKET_OPEN = re.compile(r"\[\s+")
_RE_BRACKET_CLOSE = re.compile(r"\s+\]")
_RE_DOUBLE_QUOTES = re.compile(r'"\s*(.*?)\s*"')
_RE_SLASH_SPACING = re.compile(r"\s*/\s*")
_RE_DOLLAR_NUM = re.compile(r"\$\s+(\d+)")
_RE_HASH_NUM = re.compile(r"#\s+(\d+)")
_RE_MULTI_SPACE = re.compile(r"[ \t\u00A0]+")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_URL_TAG = re.compile(r"\[\s*URL_\d+\s*\]|\(URL_\d+\)|URL_\d+", re.IGNORECASE)

# Stylometrics
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PUNCT_RE = re.compile(r"[,;:\-\(\)\[\]\{\}\"']")
_VOWELS = set("aeiouy")

_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those", "each", "every", "some", "any", "no"})
_PREPOSITIONS = frozenset({
    "in", "on", "at", "by", "for", "to", "from", "of", "with", "about",
    "into", "through", "between", "during", "before", "after", "over",
    "under", "around", "without", "within", "against", "upon", "towards",
})
_CONJUNCTIONS = frozenset({
    "and", "or", "but", "so", "yet", "because", "although", "while",
    "if", "when", "since", "unless", "whereas", "though", "even",
})
_PRONOUNS_1ST = frozenset({"i", "me", "my", "mine", "we", "us", "our", "ours"})
_PRONOUNS_2ND = frozenset({"you", "your", "yours"})
_PRONOUNS_3RD = frozenset({"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"})

_COMMON_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "make", "made", "use", "used", "show", "shows",
    "showed", "say", "says", "said", "go", "goes", "went", "get", "got",
    "think", "thinks", "thought", "see", "saw", "know", "knew", "take", "took",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
})

_ADJECTIVE_SUFFIXES = ("ous", "ful", "able", "ible", "ive", "al", "ish", "ic", "ent", "ant")


def sanitize_text(text: str) -> str:
    """Normalize raw text and remove detokenizer artifacts."""
    if not text:
        return ""
    t = _RE_URL_TAG.sub("", text)
    t = _RE_NOT_CONTRACTION.sub(r"\1n't", t)
    t = _RE_APOSTROPHE_CONTRACTION.sub(r"\1'\2", t)
    t = _RE_HYPHEN_SPACING.sub(r"\1-\2", t)
    t = _RE_PUNCT_SPACING.sub(r"\1", t)
    t = _RE_SLASH_SPACING.sub(r"/", t)
    t = _RE_PAREN_OPEN.sub("(", t)
    t = _RE_PAREN_CLOSE.sub(")", t)
    t = _RE_BRACKET_OPEN.sub("[", t)
    t = _RE_BRACKET_CLOSE.sub("]", t)
    t = _RE_DOLLAR_NUM.sub(r"$\1", t)
    t = _RE_HASH_NUM.sub(r"#\1", t)
    t = _RE_DOUBLE_QUOTES.sub(r'"\1"', t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    t = _RE_MULTI_NEWLINE.sub("\n\n", t)
    return t.strip()


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0.0 else default


def count_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    if len(token) <= 3:
        return 1
    syllables = 0
    prev_vowel = False
    for char in token:
        is_v = char in _VOWELS
        if is_v and not prev_vowel:
            syllables += 1
        prev_vowel = is_v
    if token.endswith("e") and syllables > 1 and not token.endswith(("le", "ye")):
        syllables -= 1
    if token.endswith("ed") and syllables > 1 and not token.endswith(("ted", "ded")):
        syllables -= 1
    return max(syllables, 1)


def compute_yules_k_and_simpson(words: List[str]) -> Tuple[float, float]:
    n = len(words)
    if n <= 1:
        return 0.0, 0.0
    counts = Counter(words)
    freq_of_freqs = Counter(counts.values())
    s2 = sum((i ** 2) * vi for i, vi in freq_of_freqs.items())
    yules_k = 10000.0 * (s2 - n) / (n ** 2) if n > 0 else 0.0
    simpson_num = sum(c * (c - 1) for c in counts.values())
    simpson_d = simpson_num / (n * (n - 1))
    return float(max(0.0, yules_k)), float(max(0.0, min(1.0, simpson_d)))


def compute_readability(words: List[str], sentences: List[str], syllables: int) -> Tuple[float, float, float]:
    n_words = len(words)
    n_sents = max(len(sentences), 1)
    n_chars = sum(len(w) for w in words)
    if n_words == 0:
        return 0.0, 0.0, 0.0
    asl = n_words / n_sents
    asw = syllables / n_words
    flesch_re = 206.835 - (1.015 * asl) - (84.6 * asw)
    fk_grade = (0.39 * asl) + (11.8 * asw) - 15.59
    ari = 4.71 * (n_chars / n_words) + 0.5 * asl - 21.43
    return float(flesch_re), float(fk_grade), float(ari)


def extract_stylometric_features(text: str) -> Dict[str, float]:
    raw = text.strip()
    if not raw:
        return {
            "stylo_word_count": 0.0, "stylo_sentence_count": 0.0,
            "stylo_words_per_sentence_mean": 0.0, "stylo_words_per_sentence_std": 0.0,
            "stylo_sentence_length_cv": 0.0, "stylo_ttr": 0.0,
            "stylo_yules_k": 0.0, "stylo_simpson_d": 0.0,
            "stylo_flesch_reading_ease": 0.0, "stylo_flesch_kincaid_grade": 0.0,
            "stylo_ari": 0.0, "stylo_syllables_per_word": 0.0,
            "stylo_determiner_density": 0.0, "stylo_preposition_density": 0.0,
            "stylo_conjunction_density": 0.0, "stylo_pronoun_1st_density": 0.0,
            "stylo_pronoun_2nd_density": 0.0, "stylo_pronoun_3rd_density": 0.0,
            "stylo_all_pronoun_density": 0.0, "stylo_verb_density": 0.0,
            "stylo_adverb_density": 0.0, "stylo_adjective_density": 0.0,
            "stylo_comma_density": 0.0, "stylo_punct_density": 0.0,
            "stylo_pacing_std": 0.0,
        }
    sentences = [s.strip() for s in _SENTENCE_RE.split(raw) if s.strip()] or [raw]
    words = [m.group(0).lower() for m in _WORD_RE.finditer(raw)]
    n_words = len(words)
    n_sents = len(sentences)
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    w_mean = mean(sent_word_counts) if sent_word_counts else 0.0
    w_std = pstdev(sent_word_counts) if len(sent_word_counts) > 1 else 0.0
    w_cv = _safe_div(w_std, w_mean)
    syllables = sum(count_syllables(w) for w in words)
    syllables_per_word = _safe_div(syllables, n_words)
    ttr = _safe_div(len(set(words)), n_words)
    yules_k, simpson_d = compute_yules_k_and_simpson(words)
    flesch_re, fk_grade, ari = compute_readability(words, sentences, syllables)
    det_count = sum(1 for w in words if w in _DETERMINERS)
    prep_count = sum(1 for w in words if w in _PREPOSITIONS)
    conj_count = sum(1 for w in words if w in _CONJUNCTIONS)
    pron1_count = sum(1 for w in words if w in _PRONOUNS_1ST)
    pron2_count = sum(1 for w in words if w in _PRONOUNS_2ND)
    pron3_count = sum(1 for w in words if w in _PRONOUNS_3RD)
    pron_count = pron1_count + pron2_count + pron3_count
    verb_count = sum(1 for w in words if w in _COMMON_VERBS or w.endswith(("ed", "ing")))
    adv_count = sum(1 for w in words if w.endswith("ly"))
    adj_count = sum(1 for w in words if any(w.endswith(sfx) for sfx in _ADJECTIVE_SUFFIXES))
    comma_count = raw.count(",")
    punct_count = len(_PUNCT_RE.findall(raw))
    punct_splits = _PUNCT_RE.split(raw)
    seg_lengths = [len(_WORD_RE.findall(seg)) for seg in punct_splits if seg.strip()]
    pacing_std = pstdev(seg_lengths) if len(seg_lengths) > 1 else 0.0

    return {
        "stylo_word_count": float(n_words),
        "stylo_sentence_count": float(n_sents),
        "stylo_words_per_sentence_mean": float(w_mean),
        "stylo_words_per_sentence_std": float(w_std),
        "stylo_sentence_length_cv": float(w_cv),
        "stylo_ttr": float(ttr),
        "stylo_yules_k": float(yules_k),
        "stylo_simpson_d": float(simpson_d),
        "stylo_flesch_reading_ease": float(flesch_re),
        "stylo_flesch_kincaid_grade": float(fk_grade),
        "stylo_ari": float(ari),
        "stylo_syllables_per_word": float(syllables_per_word),
        "stylo_determiner_density": float(_safe_div(det_count, n_words)),
        "stylo_preposition_density": float(_safe_div(prep_count, n_words)),
        "stylo_conjunction_density": float(_safe_div(conj_count, n_words)),
        "stylo_pronoun_1st_density": float(_safe_div(pron1_count, n_words)),
        "stylo_pronoun_2nd_density": float(_safe_div(pron2_count, n_words)),
        "stylo_pronoun_3rd_density": float(_safe_div(pron3_count, n_words)),
        "stylo_all_pronoun_density": float(_safe_div(pron_count, n_words)),
        "stylo_verb_density": float(_safe_div(verb_count, n_words)),
        "stylo_adverb_density": float(_safe_div(adv_count, n_words)),
        "stylo_adjective_density": float(_safe_div(adj_count, n_words)),
        "stylo_comma_density": float(_safe_div(comma_count, n_words)),
        "stylo_punct_density": float(_safe_div(punct_count, n_words)),
        "stylo_pacing_std": float(pacing_std),
    }


class RuntimeLMScorer:
    """Universal Hardware-Accelerated Causal LM Scorer.
    
    Supports:
    - NVIDIA RTX (CUDA)
    - AMD Radeon RX (DirectML / ROCm)
    - Intel ARC / Core Ultra (DirectML / OpenVINO)
    - Apple Silicon M1-M4 (MPS / CoreML)
    - Qualcomm Snapdragon X Elite (Hexagon NPU / DirectML / QNN)
    - CPU Fallback (Dynamic INT8 / FP32)
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        onnx_model_path: Optional[Path] = None,
        device: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path or (
            Path(__file__).resolve().parent / "fvf_mimir_backbone_int8.onnx"
        )
        self.max_length = max_length
        self._requested_device = device
        self._tokenizer = None
        self._torch_model = None
        self._ort_session = None
        self._backend = "CPU"
        self._device_name = "CPU"
        self._torch = None
        self._device = None

    def probe_hardware_and_load(self) -> bool:
        """Probe system hardware and initialize the fastest available backend."""
        # 1. First, check if ONNX Runtime with hardware execution provider is viable
        if self.onnx_model_path and self.onnx_model_path.exists():
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                # Preferred provider priority: DirectML (AMD/Intel/Qualcomm) -> CUDA (NVIDIA) -> CoreML (Apple) -> CPU
                providers_to_try = []
                if "DmlExecutionProvider" in available_providers:
                    providers_to_try.append("DmlExecutionProvider")
                if "CUDAExecutionProvider" in available_providers:
                    providers_to_try.append("CUDAExecutionProvider")
                if "CoreMLExecutionProvider" in available_providers:
                    providers_to_try.append("CoreMLExecutionProvider")
                if "OpenVINOExecutionProvider" in available_providers:
                    providers_to_try.append("OpenVINOExecutionProvider")
                providers_to_try.append("CPUExecutionProvider")

                self._ort_session = ort.InferenceSession(str(self.onnx_model_path), providers=providers_to_try)
                active_prov = self._ort_session.get_providers()[0]
                
                if "Dml" in active_prov:
                    self._backend = "DirectML"
                    self._device_name = "DirectML GPU/NPU (AMD/Intel/Qualcomm)"
                elif "CUDA" in active_prov:
                    self._backend = "CUDA"
                    self._device_name = "NVIDIA CUDA GPU"
                elif "CoreML" in active_prov:
                    self._backend = "CoreML"
                    self._device_name = "Apple Silicon Neural Engine (CoreML)"
                else:
                    self._backend = "ONNX-INT8"
                    self._device_name = "CPU (INT8 Quantized)"

                # Load Tokenizer
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                print(f"[Mimir RuntimeLMScorer] Initialized ONNX INT8 session on {self._device_name} ({active_prov})", file=sys.stderr)
                return True
            except Exception as e:
                print(f"[Mimir RuntimeLMScorer] ONNX init notice ({e}); falling back to PyTorch...", file=sys.stderr)

        # 2. PyTorch Hardware Discovery
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._torch = torch

            # Hardware Device Discovery
            if self._requested_device is not None:
                self._device = self._requested_device
                self._backend = str(self._device)
                self._device_name = str(self._device)
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._backend = "CUDA"
                self._device_name = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                self._backend = "MPS"
                self._device_name = "Apple Silicon GPU/ANE (Metal)"
            else:
                try:
                    import torch_directml
                    self._device = torch_directml.device()
                    self._backend = "DirectML"
                    self._device_name = torch_directml.device_name(0)
                except Exception:
                    self._device = torch.device("cpu")
                    self._backend = "CPU"
                    self._device_name = "CPU"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._torch_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._torch_model.eval()
            self._torch_model.to(self._device)

            print(f"[Mimir RuntimeLMScorer] Initialized PyTorch backbone on {self._device_name} ({self._backend})", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[Mimir RuntimeLMScorer] PyTorch init failed: {e}", file=sys.stderr)
            return False

    def get_hardware_info(self) -> Dict[str, str]:
        return {"device": str(self._device or "ort"), "name": self._device_name, "backend": self._backend}

    def score(self, text: str) -> Dict[str, float]:
        cleaned = text.strip()
        if not cleaned:
            return self._empty()

        if self._ort_session is None and self._torch_model is None:
            if not self.probe_hardware_and_load():
                return self._empty()

        if self._tokenizer is None:
            return self._empty()

        # Tokenize
        tokens = self._tokenizer(
            cleaned,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids_np = tokens["input_ids"].astype(np.int64)  # [1, seq_len]
        seq_len = input_ids_np.shape[1]
        if seq_len < 3:
            return self._empty()

        # Forward pass via ONNX Runtime or PyTorch
        if self._ort_session is not None:
            ort_out = self._ort_session.run(["logits"], {"input_ids": input_ids_np})
            logits_np = ort_out[0][0, :-1, :]  # [seq_len - 1, vocab]
            targets_np = input_ids_np[0, 1:]    # [seq_len - 1]
            return self._compute_metrics_numpy(logits_np, targets_np)
        
        elif self._torch_model is not None:
            torch = self._torch
            input_ids = torch.tensor(input_ids_np, device=self._device)
            with torch.no_grad():
                outputs = self._torch_model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
                nll = -target_log_probs
                probs = torch.exp(log_probs).squeeze(0)
                entropy = -(probs * log_probs.squeeze(0)).sum(dim=-1)
                ranks = (logits.squeeze(0) > target_log_probs.unsqueeze(-1)).sum(dim=-1) + 1

                nll_np = nll.detach().cpu().numpy()
                ent_np = entropy.detach().cpu().numpy()
                ranks_np = ranks.detach().cpu().numpy()
                logprobs_np = target_log_probs.detach().cpu().numpy()

            return self._assemble_features(nll_np, ent_np, ranks_np, logprobs_np, input_ids_np[0, 1:])

        return self._empty()

    def _compute_metrics_numpy(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute stable log_softmax, NLL, entropy, and ranks purely in NumPy."""
        # Log-Sum-Exp for numerically stable log_softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        log_probs = (logits - max_logits) - np.log(sum_exp)

        # Target token log-probabilities
        row_indices = np.arange(len(targets))
        target_log_probs = log_probs[row_indices, targets]
        nll = -target_log_probs

        # Entropy: -sum(p * log_p)
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * log_probs, axis=-1)

        # Ranks: count of elements greater than target log-prob + 1
        ranks = np.sum(logits > target_log_probs[:, None], axis=-1) + 1

        return self._assemble_features(nll, entropy, ranks, target_log_probs, targets)

    def _assemble_features(
        self,
        nll_np: np.ndarray,
        ent_np: np.ndarray,
        ranks_np: np.ndarray,
        logprobs_np: np.ndarray,
        target_tokens: np.ndarray,
    ) -> Dict[str, float]:
        n_tokens = float(len(nll_np))
        if n_tokens == 0:
            return self._empty()

        mean_nll = float(np.mean(nll_np))
        ppl_global = float(math.exp(min(mean_nll, 20.0)))
        mean_entropy = float(np.mean(ent_np))
        ppl_norm = float(mean_nll / max(mean_entropy, 1e-4))
        lp_mean = float(np.mean(logprobs_np))
        lp_std = float(np.std(logprobs_np))
        lp_diff = logprobs_np - lp_mean
        lp_skew = float(np.mean(lp_diff ** 3) / max(lp_std ** 3, 1e-6))
        lp_kurt = float(np.mean(lp_diff ** 4) / max(lp_std ** 4, 1e-6))
        top1_frac = float(np.mean(ranks_np == 1))
        top5_frac = float(np.mean(ranks_np <= 5))
        top10_frac = float(np.mean(ranks_np <= 10))
        top100_frac = float(np.mean(ranks_np <= 100))
        tail_frac = float(np.mean(ranks_np > 100))

        # Sentence burstiness
        dot_tokens = {
            self._tokenizer.encode(".")[0],
            self._tokenizer.encode("!")[0],
            self._tokenizer.encode("?")[0],
        } if self._tokenizer else {13, 0, 30}

        sent_ppls = []
        cur_nlls = []
        for tok_id, nll_val in zip(target_tokens, nll_np):
            cur_nlls.append(nll_val)
            if tok_id in dot_tokens and len(cur_nlls) >= 3:
                sent_ppls.append(math.exp(min(np.mean(cur_nlls), 20.0)))
                cur_nlls = []
        if cur_nlls and len(cur_nlls) >= 3:
            sent_ppls.append(math.exp(min(np.mean(cur_nlls), 20.0)))

        sent_ppl_mean = float(np.mean(sent_ppls)) if sent_ppls else ppl_global
        sent_ppl_std = float(np.std(sent_ppls)) if len(sent_ppls) > 1 else 0.0
        sent_ppl_min = float(np.min(sent_ppls)) if sent_ppls else ppl_global
        sent_ppl_max = float(np.max(sent_ppls)) if sent_ppls else ppl_global
        sent_ppl_range = float(sent_ppl_max / max(sent_ppl_min, 1e-4))
        curv_disc = float(np.mean(np.abs(nll_np - ent_np)))

        return {
            "lm_ppl_global": ppl_global, "lm_mean_nll": mean_nll,
            "lm_entropy_mean": mean_entropy, "lm_ppl_norm": ppl_norm,
            "lm_logprob_mean": lp_mean, "lm_logprob_std": lp_std,
            "lm_logprob_skew": lp_skew, "lm_logprob_kurtosis": lp_kurt,
            "lm_top1_frac": top1_frac, "lm_top5_frac": top5_frac,
            "lm_top10_frac": top10_frac, "lm_top100_frac": top100_frac,
            "lm_tail_frac": tail_frac, "lm_sent_ppl_mean": sent_ppl_mean,
            "lm_sent_ppl_std": sent_ppl_std, "lm_sent_ppl_range_ratio": sent_ppl_range,
            "lm_curvature_discrepancy": curv_disc,
        }

    def _empty(self) -> Dict[str, float]:
        return {
            "lm_ppl_global": 0.0, "lm_mean_nll": 0.0, "lm_entropy_mean": 0.0,
            "lm_ppl_norm": 0.0, "lm_logprob_mean": 0.0, "lm_logprob_std": 0.0,
            "lm_logprob_skew": 0.0, "lm_logprob_kurtosis": 0.0, "lm_top1_frac": 0.0,
            "lm_top5_frac": 0.0, "lm_top10_frac": 0.0, "lm_top100_frac": 0.0,
            "lm_tail_frac": 0.0, "lm_sent_ppl_mean": 0.0, "lm_sent_ppl_std": 0.0,
            "lm_sent_ppl_range_ratio": 0.0, "lm_curvature_discrepancy": 0.0,
        }
