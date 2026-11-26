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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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

_CONTRACTION_RE = re.compile(r"\b[A-Za-z]+'[A-Za-z]{1,3}\b")
_TRANSITION_OPENER_RE = re.compile(
    r"^\s*(?:furthermore|moreover|additionally|consequently|however|nevertheless|nonetheless|"
    r"notably|importantly|overall|ultimately|essentially|interestingly|meanwhile|"
    r"conversely|similarly|specifically|in\s+conclusion|in\s+summary|in\s+addition|"
    r"for\s+instance|for\s+example|on\s+the\s+other\s+hand|that\s+said|as\s+a\s+result)\b",
    re.IGNORECASE,
)
_HEDGE_BOOSTER_WORDS = frozenset({
    "crucial", "essential", "significant", "significantly", "nuanced", "multifaceted",
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "pivotal", "testament", "tapestry", "realm", "landscape", "vibrant",
    "foster", "fosters", "fostering", "leverage", "leveraging", "seamless", "seamlessly",
    "comprehensive", "notable", "notably", "typically", "generally", "various",
    "overall", "furthermore", "moreover", "additionally", "ultimately",
    "intricate", "dynamic", "holistic", "paramount", "indispensable",
})
_BE_AUXILIARIES = frozenset({"be", "am", "is", "are", "was", "were", "been", "being"})
_STRUCTURAL_PUNCT_RE = re.compile(r"[;:\(\)\[\]\u2014]")
_ENTROPY_PUNCT_CHARS = (",", ";", ":", "-", "\u2014", "(", ")", "?", "!")

MIMIR03_LM_COLS: Tuple[str, ...] = (
    "lm_ppl_global", "lm_mean_nll", "lm_entropy_mean", "lm_ppl_norm",
    "lm_logprob_mean", "lm_logprob_std", "lm_logprob_skew", "lm_logprob_kurtosis",
    "lm_top1_frac", "lm_top5_frac", "lm_top10_frac", "lm_top100_frac",
    "lm_tail_frac", "lm_sent_ppl_mean", "lm_sent_ppl_std", "lm_sent_ppl_range_ratio",
    "lm_curvature_discrepancy",
    "lm_nll_lexical_residual", "lm_nll_iqr", "lm_nll_autocorr_lag1",
    "lm_entropy_cv", "lm_high_surprise_burst_ratio",
)

MIMIR03_STYLO_COLS: Tuple[str, ...] = (
    "stylo_words_per_sentence_mean", "stylo_ttr", "stylo_yules_k", "stylo_simpson_d",
    "stylo_flesch_reading_ease", "stylo_flesch_kincaid_grade", "stylo_ari",
    "stylo_syllables_per_word", "stylo_determiner_density", "stylo_preposition_density",
    "stylo_conjunction_density", "stylo_pronoun_3rd_density", "stylo_all_pronoun_density",
    "stylo_verb_density", "stylo_adverb_density", "stylo_adjective_density",
    "stylo_comma_density", "stylo_punct_density", "stylo_pacing_std",
    "stylo_mattr_50", "stylo_honore_r", "stylo_sentence_length_cv_clean",
    "stylo_sentence_length_iqr_norm", "stylo_adj_sent_diff_norm", "stylo_clause_length_cv",
    "stylo_transition_opener_density", "stylo_hedge_booster_density", "stylo_passive_aux_ratio",
    "stylo_contraction_density", "stylo_punct_entropy", "stylo_structural_punct_density",
    "stylo_para_length_cv",
)

MIMIR03_TAB_COLS: Tuple[str, ...] = MIMIR03_LM_COLS + MIMIR03_STYLO_COLS


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
        empty = {
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
            "stylo_mattr_50": 0.0, "stylo_honore_r": 0.0, "stylo_sentence_length_cv_clean": 0.0,
            "stylo_sentence_length_iqr_norm": 0.0, "stylo_adj_sent_diff_norm": 0.0,
            "stylo_clause_length_cv": 0.0, "stylo_transition_opener_density": 0.0,
            "stylo_hedge_booster_density": 0.0, "stylo_passive_aux_ratio": 0.0,
            "stylo_contraction_density": 0.0, "stylo_punct_entropy": 0.0,
            "stylo_structural_punct_density": 0.0, "stylo_para_length_cv": 0.0,
        }
        return empty
    sentences = [s.strip() for s in _SENTENCE_RE.split(raw) if s.strip()] or [raw]
    words = [m.group(0).lower() for m in _WORD_RE.finditer(raw)]
    n_words = len(words)
    n_sents = len(sentences)
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    w_mean = mean(sent_word_counts) if sent_word_counts else 0.0
    w_std = pstdev(sent_word_counts) if len(sent_word_counts) > 1 else np.nan
    w_cv = _safe_div(w_std, w_mean) if len(sent_word_counts) > 1 else np.nan
    syllables = sum(count_syllables(w) for w in words)
    syllables_per_word = _safe_div(syllables, n_words)
    unique_words = len(set(words))
    ttr = _safe_div(unique_words, n_words)
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
    pacing_std = pstdev(seg_lengths) if len(seg_lengths) > 1 else np.nan

    # Length- and bias-normalized Mímir 03 stylometric signals
    if n_words <= 50:
        mattr_50 = ttr
    else:
        win = 50
        w_counts = Counter(words[:win])
        distinct = len(w_counts)
        ttr_sum = distinct / float(win)
        for idx_w in range(win, n_words):
            out_w = words[idx_w - win]
            in_w = words[idx_w]
            w_counts[out_w] -= 1
            if w_counts[out_w] == 0:
                distinct -= 1
            if w_counts[in_w] == 0:
                distinct += 1
            w_counts[in_w] += 1
            ttr_sum += distinct / float(win)
        mattr_50 = ttr_sum / float(n_words - win + 1)

    word_freqs = Counter(words)
    hapax_v1 = sum(1 for c in word_freqs.values() if c == 1)
    honore_denom = max(1.0 - (hapax_v1 / max(unique_words, 1)), 0.05)
    honore_r = min(5.0, max(0.0, (0.1 * math.log(max(n_words, 2))) / honore_denom))

    if len(sent_word_counts) > 1:
        swc_arr = np.asarray(sent_word_counts, dtype=np.float32)
        sent_cv_clean = float(np.std(swc_arr) / max(float(np.mean(swc_arr)), 1.0))
        med_swc = float(np.median(swc_arr))
        iqr_swc = float(np.percentile(swc_arr, 75) - np.percentile(swc_arr, 25))
        sent_iqr_norm = iqr_swc / max(med_swc, 1.0)
        adj_diffs = np.abs(np.diff(swc_arr))
        adj_sent_diff_norm = float(np.mean(adj_diffs) / max(float(np.mean(swc_arr)), 1.0))
    else:
        sent_cv_clean = 0.0
        sent_iqr_norm = 0.0
        adj_sent_diff_norm = 0.0

    if len(seg_lengths) > 1:
        seg_arr = np.asarray(seg_lengths, dtype=np.float32)
        clause_length_cv = float(np.std(seg_arr) / max(float(np.mean(seg_arr)), 1.0))
    else:
        clause_length_cv = 0.0

    trans_openers = sum(1 for s in sentences if _TRANSITION_OPENER_RE.match(s))
    transition_opener_density = _safe_div(trans_openers, n_sents)
    hedge_count = sum(1 for w in words if w in _HEDGE_BOOSTER_WORDS)
    hedge_booster_density = _safe_div(hedge_count, n_words)

    passive_hits = 0
    for idx_w in range(len(words) - 1):
        if words[idx_w] in _BE_AUXILIARIES and words[idx_w + 1].endswith(("ed", "en")):
            passive_hits += 1
    passive_aux_ratio = _safe_div(passive_hits, max(verb_count, 1))

    contraction_count = len(_CONTRACTION_RE.findall(raw))
    contraction_density = _safe_div(contraction_count, n_words)

    punct_hist = [raw.count(ch) for ch in _ENTROPY_PUNCT_CHARS]
    punct_tot = sum(punct_hist)
    if punct_tot > 0:
        h_val = 0.0
        for c_p in punct_hist:
            if c_p > 0:
                p_p = c_p / float(punct_tot)
                h_val -= p_p * math.log(p_p)
        punct_entropy = float(h_val / math.log(len(_ENTROPY_PUNCT_CHARS)))
    else:
        punct_entropy = 0.0

    struct_punct_count = len(_STRUCTURAL_PUNCT_RE.findall(raw))
    structural_punct_density = _safe_div(struct_punct_count, n_words)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) > 1:
        p_wc = np.asarray([len(_WORD_RE.findall(p)) for p in paragraphs], dtype=np.float32)
        para_length_cv = float(np.std(p_wc) / max(float(np.mean(p_wc)), 1.0))
    else:
        para_length_cv = 0.0

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
        "stylo_mattr_50": float(mattr_50),
        "stylo_honore_r": float(honore_r),
        "stylo_sentence_length_cv_clean": float(sent_cv_clean),
        "stylo_sentence_length_iqr_norm": float(sent_iqr_norm),
        "stylo_adj_sent_diff_norm": float(adj_sent_diff_norm),
        "stylo_clause_length_cv": float(clause_length_cv),
        "stylo_transition_opener_density": float(transition_opener_density),
        "stylo_hedge_booster_density": float(hedge_booster_density),
        "stylo_passive_aux_ratio": float(passive_aux_ratio),
        "stylo_contraction_density": float(contraction_density),
        "stylo_punct_entropy": float(punct_entropy),
        "stylo_structural_punct_density": float(structural_punct_density),
        "stylo_para_length_cv": float(para_length_cv),
    }


def create_ort_session_with_warmup(
    onnx_path: Path,
    warmup_feed: Dict[str, Any],
    log_prefix: str = "[Mimir ONNX]",
    exclude_providers: Tuple[str, ...] = (),
) -> Tuple[Any, str]:
    """Create an ONNX Runtime session and run a tiny warmup pass before accepting a hardware EP."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3

    available = ort.get_available_providers()
    excluded = set(exclude_providers)
    providers_to_try = [
        ep
        for ep in (
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
        )
        if ep in available and ep not in excluded
    ]

    for ep in providers_to_try:
        try:
            sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=[ep, "CPUExecutionProvider"])
            sess.run(["logits"], warmup_feed)
            return sess, sess.get_providers()[0]
        except Exception as ep_err:
            print(f"{log_prefix} {ep} failed warmup; trying fallback ({ep_err})", file=sys.stderr)

    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    sess.run(["logits"], warmup_feed)
    return sess, sess.get_providers()[0]


def provider_to_hardware_info(active_prov: str) -> Tuple[str, str, str]:
    """Map an active ONNX Runtime ExecutionProvider name to (device, name, backend)."""
    if "Dml" in active_prov:
        return "dml", "DirectML GPU/NPU (AMD/Intel/Qualcomm)", "DirectML"
    if "CUDA" in active_prov:
        return "cuda", "NVIDIA CUDA GPU", "CUDA"
    if "CoreML" in active_prov:
        return "coreml", "Apple Silicon Neural Engine (CoreML)", "CoreML"
    if "OpenVINO" in active_prov:
        return "openvino", "Intel OpenVINO", "OpenVINO"
    if "QNN" in active_prov:
        return "qnn", "Qualcomm Hexagon NPU", "QNN"
    return "cpu", "CPU (INT8 Quantized)", "ONNX-INT8"


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
        self.onnx_model_path = (
            Path(onnx_model_path)
            if onnx_model_path is not None
            else (Path(__file__).resolve().parent / "fvf_mimir_backbone_int8.onnx")
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
                warmup_feed = {"input_ids": np.ones((1, 8), dtype=np.int64)}
                self._ort_session, active_prov = create_ort_session_with_warmup(
                    self.onnx_model_path,
                    warmup_feed,
                    log_prefix="[Mimir RuntimeLMScorer]",
                    # GPT-2 INT8 node_view Reshape faults DirectML's shared D3D12 device
                    exclude_providers=("DmlExecutionProvider",),
                )
                _, self._device_name, self._backend = provider_to_hardware_info(active_prov)

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
                self._device = torch.device(self._requested_device) if isinstance(self._requested_device, str) else self._requested_device
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
            try:
                ort_out = self._ort_session.run(["logits"], {"input_ids": input_ids_np})
            except Exception as run_err:
                if self._ort_session.get_providers()[0] == "CPUExecutionProvider":
                    raise
                print(f"[Mimir RuntimeLMScorer] Runtime EP failed ({run_err}); switching to CPUExecutionProvider", file=sys.stderr)
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                self._ort_session = ort.InferenceSession(
                    str(self.onnx_model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
                _, self._device_name, self._backend = provider_to_hardware_info("CPUExecutionProvider")
                ort_out = self._ort_session.run(["logits"], {"input_ids": input_ids_np})
            logits_np = ort_out[0][0, :-1, :]  # [seq_len - 1, vocab]
            targets_np = input_ids_np[0, 1:]    # [seq_len - 1]
            return self._compute_metrics_numpy(logits_np, targets_np, text=cleaned)
        
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
                target_logits = logits.squeeze(0).gather(-1, targets.squeeze(0).unsqueeze(-1)).squeeze(-1)
                ranks = (logits.squeeze(0) > target_logits.unsqueeze(-1)).sum(dim=-1) + 1

                nll_np = nll.detach().cpu().numpy()
                ent_np = entropy.detach().cpu().numpy()
                ranks_np = ranks.detach().cpu().numpy()
                logprobs_np = target_log_probs.detach().cpu().numpy()

            return self._assemble_features(nll_np, ent_np, ranks_np, logprobs_np, input_ids_np[0, 1:], text=cleaned)

        return self._empty()

    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Fast batched LM scoring on PyTorch CUDA/CPU or ONNX Runtime."""
        if not texts:
            return []
        if self._ort_session is None and self._torch_model is None:
            if not self.probe_hardware_and_load():
                return [self._empty() for _ in texts]

        # On ONNX Runtime, run sequential single-item calls (variable sequence padding skews logits without mask)
        if self._torch_model is None:
            return [self.score(t) for t in texts]

        torch = self._torch
        out: List[Dict[str, float]] = [self._empty() for _ in texts]
        for start in range(0, len(texts), batch_size):
            chunk = [t.strip() if isinstance(t, str) else "" for t in texts[start : start + batch_size]]
            non_empty_pos = [i for i, t in enumerate(chunk) if t]
            if not non_empty_pos:
                continue
            batch_texts = [chunk[i] for i in non_empty_pos]
            tok = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tok["input_ids"].to(self._device, non_blocking=True)
            attn = tok["attention_mask"].to(self._device, non_blocking=True)

            with torch.no_grad():
                use_bf16 = "cuda" in str(self._device).lower()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = self._torch_model(input_ids=input_ids, attention_mask=attn)
                    logits_all = outputs.logits
                valid_lens = attn[:, 1:].sum(dim=-1).cpu().tolist()
                targets_all = input_ids[:, 1:]

                for b_i, orig_rel_i in enumerate(non_empty_pos):
                    L = int(valid_lens[b_i])
                    if L < 2:
                        continue
                    lg = logits_all[b_i, :L, :].float()
                    tg = targets_all[b_i, :L]
                    log_probs = torch.log_softmax(lg, dim=-1)
                    target_lp = log_probs.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                    nll = -target_lp
                    probs = torch.exp(log_probs)
                    entropy = -(probs * log_probs).sum(dim=-1)
                    target_logits = lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                    ranks = (lg > target_logits.unsqueeze(-1)).sum(dim=-1) + 1

                    out[start + orig_rel_i] = self._assemble_features(
                        nll.detach().cpu().numpy(),
                        entropy.detach().cpu().numpy(),
                        ranks.detach().cpu().numpy(),
                        target_lp.detach().cpu().numpy(),
                        tg.detach().cpu().numpy(),
                        text=batch_texts[b_i],
                    )
        return out

    def _compute_metrics_numpy(self, logits: np.ndarray, targets: np.ndarray, text: str = "") -> Dict[str, float]:
        """Compute stable log_softmax, NLL, entropy, and ranks purely in NumPy."""
        # Log-Sum-Exp for numerically stable log_softmax
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        log_probs = (logits - max_logits) - np.log(sum_exp)

        # Target token log-probabilities and target logits
        row_indices = np.arange(len(targets))
        target_log_probs = log_probs[row_indices, targets]
        target_logits = logits[row_indices, targets]
        nll = -target_log_probs

        # Entropy: -sum(p * log_p)
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * log_probs, axis=-1)

        # Ranks: count of elements greater than target logit + 1
        ranks = np.sum(logits > target_logits[:, None], axis=-1) + 1

        return self._assemble_features(nll, entropy, ranks, target_log_probs, targets, text=text)

    def _assemble_features(
        self,
        nll_np: np.ndarray,
        ent_np: np.ndarray,
        ranks_np: np.ndarray,
        logprobs_np: np.ndarray,
        target_tokens: np.ndarray,
        text: str = "",
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
        sent_ppl_std = float(np.std(sent_ppls)) if len(sent_ppls) > 1 else np.nan
        sent_ppl_min = float(np.min(sent_ppls)) if sent_ppls else ppl_global
        sent_ppl_max = float(np.max(sent_ppls)) if sent_ppls else ppl_global
        sent_ppl_range = float(sent_ppl_max / max(sent_ppl_min, 1e-4)) if len(sent_ppls) > 1 else np.nan
        curv_disc = float(np.mean(np.abs(nll_np - ent_np)))

        # Vocabulary-rarity-residualized LM curvature metrics (Mímir 03)
        if text:
            txt_words = _WORD_RE.findall(text)
            mean_word_chars = float(np.mean([len(w) for w in txt_words])) if txt_words else 4.7
        else:
            mean_word_chars = 4.7
        nll_lexical_residual = float(mean_nll - (1.8 + 0.35 * mean_word_chars))
        nll_iqr = float(np.percentile(nll_np, 75) - np.percentile(nll_np, 25)) if len(nll_np) >= 2 else 0.0
        if len(nll_np) >= 4 and lp_std >= 1e-6:
            a0 = nll_np[:-1] - mean_nll
            a1 = nll_np[1:] - mean_nll
            denom = float(math.sqrt(float(np.sum(a0 * a0) * np.sum(a1 * a1))))
            nll_autocorr_lag1 = float(np.sum(a0 * a1) / denom) if denom > 1e-9 else 0.0
        else:
            nll_autocorr_lag1 = 0.0
        entropy_cv = float(np.std(ent_np) / max(mean_entropy, 1e-4))
        if len(nll_np) >= 4:
            p90 = float(np.percentile(nll_np, 90))
            hi = nll_np >= p90
            hi_count = int(np.sum(hi))
            if hi_count > 0:
                adj_hi = np.zeros_like(hi, dtype=bool)
                adj_hi[1:] |= hi[:-1]
                adj_hi[:-1] |= hi[1:]
                high_surprise_burst_ratio = float(np.sum(hi & adj_hi) / float(hi_count))
            else:
                high_surprise_burst_ratio = 0.0
        else:
            high_surprise_burst_ratio = 0.0

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
            "lm_nll_lexical_residual": nll_lexical_residual,
            "lm_nll_iqr": nll_iqr,
            "lm_nll_autocorr_lag1": nll_autocorr_lag1,
            "lm_entropy_cv": entropy_cv,
            "lm_high_surprise_burst_ratio": high_surprise_burst_ratio,
        }

    def _empty(self) -> Dict[str, float]:
        return {
            "lm_ppl_global": 0.0, "lm_mean_nll": 0.0, "lm_entropy_mean": 0.0,
            "lm_ppl_norm": 0.0, "lm_logprob_mean": 0.0, "lm_logprob_std": 0.0,
            "lm_logprob_skew": 0.0, "lm_logprob_kurtosis": 0.0, "lm_top1_frac": 0.0,
            "lm_top5_frac": 0.0, "lm_top10_frac": 0.0, "lm_top100_frac": 0.0,
            "lm_tail_frac": 0.0, "lm_sent_ppl_mean": 0.0, "lm_sent_ppl_std": 0.0,
            "lm_sent_ppl_range_ratio": 0.0, "lm_curvature_discrepancy": 0.0,
            "lm_nll_lexical_residual": 0.0, "lm_nll_iqr": 0.0,
            "lm_nll_autocorr_lag1": 0.0, "lm_entropy_cv": 0.0,
            "lm_high_surprise_burst_ratio": 0.0,
        }


def extract_mimir03_tab_vector(
    text: str,
    lm_scorer: Optional[RuntimeLMScorer] = None,
    lm_feats: Optional[Dict[str, float]] = None,
    sty_feats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Return a clean finite 54-d float32 vector (22 LM curvature + 32 length-normalized stylometrics)."""
    lm_d = lm_feats if lm_feats is not None else (lm_scorer.score(text) if lm_scorer is not None else {})
    st_d = sty_feats if sty_feats is not None else extract_stylometric_features(text)
    merged = {**lm_d, **st_d}
    vec = np.zeros(len(MIMIR03_TAB_COLS), dtype=np.float32)
    for i, col in enumerate(MIMIR03_TAB_COLS):
        val = merged.get(col, 0.0)
        if val is None:
            continue
        fval = float(val)
        if not math.isfinite(fval):
            continue
        vec[i] = float(np.clip(fval, -1e4, 1e4))
    return vec


class RuntimeTextEncoder:
    """Channel 1 Semantic & Discourse Forensic Encoder (DistilRoBERTa-AFPL INT8).

    Universal Hardware Priority:
    DirectML (AMD/Intel/Qualcomm) -> CUDA (NVIDIA) -> CoreML (Apple) ->
    OpenVINO (Intel) -> QNN (Qualcomm) -> ONNX CPU INT8 -> PyTorch GPU/CPU Fallback.
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilroberta-base",
        onnx_model_path: Optional[Path] = None,
        tokenizer_dir: Optional[Path] = None,
        device: Optional[Any] = None,
        max_length: int = 384,
    ):
        base_dir = Path(__file__).resolve().parent
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path or (base_dir / "fvf_mimir_encoder_int8.onnx")
        self.tokenizer_dir = tokenizer_dir or (base_dir / "mimir_tokenizer")
        self.max_length = max_length
        self._requested_device = device
        self._tokenizer = None
        self._ort_session = None
        self._torch_model = None
        self._torch = None
        self._device = None
        self._backend = "CPU"
        self._device_name = "CPU"

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        if self.tokenizer_dir and Path(self.tokenizer_dir).is_dir() and (Path(self.tokenizer_dir) / "tokenizer.json").exists():
            return AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
        return AutoTokenizer.from_pretrained(self.model_name)

    def probe_hardware_and_load(self) -> bool:
        """Initialize the Channel 1 Encoder using ONNX Runtime INT8 or PyTorch."""
        if self.onnx_model_path and self.onnx_model_path.exists():
            try:
                warmup_feed = {
                    "input_ids": np.ones((1, 8), dtype=np.int64),
                    "attention_mask": np.ones((1, 8), dtype=np.int64),
                }
                self._ort_session, active_prov = create_ort_session_with_warmup(
                    self.onnx_model_path,
                    warmup_feed,
                    log_prefix="[Mimir RuntimeTextEncoder]",
                )
                _, self._device_name, self._backend = provider_to_hardware_info(active_prov)

                self._tokenizer = self._load_tokenizer()
                print(f"[Mimir RuntimeTextEncoder] Initialized ONNX INT8 session on {self._device_name} ({active_prov})", file=sys.stderr)
                return True
            except Exception as e:
                print(f"[Mimir RuntimeTextEncoder] ONNX init notice ({e}); trying PyTorch fallback...", file=sys.stderr)

        # Fallback: PyTorch local fine-tuned checkpoint (if present in dev/training tree)
        ckpt_dir = Path(__file__).resolve().parent.parent.parent / "training" / "checkpoints" / "mimir_encoder_best"
        if ckpt_dir.is_dir():
            try:
                import torch
                from transformers import AutoModelForSequenceClassification
                self._torch = torch
                if self._requested_device is not None:
                    self._device = self._requested_device
                elif torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
                self._backend = str(self._device).upper()
                self._device_name = torch.cuda.get_device_name(0) if self._device.type == "cuda" else str(self._device)
                self._tokenizer = self._load_tokenizer()
                self._torch_model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir)).to(self._device).eval()
                print(f"[Mimir RuntimeTextEncoder] Initialized PyTorch checkpoint on {self._device_name}", file=sys.stderr)
                return True
            except Exception as e:
                print(f"[Mimir RuntimeTextEncoder] PyTorch init failed: {e}", file=sys.stderr)
        return False

    def _run_window(self, window_text: str) -> float:
        tokens = self._tokenizer(
            window_text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        if self._ort_session is not None:
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            try:
                ort_out = self._ort_session.run(["logits"], feeds)
            except Exception as run_err:
                if self._ort_session.get_providers()[0] == "CPUExecutionProvider":
                    raise
                print(f"[Mimir RuntimeTextEncoder] Runtime EP failed ({run_err}); switching to CPUExecutionProvider", file=sys.stderr)
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.log_severity_level = 3
                self._ort_session = ort.InferenceSession(
                    str(self.onnx_model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
                _, self._device_name, self._backend = provider_to_hardware_info("CPUExecutionProvider")
                ort_out = self._ort_session.run(["logits"], feeds)
            return float(np.asarray(ort_out[0]).reshape(-1)[0])

        if self._torch_model is not None and self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                ids_t = torch.tensor(input_ids, device=self._device)
                mask_t = torch.tensor(attention_mask, device=self._device)
                out = self._torch_model(input_ids=ids_t, attention_mask=mask_t)
                return float(out.logits.reshape(-1)[0].item())

        return 0.0

    def score(self, text: str) -> Dict[str, float]:
        cleaned = text.strip()
        if not cleaned:
            return {"enc_logit": 0.0, "enc_prob": 0.5}

        if self._ort_session is None and self._torch_model is None:
            if not self.probe_hardware_and_load():
                return {"enc_logit": 0.0, "enc_prob": 0.5}

        logit = self._run_window(cleaned)
        if len(cleaned) > 1600:
            w2 = cleaned[850:2500].strip()
            if len(w2) >= 120:
                logit = 0.5 * (logit + self._run_window(w2))

        prob = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
        return {"enc_logit": float(logit), "enc_prob": float(prob)}

    def get_hardware_info(self) -> Dict[str, str]:
        return {"device": str(self._device or "ort"), "name": self._device_name, "backend": self._backend}
