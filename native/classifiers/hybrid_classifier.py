"""Hybrid AI-vs-human text classifier.

This module provides two things:
- a feature extractor that computes the dataset-style handcrafted features
- a BaseClassifier implementation that can run either with a trained
  DistilBERT+feature model or a deterministic feature baseline when weights
  are not available yet

The goal is to keep the code usable inside the native app harness while still
supporting a proper hybrid model once training is finished.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
import re
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .base_classifier import BaseClassifier
except Exception:  # pragma: no cover
    import importlib.util

    _base_path = Path(__file__).resolve().parent / "base_classifier.py"
    _spec = importlib.util.spec_from_file_location("classifier_base", str(_base_path))
    if _spec is None or _spec.loader is None:
        raise ImportError("Could not load base_classifier")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


FEATURE_COLUMNS: Tuple[str, ...] = (
    "unique_words_relative",
    "flesch_reading_ease",
    "flesch_kincaid_grade_level",
    "personal_pronoun_relative",
    "pos_per_sentence_mean",
    "words_per_sentence_mean",
    "words_per_sentence_stdev",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "uppercase_letters_relative",
    "unique_words_per_sentence_mean",
    "unique_words_per_sentence_stdev",
    "ppl_mean",
    "token_logprob_mean",
    "token_logprob_std",
    "token_top1_frac",
    "token_top5_frac",
    "token_top10_frac",
    "token_entropy_mean",
)


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_VOWELS = set("aeiouy")
_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "strong",
    "effective",
    "beneficial",
    "clear",
    "smart",
    "best",
    "successful",
    "improve",
    "help",
    "progress",
    "important",
    "valuable",
}
_NEGATIVE_WORDS = {
    "bad",
    "poor",
    "weak",
    "negative",
    "hard",
    "difficult",
    "problem",
    "issue",
    "worst",
    "fail",
    "failure",
    "wrong",
    "confusing",
    "risk",
}
_PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "they",
    "them",
    "their",
    "theirs",
}
_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
_PREPOSITIONS = {
    "in",
    "on",
    "at",
    "by",
    "for",
    "to",
    "from",
    "of",
    "with",
    "about",
    "into",
    "through",
    "between",
    "during",
    "before",
    "after",
    "over",
    "under",
    "around",
    "without",
    "within",
}
_CONJUNCTIONS = {"and", "or", "but", "so", "yet", "because", "although", "while", "if", "when"}
_COMMON_VERBS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "make",
    "made",
    "use",
    "used",
    "show",
    "shows",
    "showed",
    "say",
    "says",
    "said",
    "go",
    "goes",
    "went",
    "get",
    "got",
    "think",
    "thinks",
    "thought",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _tokenize_words(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _split_sentences(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in _SENTENCE_RE.split(cleaned) if part.strip()]
    return parts or [cleaned]


def _count_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 0
    syllables = 0
    previous_is_vowel = False
    for character in token:
        is_vowel = character in _VOWELS
        if is_vowel and not previous_is_vowel:
            syllables += 1
        previous_is_vowel = is_vowel
    if token.endswith("e") and syllables > 1 and not token.endswith(("le", "ye")):
        syllables -= 1
    return max(syllables, 1)


def _estimate_pos_categories(tokens: Sequence[str]) -> int:
    categories = set()
    for token in tokens:
        if token in _PRONOUNS:
            categories.add("pronoun")
        elif token in _DETERMINERS:
            categories.add("determiner")
        elif token in _PREPOSITIONS:
            categories.add("preposition")
        elif token in _CONJUNCTIONS:
            categories.add("conjunction")
        elif token in _COMMON_VERBS or token.endswith(("ed", "ing")):
            categories.add("verb")
        elif token.endswith(("ly",)):
            categories.add("adverb")
        elif token.endswith(("ous", "ful", "able", "ible", "ive", "al", "ish")):
            categories.add("adjective")
        else:
            categories.add("noun")
    return len(categories)


def _sentiment_scores(tokens: Sequence[str]) -> Tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    positive_hits = sum(1 for token in tokens if token in _POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in _NEGATIVE_WORDS)
    matched = positive_hits + negative_hits
    polarity = _safe_divide(positive_hits - negative_hits, len(tokens))
    subjectivity = _safe_divide(matched, len(tokens))
    return polarity, subjectivity


def _readability_scores(word_count: int, sentence_count: int, syllable_count: int) -> Tuple[float, float]:
    if word_count == 0 or sentence_count == 0:
        return 0.0, 0.0
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count if word_count else 0.0
    flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    flesch_kincaid_grade_level = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    return flesch_reading_ease, flesch_kincaid_grade_level


@dataclass
class TextFeatureVector:
    """Container for one row of features."""

    values: Dict[str, float]

    def as_list(self, feature_names: Sequence[str] = FEATURE_COLUMNS) -> List[float]:
        return [float(self.values.get(name, 0.0)) for name in feature_names]


class GPT2PerplexityScorer:
    """Exact GPT-2 perplexity scorer with sliding token windows.

    ppl = exp(sum(nll) / total_scored_tokens)
    """

    def __init__(self, model_name: str = "gpt2", max_length: int = 1024, stride: int = 256, device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.device = device
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            print(f"[GPT2PerplexityScorer] Import failed: {e}", file=__import__('sys').stderr, flush=True)
            return False

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        if self.device is not None:
            requested = str(self.device).lower()
            if requested.startswith("cuda") and torch.cuda.is_available():
                self._device = torch.device(requested)
            elif requested == "cpu":
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        print(f"[GPT2PerplexityScorer] Loaded {self.model_name} on {self._device}", file=__import__('sys').stderr, flush=True)
        return True

    def score(self, text: str) -> Dict[str, float]:
        if not text.strip():
            return {
                "ppl_mean": 0.0,
                "token_logprob_mean": 0.0,
                "token_logprob_std": 0.0,
                "token_top1_frac": 0.0,
                "token_top5_frac": 0.0,
                "token_top10_frac": 0.0,
                "token_entropy_mean": 0.0,
            }

        if not self._ensure_loaded():
            result = self._fallback_score(text)
            print(f"[GPT2PerplexityScorer] Using fallback (fast heuristic) ppl={result.get('ppl_mean', 0):.1f}", file=__import__('sys').stderr, flush=True)
            return result

        tokenized = self._tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        seq_len = input_ids.size(1)
        if seq_len < 2:
            return {
                "ppl_mean": 0.0,
                "token_logprob_mean": 0.0,
                "token_logprob_std": 0.0,
                "token_top1_frac": 0.0,
                "token_top5_frac": 0.0,
                "token_top10_frac": 0.0,
                "token_entropy_mean": 0.0,
            }

        total_nll = 0.0
        total_tokens = 0
        scored_logprobs: List[float] = []
        scored_entropies: List[float] = []
        top1_hits = 0
        top5_hits = 0
        top10_hits = 0

        torch = self._torch
        stride = max(1, min(self.stride, self.max_length))
        previous_end = 0
        with torch.no_grad():
            for begin in range(0, seq_len, stride):
                end = min(begin + self.max_length, seq_len)
                window_input = input_ids[:, begin:end].to(self._device)
                window_length = window_input.size(1)
                if window_length < 2:
                    previous_end = end
                    continue

                if begin == 0:
                    score_start = 1
                else:
                    score_start = max(previous_end - begin, 1)
                if score_start >= window_length:
                    previous_end = end
                    continue

                outputs = self._model(input_ids=window_input)
                logits = outputs.logits[:, score_start - 1 : window_length - 1, :]
                targets = window_input[:, score_start:]
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

                scored_count = targets.numel()
                total_nll += float(-target_log_probs.sum().item())
                total_tokens += scored_count
                scored_logprobs.extend(target_log_probs.detach().cpu().reshape(-1).tolist())
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
                scored_entropies.extend(entropy.detach().cpu().reshape(-1).tolist())

                top1 = logits.argmax(dim=-1)
                top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices
                top10 = logits.topk(min(10, logits.size(-1)), dim=-1).indices
                top1_hits += int((top1 == targets).sum().item())
                top5_hits += int((top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item())
                top10_hits += int((top10 == targets.unsqueeze(-1)).any(dim=-1).sum().item())
                previous_end = end

        if total_tokens <= 0:
            return self._fallback_score(text)

        mean_nll = total_nll / total_tokens
        perplexity = exp(mean_nll)
        logprob_mean = sum(scored_logprobs) / len(scored_logprobs) if scored_logprobs else -mean_nll
        logprob_std = pstdev(scored_logprobs) if len(scored_logprobs) > 1 else 0.0
        entropy_mean = sum(scored_entropies) / len(scored_entropies) if scored_entropies else 0.0
        result = {
            "ppl_mean": perplexity,
            "token_logprob_mean": logprob_mean,
            "token_logprob_std": logprob_std,
            "token_top1_frac": top1_hits / total_tokens,
            "token_top5_frac": top5_hits / total_tokens,
            "token_top10_frac": top10_hits / total_tokens,
            "token_entropy_mean": entropy_mean,
        }
        print(f"[GPT2PerplexityScorer] Using real GPT-2 perplexity ppl={perplexity:.1f}", file=__import__('sys').stderr, flush=True)
        return result

    def _fallback_score(self, text: str) -> Dict[str, float]:
        words = _tokenize_words(text)
        if not words:
            return {
                "ppl_mean": 1.0,
                "token_logprob_mean": 0.0,
                "token_logprob_std": 0.0,
                "token_top1_frac": 0.0,
                "token_top5_frac": 0.0,
                "token_top10_frac": 0.0,
                "token_entropy_mean": 0.0,
            }

        unique_ratio = len(set(words)) / len(words)
        repetition_ratio = 1.0 - unique_ratio
        avg_word_length = sum(len(word) for word in words) / len(words)
        punctuation_ratio = sum(1 for character in text if character in ",;:-()[]{}") / max(len(text), 1)
        approximated_ppl = 1.0 + (avg_word_length * 2.2) + (repetition_ratio * 25.0) + (punctuation_ratio * 8.0)
        return {
            "ppl_mean": approximated_ppl,
            "token_logprob_mean": -log(max(approximated_ppl, 1.0)),
            "token_logprob_std": repetition_ratio,
            "token_top1_frac": max(0.0, 1.0 - repetition_ratio),
            "token_top5_frac": min(1.0, 0.55 + unique_ratio * 0.4),
            "token_top10_frac": min(1.0, 0.75 + unique_ratio * 0.2),
            "token_entropy_mean": log(len(set(words)) + 1.0),
        }


class TextFeatureExtractor:
    """Compute the hybrid classifier's handcrafted features from raw text."""

    def __init__(self, perplexity_scorer: Optional[GPT2PerplexityScorer] = None):
        self.perplexity_scorer = perplexity_scorer or GPT2PerplexityScorer()

    def extract(self, text: str) -> TextFeatureVector:
        text = text or ""
        words = _tokenize_words(text)
        sentences = _split_sentences(text)
        sentence_words = [_tokenize_words(sentence) for sentence in sentences]
        total_words = len(words)
        total_sentences = max(len(sentences), 1)
        unique_words = len(set(words))
        unique_words_relative = _safe_divide(unique_words, total_words)
        words_per_sentence = [len(words_in_sentence) for words_in_sentence in sentence_words]
        words_per_sentence_mean = sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0.0
        words_per_sentence_stdev = pstdev(words_per_sentence) if len(words_per_sentence) > 1 else 0.0
        unique_words_per_sentence = [len(set(words_in_sentence)) for words_in_sentence in sentence_words if words_in_sentence]
        unique_words_per_sentence_mean = mean(unique_words_per_sentence) if unique_words_per_sentence else 0.0
        unique_words_per_sentence_stdev = pstdev(unique_words_per_sentence) if len(unique_words_per_sentence) > 1 else 0.0
        pronouns = sum(1 for word in words if word in _PRONOUNS)
        personal_pronoun_relative = _safe_divide(pronouns, total_words)
        uppercase_letters = sum(1 for character in text if character.isupper())
        letters = sum(1 for character in text if character.isalpha())
        uppercase_letters_relative = _safe_divide(uppercase_letters, letters)
        syllable_count = sum(_count_syllables(word) for word in words)
        flesch_reading_ease, flesch_kincaid_grade_level = _readability_scores(total_words, total_sentences, syllable_count)
        pos_per_sentence_mean = mean(_estimate_pos_categories(sentence_words_in_sentence) for sentence_words_in_sentence in sentence_words) if sentence_words else 0.0
        sentiment_polarity, sentiment_subjectivity = _sentiment_scores(words)

        lm_stats = self.perplexity_scorer.score(text)

        values = {
            "unique_words_relative": unique_words_relative,
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade_level": flesch_kincaid_grade_level,
            "personal_pronoun_relative": personal_pronoun_relative,
            "pos_per_sentence_mean": pos_per_sentence_mean,
            "words_per_sentence_mean": words_per_sentence_mean,
            "words_per_sentence_stdev": words_per_sentence_stdev,
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "uppercase_letters_relative": uppercase_letters_relative,
            "unique_words_per_sentence_mean": unique_words_per_sentence_mean,
            "unique_words_per_sentence_stdev": unique_words_per_sentence_stdev,
            **lm_stats,
        }
        return TextFeatureVector(values=values)


class HybridTextModel:
    """Placeholder class that will be instantiated only when torch is available."""


class HybridTextClassifier(BaseClassifier):
    """AI-vs-human classifier that plugs into the BaseClassifier harness."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        text_model_name: str = "distilbert-base-uncased",
        feature_importance_init: Optional[Dict[str, float]] = None,
        max_length: int = 256,
    ):
        super().__init__(model_path=model_path or Path("models/hybrid_classifier.pt"))
        self.text_model_name = text_model_name
        self.max_length = max_length
        self.feature_importance_init = feature_importance_init or {
            "ppl_mean": 1.6,
            "token_logprob_mean": 1.2,
            "token_entropy_mean": 1.1,
            "flesch_reading_ease": -0.8,
            "flesch_kincaid_grade_level": 0.8,
            "unique_words_relative": -0.5,
            "unique_words_per_sentence_mean": -0.4,
            "sentiment_subjectivity": -0.2,
            "words_per_sentence_mean": 0.3,
            "words_per_sentence_stdev": 0.2,
            "uppercase_letters_relative": 0.1,
            "personal_pronoun_relative": -0.15,
            "token_top1_frac": -0.7,
            "token_top5_frac": -0.4,
            "token_top10_frac": -0.25,
            "token_logprob_std": 0.2,
            "sentiment_polarity": -0.1,
            "pos_per_sentence_mean": -0.1,
        }
        self.feature_extractor = TextFeatureExtractor()
        self._device_info: Dict[str, Any] = {"device": "cpu", "name": "CPU", "backend": "CPU"}
        self._mode = "baseline"
        self._torch = None
        self._nn = None
        self._F = None
        self._tokenizer = None
        self._model = None
        self._baseline_weights = self._build_baseline_weights()

    def get_supported_modalities(self) -> Set[str]:
        return {"text"}

    def get_model_name(self) -> str:
        return "HybridTextClassifier-DistilBERT+Features"

    def get_device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def _build_baseline_weights(self) -> List[float]:
        weights = []
        for feature_name in FEATURE_COLUMNS:
            weights.append(float(self.feature_importance_init.get(feature_name, 0.0)))
        return weights

    def _ensure_torch_stack(self) -> bool:
        if self._torch is not None:
            return True
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return False
        self._torch = torch
        self._nn = nn
        self._F = F
        self._AutoModel = AutoModel
        self._AutoTokenizer = AutoTokenizer
        return True

    def _device_from_torch(self):
        torch = self._torch
        if torch is None:
            return None
        return self._try_cuda_then_directml(torch)

    def load_model(self) -> Tuple[bool, Optional[str]]:
        if not self._ensure_torch_stack():
            self._mode = "baseline"
            self._is_loaded = True
            print("[HybridTextClassifier] Running in baseline mode because torch/transformers are not installed yet.", file=__import__('sys').stderr, flush=True)
            return True, None

        torch = self._torch
        device = self._device_from_torch()
        self.device = device

        if self.model_path and self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
                config = checkpoint.get("config", {})
                self._tokenizer = self._AutoTokenizer.from_pretrained(config.get("text_model_name", self.text_model_name))
                model = _HybridTextTorchModel(
                    feature_names=FEATURE_COLUMNS,
                    text_model_name=config.get("text_model_name", self.text_model_name),
                    feature_importance_init=checkpoint.get("feature_importance_init", self._baseline_weights),
                    feature_hidden_dim=int(config.get("feature_hidden_dim", 128)),
                    merge_hidden_dim=int(config.get("merge_hidden_dim", 128)),
                    dropout=float(config.get("dropout", 0.2)),
                )
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict")), strict=False)
                model.to(device)
                model.eval()
                self._model = model
                self._mode = "hybrid"
                self._device_info = {
                    "device": str(device),
                    "name": checkpoint.get("device_name", self._device_info.get("name", "CPU")),
                    "backend": checkpoint.get("backend", self._device_info.get("backend", "CPU")),
                }
                self._is_loaded = True
                print(f"[HybridTextClassifier] Loaded hybrid classifier from {self.model_path}", file=__import__('sys').stderr, flush=True)
                return True, None
            except Exception as exc:
                return False, f"failed to load model checkpoint: {exc}"

        self._mode = "baseline"
        self._device_info = {"device": "cpu", "name": "CPU", "backend": "CPU"}
        self._is_loaded = True
        print("[HybridTextClassifier] No trained weights found; using deterministic feature baseline until training is completed.", file=__import__('sys').stderr, flush=True)
        return True, None

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "text":
            return {"texts": [], "features": [], "input_ids": None, "attention_mask": None}, []

        valid_indices: List[int] = []
        valid_texts: List[str] = []
        feature_rows: List[List[float]] = []
        for index, item in enumerate(inputs):
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            valid_indices.append(index)
            valid_texts.append(cleaned)
            feature_rows.append(self.feature_extractor.extract(cleaned).as_list())

        if not valid_texts:
            return {"texts": [], "features": [], "input_ids": None, "attention_mask": None}, []

        batch: Dict[str, Any] = {
            "texts": valid_texts,
            "features": feature_rows,
            "input_ids": None,
            "attention_mask": None,
        }

        if self._mode == "hybrid" and self._tokenizer is not None:
            encoded = self._tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch["input_ids"] = encoded["input_ids"].to(self.device)
            batch["attention_mask"] = encoded["attention_mask"].to(self.device)

        return batch, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        texts = batch_tensor.get("texts", [])
        features = batch_tensor.get("features", [])
        if not texts:
            return []

        if self._mode == "hybrid" and self._model is not None:
            torch = self._torch
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self._model(
                    input_ids=batch_tensor.get("input_ids"),
                    attention_mask=batch_tensor.get("attention_mask"),
                    features=feature_tensor,
                )
                probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            return [float(probability) for probability in probabilities]

        weights = self._baseline_weights
        bias = -0.35
        scores: List[float] = []
        for row in features:
            logit = bias
            for weight, value in zip(weights, row):
                logit += weight * float(value)
            scores.append(float(1.0 / (1.0 + exp(-max(min(logit, 40.0), -40.0)))))
        return scores


class _HybridTextTorchModel:
    """Torch implementation of the hybrid network.

    This is intentionally kept separate so the module still imports cleanly in
    environments where torch/transformers are not installed.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        text_model_name: str,
        feature_importance_init: Sequence[float],
        feature_hidden_dim: int = 128,
        merge_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        torch = __import__("torch")
        nn = __import__("torch.nn", fromlist=["Module"])
        AutoModel = __import__("transformers", fromlist=["AutoModel"]).AutoModel

        class HybridModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_names = tuple(feature_names)
                self.text_backbone = AutoModel.from_pretrained(text_model_name)
                hidden_size = int(self.text_backbone.config.hidden_size)
                self.text_projection = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                feature_count = len(self.feature_names)
                self.feature_projection = nn.Sequential(
                    nn.Linear(feature_count, feature_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_hidden_dim, feature_hidden_dim),
                    nn.ReLU(),
                )
                self.merge = nn.Sequential(
                    nn.Linear(256 + feature_hidden_dim, merge_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(merge_hidden_dim, 1),
                )

            def forward(self, input_ids, attention_mask, features):
                backbone_output = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = backbone_output.last_hidden_state
                mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                text_repr = self.text_projection(pooled)
                feature_repr = self.feature_projection(features)
                merged = torch.cat([text_repr, feature_repr], dim=-1)
                return self.merge(merged).squeeze(-1)

        self.module = HybridModule()

    def __getattr__(self, item: str):
        return getattr(self.module, item)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _smoke_test() -> None:
    classifier = HybridTextClassifier()
    classifier.load_model()
    sample_texts = [
        "I think this plan is a good idea because it helps people work faster.",
        "The system accordingly synthesizes diverse indicators and produces a result.",
    ]
    batch, valid_indices = classifier.preprocess_batch(sample_texts, "text")
    scores = classifier.classify_batch(batch)
    print({"valid_indices": valid_indices, "scores": scores})


if __name__ == "__main__":
    _smoke_test()