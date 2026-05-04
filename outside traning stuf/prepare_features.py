"""Parallel feature preparation for AI-vs-human text classification.

Converts raw text rows into the hybrid feature table used by the classifier.
Supports paired source rows (human_text / ai_text) or single text+label rows.

Key improvements over v1:
- Batched GPT-2 inference (much faster than one-at-a-time)
- Resume support: skips rows already written to output
- tqdm progress bar with ETA
- --limit flag to process a subset for quick training runs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from hybrid_classifier import FEATURE_COLUMNS, TextFeatureExtractor

# Increase CSV field size limit for large text fields
csv.field_size_limit(10_000_000)


# ---------------------------------------------------------------------------
# GPT-2 batched scorer (replaces the single-text scorer for bulk extraction)
# ---------------------------------------------------------------------------

class BatchedGPT2Scorer:
    """Score a batch of texts with GPT-2 using sliding windows for accurate perplexity."""

    def __init__(self, model_name: str = "gpt2", max_length: int = 1024, stride: int = 256, device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self._device = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._requested_device = device

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        print("Loading GPT-2 tokenizer...", file=sys.stderr, flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print("Loading GPT-2 model...", file=sys.stderr, flush=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()

        req = (self._requested_device or "").lower()
        if req == "dml":
            try:
                import torch_directml
                self._device = torch_directml.device()
                print(f"GPT-2 loaded on DirectML: {torch_directml.device_name(0)}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"DirectML failed ({e}), falling back to CPU", file=sys.stderr, flush=True)
                self._device = torch.device("cpu")
        elif req.startswith("cuda") and torch.cuda.is_available():
            self._device = torch.device(req)
            print(f"GPT-2 loaded on {self._device}", file=sys.stderr, flush=True)
        else:
            self._device = torch.device("cpu")
            print(f"GPT-2 loaded on cpu", file=sys.stderr, flush=True)
        self._model.to(self._device)

    def score_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return a list of feature dicts, one per text using sliding windows."""
        self._load()
        torch = self._torch
        results: List[Dict[str, float]] = []

        for text in texts:
            if not text.strip():
                results.append(self._zeros())
                continue

            enc = self._tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"]
            seq_len = input_ids.size(1)
            if seq_len < 2:
                results.append(self._zeros())
                continue

            total_nll = 0.0
            total_tokens = 0
            scored_logprobs = []
            scored_entropies = []
            top1_hits = 0
            top5_hits = 0
            top10_hits = 0
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
                    
                    # WORKAROUND: DirectML has a bug with .gather(), move to CPU first
                    if str(self._device).startswith('privateuseone'):
                        log_probs_cpu = log_probs.cpu()
                        targets_cpu = targets.cpu()
                        target_log_probs = log_probs_cpu.gather(-1, targets_cpu.unsqueeze(-1)).squeeze(-1)
                    else:
                        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

                    scored_count = targets.numel()
                    total_nll += float(-target_log_probs.sum().item())
                    total_tokens += scored_count
                    scored_logprobs.extend(target_log_probs.detach().cpu().reshape(-1).tolist())
                    
                    # Move logits to CPU for topk operations (DirectML topk is also buggy)
                    logits_cpu = logits.cpu()
                    targets_cpu = targets.cpu()
                    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
                    scored_entropies.extend(entropy.detach().cpu().reshape(-1).tolist())

                    top1 = logits_cpu.argmax(dim=-1)
                    top5 = logits_cpu.topk(min(5, logits_cpu.size(-1)), dim=-1).indices
                    top10 = logits_cpu.topk(min(10, logits_cpu.size(-1)), dim=-1).indices
                    top1_hits += int((top1 == targets_cpu).sum().item())
                    top5_hits += int((top5 == targets_cpu.unsqueeze(-1)).any(dim=-1).sum().item())
                    top10_hits += int((top10 == targets_cpu.unsqueeze(-1)).any(dim=-1).sum().item())
                    previous_end = end

            if total_tokens <= 0:
                results.append(self._zeros())
                continue

            from math import exp
            from statistics import pstdev
            mean_nll = total_nll / total_tokens
            perplexity = exp(mean_nll)
            logprob_mean = sum(scored_logprobs) / len(scored_logprobs) if scored_logprobs else -mean_nll
            logprob_std = pstdev(scored_logprobs) if len(scored_logprobs) > 1 else 0.0
            entropy_mean = sum(scored_entropies) / len(scored_entropies) if scored_entropies else 0.0

            results.append({
                "ppl_mean": perplexity,
                "token_logprob_mean": logprob_mean,
                "token_logprob_std": logprob_std,
                "token_top1_frac": top1_hits / total_tokens,
                "token_top5_frac": top5_hits / total_tokens,
                "token_top10_frac": top10_hits / total_tokens,
                "token_entropy_mean": entropy_mean,
            })

        return results

    @staticmethod
    def _zeros() -> Dict[str, float]:
        return {
            "ppl_mean": 0.0, "token_logprob_mean": 0.0, "token_logprob_std": 0.0,
            "token_top1_frac": 0.0, "token_top5_frac": 0.0,
            "token_top10_frac": 0.0, "token_entropy_mean": 0.0,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_rows(input_path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        if "text" in fieldnames:
            for idx, row in enumerate(reader):
                if limit is not None and len(rows) >= limit:
                    break
                text = (row.get("text") or "").strip()
                if text:
                    row["__idx"] = str(idx)
                    rows.append(row)
        else:
            for idx, row in enumerate(reader):
                if limit is not None and len(rows) >= limit:
                    break
                human = (row.get("human_text") or "").strip()
                ai = (row.get("ai_text") or "").strip()
                instr = (row.get("instructions") or "").strip()
                if human:
                    rows.append({"__idx": str(idx * 2), "text": human, "label": "0",
                                 "source_type": "human", "instructions": instr})
                if ai and (limit is None or len(rows) < limit):
                    rows.append({"__idx": str(idx * 2 + 1), "text": ai, "label": "1",
                                 "source_type": "ai", "instructions": instr})
    return rows


def _already_written(output_path: Path) -> int:
    """Return number of data rows already in output (for resume)."""
    if not output_path.exists():
        return 0
    try:
        with output_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return sum(1 for _ in reader)  # Count actual CSV rows, not lines
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def prepare_features(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    limit: Optional[int],
    resume: bool,
) -> None:
    print(f"Reading {input_path} ...", file=sys.stderr, flush=True)
    rows = _read_rows(input_path, limit)
    if not rows:
        raise SystemExit(f"No usable rows found in {input_path}")
    print(f"Total rows to process: {len(rows)}", file=sys.stderr, flush=True)

    skip = _already_written(output_path) if resume else 0
    if skip:
        print(f"Resuming: skipping {skip} already-written rows", file=sys.stderr, flush=True)
        rows = rows[skip:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "label", "source_type", "instructions", *FEATURE_COLUMNS]

    # Build a lightweight extractor that does NOT use GPT-2 (we handle that separately)
    from hybrid_classifier import (
        _tokenize_words, _split_sentences, _count_syllables,
        _estimate_pos_categories, _sentiment_scores, _readability_scores,
        _safe_divide, _PRONOUNS,
    )
    from statistics import mean, pstdev

    def extract_handcrafted(text: str) -> Dict[str, float]:
        words = _tokenize_words(text)
        sentences = _split_sentences(text)
        sentence_words = [_tokenize_words(s) for s in sentences]
        total_words = len(words)
        total_sentences = max(len(sentences), 1)
        unique_words = len(set(words))
        wps = [len(sw) for sw in sentence_words]
        uwps = [len(set(sw)) for sw in sentence_words if sw]
        syllables = sum(_count_syllables(w) for w in words)
        fre, fkgl = _readability_scores(total_words, total_sentences, syllables)
        pol, subj = _sentiment_scores(words)
        pos_mean = mean(_estimate_pos_categories(sw) for sw in sentence_words) if sentence_words else 0.0
        return {
            "unique_words_relative": _safe_divide(unique_words, total_words),
            "flesch_reading_ease": fre,
            "flesch_kincaid_grade_level": fkgl,
            "personal_pronoun_relative": _safe_divide(sum(1 for w in words if w in _PRONOUNS), total_words),
            "pos_per_sentence_mean": pos_mean,
            "words_per_sentence_mean": sum(wps) / len(wps) if wps else 0.0,
            "words_per_sentence_stdev": pstdev(wps) if len(wps) > 1 else 0.0,
            "sentiment_polarity": pol,
            "sentiment_subjectivity": subj,
            "uppercase_letters_relative": _safe_divide(
                sum(1 for c in text if c.isupper()),
                sum(1 for c in text if c.isalpha()),
            ),
            "unique_words_per_sentence_mean": mean(uwps) if uwps else 0.0,
            "unique_words_per_sentence_stdev": pstdev(uwps) if len(uwps) > 1 else 0.0,
        }

    scorer = BatchedGPT2Scorer(device=device)

    write_mode = "a" if (resume and skip > 0) else "w"
    with output_path.open(write_mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        with tqdm(total=len(rows), unit="row", desc="features") as bar:
            for start in range(0, len(rows), batch_size):
                batch = rows[start: start + batch_size]
                texts = [r["text"] for r in batch]
                lm_stats = scorer.score_batch(texts)

                for row, lm in zip(batch, lm_stats):
                    hc = extract_handcrafted(row["text"])
                    out = dict(row)
                    out.update(hc)
                    out.update(lm)
                    writer.writerow(out)

                fh.flush()
                bar.update(len(batch))

    print(f"\nWrote {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare hybrid classifier features")
    p.add_argument("--input", type=Path, default=Path("Dataset/model_training_dataset.csv"))
    p.add_argument("--output", type=Path, default=Path("Dataset/features_generated.csv"))
    p.add_argument("--batch-size", type=int, default=32,
                   help="Texts per GPT-2 batch (higher = faster but more RAM)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process this many output rows (useful for quick training runs)")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already written to output file")
    return p


def main() -> None:
    args = build_parser().parse_args()
    prepare_features(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
