#!/usr/bin/env python3
"""Diagnostic test for HybridTextClassifier perplexity scoring."""

import sys
import time
from pathlib import Path

# Add classifiers to path
sys.path.insert(0, str(Path(__file__).parent / "classifiers"))

from hybrid_classifier import HybridTextClassifier

print("=" * 60)
print("HybridTextClassifier Diagnostic Test")
print("=" * 60)
print()

# Create classifier instance
print("[1] Initializing HybridTextClassifier...")
classifier = HybridTextClassifier()

# Load model and observe mode
print("[2] Loading model...")
success, error = classifier.load_model()
print(f"    Load success: {success}, Error: {error}")
print(f"    Mode: {classifier._mode}")
print()

# Test text samples
samples = [
    "The quick brown fox jumps over the lazy dog.",
    "AI is transforming how we work and live today.",
    "aaaaaaaaa aaaa aaaaa aaa aaaa aaaa aa.",
]

print(f"[3] Running batch classification on {len(samples)} samples...")
print("    Watching stderr for perplexity scoring logs...")
print()

start = time.time()

# Preprocess
batch, indices = classifier.preprocess_batch(samples, "text")
print(f"    Preprocessed {len(indices)} valid samples")
print()

# Classify
print("    Classifying...")
scores = classifier.classify_batch(batch)

elapsed = time.time() - start
print()
print(f"[4] Results:")
print(f"    Total time: {elapsed:.2f}s")
print(f"    Per-sample avg: {elapsed/len(samples):.3f}s")
print(f"    Scores: {[f'{s:.3f}' for s in scores]}")
print()
print("=" * 60)
print("Test complete. Check stderr output above for perplexity logs.")
print("=" * 60)
