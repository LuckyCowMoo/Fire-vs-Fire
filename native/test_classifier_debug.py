#!/usr/bin/env python3
"""Quick debug test for the ConvNeXtLargeArtifactV2 classifier."""

import sys
from pathlib import Path
from PIL import Image
import time

# Add native dir to path
sys.path.insert(0, str(Path(__file__).parent))

from classifiers.convnext_large_artifact_classifier_V2 import ConvNeXtLargeArtifactV2Classifier

# Create a test image (simple pattern)
print("[TEST] Creating test image...")
test_img = Image.new('RGB', (224, 224), color='red')

# Initialize classifier
print("[TEST] Initializing classifier...")
classifier = ConvNeXtLargeArtifactV2Classifier()

# Load model
print("[TEST] Loading model...")
success, error = classifier.load_model()
if not success:
    print(f"[TEST] FAILED to load model: {error}")
    sys.exit(1)

print(f"[TEST] Model loaded successfully")
print(f"[TEST] Device info: {classifier.get_device_info()}")
print(f"[TEST] Model path: {classifier.model_path}")
print(f"[TEST] Supported modalities: {classifier.get_supported_modalities()}")

# Test preprocessing
print(f"\n[TEST] Testing preprocessing...")
batch_data, valid_indices = classifier.preprocess_batch([test_img], 'image')
if batch_data is None:
    print("[TEST] FAILED at preprocessing")
    sys.exit(1)

print(f"[TEST] Preprocessing OK")
print(f"[TEST] RGB tensor shape: {batch_data['rgb'].shape}")
print(f"[TEST] Artifact tensor shape: {batch_data['artifact'].shape}")

# Test inference
print(f"\n[TEST] Testing inference...")
t0 = time.time()
scores = classifier.classify_batch(batch_data)
inference_ms = int((time.time() - t0) * 1000)

print(f"[TEST] Inference took {inference_ms}ms")
print(f"[TEST] Scores: {scores}")
print(f"[TEST] AI class index: {classifier._ai_class_index}")

if scores and len(scores) > 0:
    score = scores[0]
    print(f"[TEST] Score value: {score}")
    print(f"[TEST] Label: {'AI' if score >= 0.5 else 'Real'}")
else:
    print("[TEST] No scores returned")
    
print("\n[TEST] Test complete!")
