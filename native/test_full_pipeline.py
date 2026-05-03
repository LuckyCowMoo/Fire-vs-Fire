#!/usr/bin/env python3
"""Test the full pipeline like the native host does."""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from model_registry import get_registry

# Create test images
print("[TEST] Creating test images...")
red_img = Image.new('RGB', (224, 224), color=(255, 0, 0))
green_img = Image.new('RGB', (224, 224), color=(0, 255, 0))

# Get registry and load classifier
print("[TEST] Getting registry...")
registry = get_registry()
registry.discover_classifiers()

print("[TEST] Available classifiers:")
for clf_id in registry._classifiers.keys():
    print(f"  - {clf_id}")

# Try the conv_ne_xt_large_artifact_v2 classifier
print("\n[TEST] Loading conv_ne_xt_large_artifact_v2 classifier...")
classifier = registry.get_classifier('conv_ne_xt_large_artifact_v2', lazy_load=False)
if classifier is None:
    print("[TEST] FAILED - classifier returned None")
    sys.exit(1)

print(f"[TEST] Classifier type: {type(classifier)}")
print(f"[TEST] Is loaded: {classifier.is_loaded()}")

if not classifier.is_loaded():
    print("[TEST] Loading model...")
    success, error = classifier.load_model()
    if not success:
        print(f"[TEST] FAILED to load: {error}")
        sys.exit(1)
    print(f"[TEST] Model loaded successfully")

print(f"[TEST] AI class index: {classifier._ai_class_index}")
print(f"[TEST] Device: {classifier.get_device_info()}")

# Test process_batch
print("\n[TEST] Testing process_batch...")
scores = classifier.process_batch([red_img, green_img], 'image')
print(f"[TEST] Scores: {scores}")
print(f"[TEST] Passed! This means the pipeline works.")
