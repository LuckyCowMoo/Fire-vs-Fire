#!/usr/bin/env python3
"""Test with flipped AI class index."""

import sys
import os
from pathlib import Path
from PIL import Image

# Set the env variable BEFORE importing the classifier
os.environ['FIRE_CONVNEXT_AI_CLASS_INDEX'] = '1'

sys.path.insert(0, str(Path(__file__).parent))

from classifiers.convnext_large_artifact_classifier_V2 import ConvNeXtLargeArtifactV2Classifier

# Create test images (different patterns)
print("[TEST] Creating test images...")
red_img = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red
green_img = Image.new('RGB', (224, 224), color=(0, 255, 0))  # Green

classifier = ConvNeXtLargeArtifactV2Classifier()
success, error = classifier.load_model()
if not success:
    print(f"FAILED: {error}")
    sys.exit(1)

print(f"[TEST] Model loaded with AI class index: {classifier._ai_class_index}")

# Test with both images
images = [red_img, green_img]
batch_data, valid_indices = classifier.preprocess_batch(images, 'image')
scores = classifier.classify_batch(batch_data)

print(f"\n[TEST] Scores with AI class index {classifier._ai_class_index}:")
print(f"  Red image:   {scores[0]:.4f} → {('AI' if scores[0] >= 0.5 else 'Real')}")
print(f"  Green image: {scores[1]:.4f} → {('AI' if scores[1] >= 0.5 else 'Real')}")
