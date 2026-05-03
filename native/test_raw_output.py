#!/usr/bin/env python3
"""Debug test to check raw model output."""

import sys
from pathlib import Path
from PIL import Image
import torch

# Add native dir to path
sys.path.insert(0, str(Path(__file__).parent))

from classifiers.convnext_large_artifact_classifier_V2 import ConvNeXtLargeArtifactV2Classifier

# Create test images (different patterns)
print("[TEST] Creating test images...")
ai_img = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red
real_img = Image.new('RGB', (224, 224), color=(0, 255, 0))  # Green

classifier = ConvNeXtLargeArtifactV2Classifier()
success, error = classifier.load_model()
if not success:
    print(f"FAILED: {error}")
    sys.exit(1)

print("[TEST] Model loaded")

# Test with both images
images = [ai_img, real_img]
batch_data, valid_indices = classifier.preprocess_batch(images, 'image')

# Get raw logits
rgb = batch_data['rgb']
artifact = batch_data['artifact']

with torch.no_grad():
    logits = classifier.model(rgb, artifact)
    probs = torch.softmax(logits, dim=1)

print(f"\n[TEST] Raw logits:\n{logits}")
print(f"\n[TEST] Probabilities:\n{probs}")
print(f"\n[TEST] AI class index: {classifier._ai_class_index}")
print(f"\n[TEST] Using class index {classifier._ai_class_index} as AI:")
print(f"  Image 0 (red):   {probs[0, classifier._ai_class_index].item():.4f}")
print(f"  Image 1 (green): {probs[1, classifier._ai_class_index].item():.4f}")

# Try the opposite class too
other_idx = 1 - classifier._ai_class_index
print(f"\n[TEST] Using class index {other_idx} as AI:")
print(f"  Image 0 (red):   {probs[0, other_idx].item():.4f}")
print(f"  Image 1 (green): {probs[1, other_idx].item():.4f}")
