#!/usr/bin/env python3
"""Test different models to find one that works."""

import sys
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))

from classifiers.convnext_large_artifact_classifier_V2 import ConvNeXtLargeArtifactV2Classifier

# Create test images
red_img = Image.new('RGB', (224, 224), color=(255, 0, 0))
green_img = Image.new('RGB', (224, 224), color=(0, 255, 0))
blue_img = Image.new('RGB', (224, 224), color=(0, 0, 255))

models = [
    'immage_classifier_V3-2_ConvNeXtLarge_Artifact_epoch0005.pt',
    'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0011_ILL_DO_IT_MYSELF_1.pt',
    'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0011_ILL_DO_IT_MYSELF_2.pt',
    'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0012.pt',
]

for model_name in models:
    model_path = Path(__file__).parent / 'classifiers' / model_name
    if not model_path.exists():
        print(f"\n❌ {model_name} - NOT FOUND")
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    classifier = ConvNeXtLargeArtifactV2Classifier(model_path)
    success, error = classifier.load_model()
    if not success:
        print(f"❌ Failed to load: {error}")
        continue
    
    # Test both class indices
    for ai_idx in [0, 1]:
        classifier._ai_class_index = ai_idx
        batch_data, _ = classifier.preprocess_batch([red_img, green_img, blue_img], 'image')
        scores = classifier.classify_batch(batch_data)
        
        print(f"\nWith AI class index {ai_idx}:")
        print(f"  Red:   {scores[0]:.4f} → {('AI' if scores[0] >= 0.5 else 'Real')}")
        print(f"  Green: {scores[1]:.4f} → {('AI' if scores[1] >= 0.5 else 'Real')}")
        print(f"  Blue:  {scores[2]:.4f} → {('AI' if scores[2] >= 0.5 else 'Real')}")
        
        # Check variance/discrimination
        import numpy as np
        variance = np.var([scores[0], scores[1], scores[2]])
        print(f"  Variance: {variance:.6f} (higher is better discrimination)")
