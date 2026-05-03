#!/usr/bin/env python3
"""Check what metadata is in the model checkpoint."""

import torch
from pathlib import Path

model_path = Path(__file__).parent / 'classifiers' / 'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0011_ILL_DO_IT_MYSELF_2.pt'

print(f"[CHECKPOINT] Loading: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"\n[CHECKPOINT] Keys in checkpoint: {list(checkpoint.keys())}")

# Check for class index metadata
for key in ['ai_class_index', 'positive_class_index', 'target_class_index', 'class_index', 'args', 'metadata']:
    if key in checkpoint:
        print(f"[CHECKPOINT] {key}: {checkpoint[key]}")

# Check if there's an args object with metadata
if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
    print(f"\n[CHECKPOINT] args object keys: {checkpoint['args'].keys()}")
    for k, v in checkpoint['args'].items():
        if 'class' in k.lower() or 'index' in k.lower():
            print(f"[CHECKPOINT]   {k}: {v}")
