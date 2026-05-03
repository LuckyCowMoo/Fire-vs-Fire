"""
Grab the most recent incoming_debug image (or simulate a small real image)
and show exactly what the artifact branch sees.
"""
import sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, 'native')
from classifiers.convnext_large_artifact_classifier import ConvNeXtLargeArtifactClassifier
import torch

clf = ConvNeXtLargeArtifactClassifier()
clf.load_model()

def score_and_show(img, label):
    bt, _ = clf.preprocess_batch([img], 'image')
    with torch.no_grad():
        p = torch.softmax(clf.model(bt['rgb'], bt['artifact']), dim=1).cpu().numpy()
    print(f"{label:45s}  size={img.size}  AI={p[0,0]:.4f}  Real={p[0,1]:.4f}")

# Check if we have any saved incoming debug images
debug_dir = os.path.join('native', 'incoming_debug')
saved = sorted(glob.glob(os.path.join(debug_dir, '*'))) if os.path.isdir(debug_dir) else []
if saved:
    for p in saved[-5:]:
        try:
            img = Image.open(p).convert('RGB')
            score_and_show(img, os.path.basename(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
else:
    print("No incoming_debug images found - simulating small images")

print()
print("--- Size sensitivity test ---")
# Make a realistic photo-like image at various sizes
base = Image.new('RGB', (800, 600))
px = np.array(base)
# Add some texture
rng = np.random.RandomState(42)
px = rng.randint(80, 180, (600, 800, 3), dtype=np.uint8)
# Add smooth gradient on top
for i in range(600):
    px[i, :, 0] = np.clip(px[i, :, 0] + i // 4, 0, 255)
base = Image.fromarray(px, 'RGB')

for size in [(800, 600), (400, 300), (242, 208), (150, 150), (100, 100)]:
    img = base.resize(size, Image.LANCZOS)
    score_and_show(img, f"real-photo resized to {size}")

print()
print("--- PNG round-trip at each size ---")
import io
for size in [(800, 600), (400, 300), (242, 208), (150, 150)]:
    img = base.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img2 = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
    score_and_show(img2, f"PNG round-trip {size}")
