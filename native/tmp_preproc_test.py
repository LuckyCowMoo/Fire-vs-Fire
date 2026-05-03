import sys, io
import numpy as np
from PIL import Image
sys.path.insert(0, 'native')
from classifiers.convnext_large_artifact_classifier import ConvNeXtLargeArtifactClassifier
import torch

clf = ConvNeXtLargeArtifactClassifier()
clf.load_model()

def score(img):
    bt, _ = clf.preprocess_batch([img], 'image')
    with torch.no_grad():
        p = torch.softmax(clf.model(bt['rgb'], bt['artifact']), dim=1).cpu().numpy()
    return p[0,0], p[0,1]

# 1. Direct PIL (like training reads from disk)
arr = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
img_orig = Image.fromarray(arr, 'RGB')
ai, real = score(img_orig)
print(f"Direct PIL noise:        AI={ai:.4f} Real={real:.4f}")

# 2. JPEG round-trip quality=1.0 (browser default imageCaptureQuality)
buf = io.BytesIO()
img_orig.save(buf, format='JPEG', quality=95)
ai, real = score(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
print(f"JPEG q=1.0 (q95):        AI={ai:.4f} Real={real:.4f}")

# 3. JPEG round-trip quality=0.5
buf = io.BytesIO()
img_orig.save(buf, format='JPEG', quality=50)
ai, real = score(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
print(f"JPEG q=0.5 (q50):        AI={ai:.4f} Real={real:.4f}")

# 4. PNG lossless
buf = io.BytesIO()
img_orig.save(buf, format='PNG')
ai, real = score(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
print(f"PNG lossless:            AI={ai:.4f} Real={real:.4f}")

# 5. Simulate canvas resize: browser draws full image onto 224x224 canvas first
img_resized = img_orig.resize((224, 224), Image.LANCZOS)
buf = io.BytesIO()
img_resized.save(buf, format='JPEG', quality=95)
ai, real = score(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
print(f"Canvas 224x224 JPEG q95: AI={ai:.4f} Real={real:.4f}")

# 6. Smooth gradient (more photo-like, less noise)
g = np.tile(np.linspace(0, 255, 600, dtype=np.uint8), (400, 1))
img_grad = Image.fromarray(np.stack([g, g[::-1], g], axis=-1), 'RGB')
ai, real = score(img_grad)
print(f"Smooth gradient direct:  AI={ai:.4f} Real={real:.4f}")

buf = io.BytesIO()
img_grad.save(buf, format='JPEG', quality=95)
ai, real = score(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
print(f"Smooth gradient JPEG q95:AI={ai:.4f} Real={real:.4f}")

# 7. Check training image size - training uses Resize((224,224)) then ToTensor
# Browser sends naturalWidth x naturalHeight canvas, then classifier resizes to 224x224
# Are there any differences in resize method?
print("\n--- Resize method comparison ---")
img_large = Image.fromarray(np.random.randint(0,256,(1024,1024,3),dtype=np.uint8),'RGB')
# Training path: PIL Resize (BICUBIC by default in torchvision)
from torchvision import transforms
train_resize = transforms.Resize((224, 224))
img_train = train_resize(img_large)
ai, real = score(img_train)
print(f"torchvision Resize (train path): AI={ai:.4f} Real={real:.4f}")

# Inference path: same torchvision Resize in classifier
ai, real = score(img_large)
print(f"Direct large PIL (infer path):   AI={ai:.4f} Real={real:.4f}")
