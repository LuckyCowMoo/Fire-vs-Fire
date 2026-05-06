"""ResNet-50 spider detector classifier.

This module is intentionally self-contained so it can be auto-discovered by
the classifier registry without depending on any other classifier files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sys

try:
    from .base_classifier import BaseClassifier
except Exception:  # pragma: no cover
    import importlib.util

    _base_path = Path(__file__).resolve().parent / "base_classifier.py"
    _spec = importlib.util.spec_from_file_location("classifier_base", str(_base_path))
    if _spec is None or _spec.loader is None:
        raise ImportError("Could not load base_classifier")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


class ResNet50SpiderClassifier(BaseClassifier):
    """ResNet-50 image classifier for spider detection."""

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path)

        self.torch = None
        self.nn = None
        self.models = None
        self.transforms = None
        self.Image = None
        self._SpiderResNet50 = None

        self._preprocess = None
        self._device_info: Dict[str, Any] = {
            "device": "unknown",
            "name": "Not loaded",
            "backend": "N/A",
        }

        if self.model_path is None:
            self.model_path = Path(__file__).resolve().parent / "spider_detector_resnet50.pt"

    def _log(self, message: str) -> None:
        try:
            print(message, file=sys.stderr, flush=True)
        except Exception:
            pass

    def _ensure_imports(self) -> bool:
        if self.torch is not None:
            return True

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from PIL import Image

            self.torch = torch
            self.nn = nn
            self.models = models
            self.transforms = transforms
            self.Image = Image

            class SpiderResNet50(self.nn.Module):
                def __init__(self, models_mod, nn_mod):
                    super().__init__()
                    self.backbone = models_mod.resnet50(weights=None)
                    self.backbone.fc = nn_mod.Identity()
                    self.head = nn_mod.Sequential(
                        nn_mod.Dropout(p=0.2),
                        nn_mod.Linear(2048, 2),
                    )

                def forward(self, x):
                    x = self.backbone(x)
                    return self.head(x)

            self._SpiderResNet50 = SpiderResNet50
            return True
        except Exception as exc:
            self._log(f"[ResNet50Spider] Failed to import libraries: {exc}")
            return False

    def _setup_device(self) -> None:
        try:
            import torch_directml

            self.device = torch_directml.device()
            try:
                device_name = torch_directml.device_name(0)
            except Exception:
                device_name = "DirectML GPU"
            self._device_info = {
                "device": "privateuseone:0",
                "name": device_name,
                "backend": "DirectML",
            }
            self._log(f"[ResNet50Spider] Using GPU: {device_name} (DirectML)")
            return
        except ImportError:
            self._log("[ResNet50Spider] DirectML not available")
        except Exception as exc:
            self._log(f"[ResNet50Spider] DirectML failed: {exc}")

        if self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
            try:
                device_name = self.torch.cuda.get_device_name(0)
            except Exception:
                device_name = "CUDA GPU"
            self._device_info = {
                "device": "cuda",
                "name": device_name,
                "backend": "CUDA",
            }
            self._log(f"[ResNet50Spider] Using GPU: {device_name} (CUDA)")
            return

        self.device = self.torch.device("cpu")
        self._device_info = {
            "device": "cpu",
            "name": "CPU",
            "backend": "CPU",
        }
        self._log("[ResNet50Spider] Using CPU")

    def get_supported_modalities(self) -> Set[str]:
        return {"image"}

    def load_model(self) -> Tuple[bool, Optional[str]]:
        if self._is_loaded:
            return True, None

        if not self._ensure_imports():
            return False, "import_failed"

        if self.model_path is None or not Path(self.model_path).exists():
            return False, f"Model file not found: {self.model_path}"

        try:
            self._setup_device()

            model = self._SpiderResNet50(self.models, self.nn)

            checkpoint = self.torch.load(self.model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state = checkpoint["state_dict"]
                else:
                    state = checkpoint
            else:
                state = checkpoint

            if isinstance(state, dict) and state:
                if all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
                    state = {k.split("module.", 1)[1]: v for k, v in state.items()}
                if all(isinstance(k, str) and k.startswith("backbone.") for k in state.keys()):
                    state = {k.split("backbone.", 1)[1]: v for k, v in state.items()}

            model.load_state_dict(state, strict=False)
            model.eval()
            model = model.to(self.device)

            self.model = model
            self._is_loaded = True

            self._preprocess = self.transforms.Compose([
                self.transforms.Resize((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self._log(f"[ResNet50Spider] Loaded model from: {self.model_path}")
            return True, None
        except Exception as exc:
            self._log(f"[ResNet50Spider] Failed to load model: {exc}")
            return False, "load_failed"

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "image":
            return None, []

        if not self._is_loaded or self._preprocess is None:
            return None, []

        valid_tensors = []
        valid_indices: List[int] = []

        for idx, item in enumerate(inputs):
            try:
                if isinstance(item, self.Image.Image):
                    image = item.convert("RGB")
                else:
                    image = self.Image.open(item).convert("RGB")
                tensor = self._preprocess(image)
                valid_tensors.append(tensor)
                valid_indices.append(idx)
            except Exception:
                continue

        if not valid_tensors:
            return None, []

        batch_tensor = self.torch.stack(valid_tensors).to(self.device)
        return batch_tensor, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not self._is_loaded or batch_tensor is None:
            return []

        try:
            with self.torch.no_grad():
                logits = self.model(batch_tensor)
                probs = self.torch.softmax(logits, dim=1)

            spider_probs = probs[:, 1].detach().cpu().numpy()
            return [float(score) for score in spider_probs]
        except Exception as exc:
            self._log(f"[ResNet50Spider] Inference failed: {exc}")
            return [-1.0] * batch_tensor.shape[0]

    def get_device_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {"device": "unknown", "name": "Not loaded", "backend": "N/A"}
        return self._device_info

    def get_model_name(self) -> str:
        return "ResNet50 Spider"
    
    
    def get_label(self) -> str:
        return "Spider"


if __name__ == "__main__":
    from PIL import Image

    print("Instantiating classifier...")
    c = ResNet50SpiderClassifier()
    print("Using model file:", c.model_path)
    success, err = c.load_model()
    print("Load result:", success, err)
    if success:
        print("Device info:", c.get_device_info())
        sample_image = Path("spider_dataset/spiders/spider_000000.jpg")
        if sample_image.exists():
            print("Opening sample image:", sample_image)
            img = Image.open(sample_image).convert("RGB")
            print("Running inference...")
            scores = c.process_batch([img], "image")
            print("Scores:", scores)
        else:
            print("Sample image not found, skipping inference test.")