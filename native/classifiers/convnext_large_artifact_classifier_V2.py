"""ConvNeXt-Large classifier with artifact/residual branch.

Design goals:
- Keep the BaseClassifier plug-and-play interface (no changes to orchestrator).
- Use ImageNet-pretrained ConvNeXt-Large as the RGB backbone.
- Add a lightweight artifact branch that consumes fixed residual features:
  - Multi-scale high-pass filter responses (Laplacian/Sobel + SRM-like kernels)
  - Optional wavelet subbands (if pywavelets available)
  - Low-resolution FFT magnitude (downsampled) as an additional channel

The model fuses RGB + artifact embeddings late (feature concatenation) and
predicts 2-class logits (AI vs Real). Score returned is P(AI).

Notes:
- CPU-only friendly by default; will use DirectML/CUDA if available.
- Training is expected to freeze most of ConvNeXt-Large and fine-tune late.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sys


try:
    # Matches the existing harness pattern when loaded as a package.
    from .base_classifier import BaseClassifier
except Exception:  # pragma: no cover
    # Robust fallback when this folder isn't an importable package.
    import importlib.util

    _base_path = Path(__file__).resolve().parent / 'base_classifier.py'
    _spec = importlib.util.spec_from_file_location('classifier_harness_base', str(_base_path))
    if _spec is None or _spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


class ConvNeXtLargeArtifactV2Classifier(BaseClassifier):
    """ConvNeXt-Large + artifact branch image classifier (V2 preprocessing)."""

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path)

        self.torch = None
        self.nn = None
        self.F = None
        self.models = None
        self.transforms = None
        self.Image = None

        self.np = None
        self.pywt = None

        self._device_info: Dict[str, Any] = {
            'device': 'unknown',
            'name': 'Not loaded',
            'backend': 'N/A',
        }

        self._rgb_to_tensor = None
        self._rgb_normalize = None

        self._artifact_kernels = None
        self._artifact_in_channels: int = 0
        self._ai_class_index: int = 0

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
            import torch.nn.functional as F
            from torchvision import models, transforms
            from PIL import Image
            import numpy as np

            self.torch = torch
            self.nn = nn
            self.F = F
            self.models = models
            self.transforms = transforms
            self.Image = Image
            self.np = np

            try:
                import pywt
                self.pywt = pywt
            except Exception:
                self.pywt = None

            return True
        except Exception as exc:
            self._log(f"[ConvNeXtLargeArtifact] Failed to import libraries: {exc}")
            return False

    def _setup_device(self) -> None:
        # Prefer DirectML on Windows if present, else CUDA, else CPU.
        try:
            import torch_directml

            self.device = torch_directml.device()
            device_name = torch_directml.device_name(0)
            self._device_info = {
                'device': 'privateuseone:0',
                'name': device_name,
                'backend': 'DirectML',
            }
            self._log(f"[ConvNeXtLargeArtifact] ✓ Using GPU: {device_name} (DirectML)")
            return
        except ImportError:
            pass
        except Exception as exc:
            self._log(f"[ConvNeXtLargeArtifact] DirectML failed: {exc}")

        if self.torch.cuda.is_available():
            self.device = self.torch.device('cuda')
            try:
                device_name = self.torch.cuda.get_device_name(0)
            except Exception:
                device_name = 'CUDA GPU'
            self._device_info = {
                'device': 'cuda',
                'name': device_name,
                'backend': 'CUDA',
            }
            self._log(f"[ConvNeXtLargeArtifact] ✓ Using GPU: {device_name} (CUDA)")
        else:
            self.device = self.torch.device('cpu')
            self._device_info = {
                'device': 'cpu',
                'name': 'CPU',
                'backend': 'CPU',
            }
            self._log("[ConvNeXtLargeArtifact] ⚠ Using CPU")

    def _find_model_path(self) -> Optional[Path]:
        if self.model_path and self.model_path.exists():
            return self.model_path

        repo_root = Path(__file__).resolve().parents[2]
        imm_dir = repo_root / 'Immage Models'

        classifier_dir = Path(__file__).resolve().parent
        candidates = [
            classifier_dir / 'immage_classifier_V3_web_only_epoch0004_epoch0013.pt',
            classifier_dir / 'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0011_ILL_DO_IT_MYSELF_2.pt',
            classifier_dir / 'immage_classifier_V3_ConvNeXtLarge_Artifact_epoch0011_ILL_DO_IT_MYSELF_1.pt',
            classifier_dir / 'immage_classifier_V3-2_ConvNeXtLarge_Artifact_epoch0005.pt',
            imm_dir / 'immage_classifier_V3-2_ConvNeXtLarge_Artifact_epoch0005.pt',
            imm_dir / 'image_classifier_convnext_large_artifact_v2.pt',
            repo_root / 'training' / 'models' / 'image_classifier.pt',
        ]

        for p in candidates:
            if p.exists():
                return p

        # Fallback: newest .pt with 'v3-2' in name
        artifact_pts: List[Path] = []
        try:
            for p in classifier_dir.glob('*.pt'):
                name = p.name.lower()
                if 'v3-2' in name or 'v3_2' in name:
                    artifact_pts.append(p)
        except Exception:
            artifact_pts = []

        if artifact_pts:
            try:
                artifact_pts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            except Exception:
                artifact_pts.sort()
            return artifact_pts[0]

        return None

    def get_supported_modalities(self) -> Set[str]:
        return {'image'}

    def get_model_name(self) -> str:
        return 'ConvNeXt-Large-Artifact-V2'

    def get_device_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {'device': 'unknown', 'name': 'Not loaded', 'backend': 'N/A'}
        return self._device_info

    def _build_artifact_kernels(self) -> Any:
        """Create fixed 2D kernels for residual extraction.

        Returns torch.Tensor of shape (K, 1, kH, kW)
        """
        # Laplacian
        lap = self.torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=self.torch.float32)
        # Sobel
        sobel_x = self.torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=self.torch.float32)
        sobel_y = self.torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=self.torch.float32)
        # Two SRM-like high-pass kernels (compact)
        srm1 = self.torch.tensor(
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            dtype=self.torch.float32,
        )
        srm2 = self.torch.tensor(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            dtype=self.torch.float32,
        )

        kernels_3 = [lap, sobel_x, sobel_y]
        kernels_5 = [srm1, srm2]

        k3 = self.torch.stack(kernels_3, dim=0)[:, None, :, :]  # (3,1,3,3)
        k5 = self.torch.stack(kernels_5, dim=0)[:, None, :, :]  # (2,1,5,5)
        return k3, k5

    def load_model(self) -> Tuple[bool, Optional[str]]:
        if self._is_loaded:
            return True, None

        if not self._ensure_imports():
            return False, 'import_failed'

        self._setup_device()

        path = self._find_model_path()
        if not path:
            return False, 'model_not_found'

        try:
            # Preprocessing: rescale shorter side to 257px (1.15 * 224) with
            # BICUBIC then centre-crop to 224x224. This matches training exactly
            # and equalises resampling artifacts across AI and real images.
            self._rgb_to_tensor = self.transforms.Compose([
                self.transforms.Resize(257, interpolation=self.transforms.InterpolationMode.BICUBIC),
                self.transforms.CenterCrop(224),
                self.transforms.ToTensor(),
            ])

            try:
                weights = self.models.ConvNeXt_Large_Weights.DEFAULT
                mean = weights.meta.get('mean', [0.485, 0.456, 0.406])
                std = weights.meta.get('std', [0.229, 0.224, 0.225])
            except Exception:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            self._rgb_normalize = self.transforms.Normalize(mean=mean, std=std)

            # RGB backbone: ConvNeXt-Large (pretrained).
            try:
                rgb_backbone = self.models.convnext_large(weights=self.models.ConvNeXt_Large_Weights.DEFAULT)
            except Exception:
                rgb_backbone = self.models.convnext_large(weights=None)

            # Artifact feature channels fixed by our preprocess recipe.
            # Channels: multi-scale filters (5 kernels x2 scales=10) + wavelet (3) + fft (1) = 14.
            artifact_in_ch = 14

            nn = self.nn
            torch = self.torch

            class ConvNeXtArtifactModel(nn.Module):
                def __init__(self, rgb_model, artifact_in_ch: int):
                    super().__init__()
                    self.rgb_model = rgb_model
                    self.rgb_norm = self.rgb_model.classifier[0]
                    rgb_feat_dim = self.rgb_model.classifier[-1].in_features

                    self.artifact_branch = nn.Sequential(
                        nn.Conv2d(artifact_in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.GELU(),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.GELU(),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.GELU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                    )

                    self.head = nn.Sequential(
                        nn.Linear(rgb_feat_dim + 256, 512),
                        nn.GELU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(512, 2),
                    )

                def forward(self, rgb, artifact):
                    x = self.rgb_model.features(rgb)
                    x = self.rgb_model.avgpool(x)
                    x = self.rgb_norm(x)
                    x = x.flatten(1)

                    a = self.artifact_branch(artifact).flatten(1)
                    z = torch.cat([x, a], dim=1)
                    return self.head(z)

            model = ConvNeXtArtifactModel(rgb_backbone, artifact_in_ch=artifact_in_ch)

            # Load checkpoint (accept either raw state_dict or wrapper)
            try:
                checkpoint = self.torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = self.torch.load(path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            else:
                state = checkpoint

            ai_class_index = 1  # default: label=1 is AI in the training manifest
            import os
            env_idx = os.environ.get('FIRE_CONVNEXT_AI_CLASS_INDEX')
            if env_idx is not None:
                try:
                    ai_class_index = int(env_idx)
                except Exception:
                    pass
            if isinstance(checkpoint, dict):
                for key in ('ai_class_index', 'positive_class_index', 'target_class_index'):
                    if key in checkpoint:
                        try:
                            ai_class_index = int(checkpoint[key])
                            break
                        except Exception:
                            pass
                args_obj = checkpoint.get('args')
                if isinstance(args_obj, dict):
                    for key in ('ai_class_index', 'positive_class_index', 'target_class_index'):
                        if key in args_obj:
                            try:
                                ai_class_index = int(args_obj[key])
                                break
                            except Exception:
                                pass
            self._ai_class_index = ai_class_index
            self._log(f"[ConvNeXtLargeArtifactV2] AI class index: {self._ai_class_index}")

            if isinstance(state, dict):
                # Allow optional prefixes
                cleaned = {}
                for k, v in state.items():
                    if isinstance(k, str) and k.startswith('model.'):
                        cleaned[k.split('model.', 1)[1]] = v
                    else:
                        cleaned[k] = v
                state = cleaned

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                self._log(
                    f"[ConvNeXtLargeArtifact] load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}"
                )
            model.eval()
            model = model.to(self.device)

            self.model = model
            self.model_path = path
            self._is_loaded = True

            # Fixed kernels for preprocess
            self._artifact_kernels = self._build_artifact_kernels()
            self._artifact_in_channels = artifact_in_ch

            return True, None
        except Exception as exc:
            self._log(f"[ConvNeXtLargeArtifact] Failed to load model: {exc}")
            return False, 'load_failed'

    def _compute_fft_channel(self, gray_224: Any) -> Any:
        """Compute low-res FFT magnitude channel from grayscale tensor.

        Args:
            gray_224: torch.Tensor (1, 1, 224, 224) in [0,1]

        Returns:
            torch.Tensor (1, 1, 224, 224) normalized per-image.
        """
        # Downsample for speed
        gray_112 = self.F.avg_pool2d(gray_224, kernel_size=2, stride=2)
        fft = self.torch.fft.fft2(gray_112)
        mag = self.torch.abs(fft)
        mag = self.torch.log1p(mag)

        # Normalize
        mag = mag - mag.amin(dim=(-2, -1), keepdim=True)
        denom = mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mag = mag / denom

        mag_224 = self.F.interpolate(mag, size=(224, 224), mode='bilinear', align_corners=False)
        return mag_224

    def _compute_wavelet_channels(self, gray_np_224) -> Any:
        """Compute wavelet subbands (LH, HL, HH) at 1 level and upsample to 224.

        Returns torch.Tensor (1,3,224,224). If pywt unavailable, returns zeros.
        """
        if self.pywt is None:
            return self.torch.zeros((1, 3, 224, 224), dtype=self.torch.float32)

        try:
            coeffs2 = self.pywt.dwt2(gray_np_224, 'haar')
            _, (lh, hl, hh) = coeffs2
            w = self.np.stack([lh, hl, hh], axis=0).astype('float32')  # (3,112,112)

            # Normalize each channel to [0,1]
            w_min = w.reshape(3, -1).min(axis=1)[:, None, None]
            w_max = w.reshape(3, -1).max(axis=1)[:, None, None]
            denom = (w_max - w_min)
            denom[denom < 1e-6] = 1.0
            w = (w - w_min) / denom

            wt = self.torch.from_numpy(w)[None, :, :, :]  # (1,3,112,112)
            wt = self.F.interpolate(wt, size=(224, 224), mode='bilinear', align_corners=False)
            return wt
        except Exception:
            return self.torch.zeros((1, 3, 224, 224), dtype=self.torch.float32)

    def _artifact_from_rgb_tensor(self, rgb_224: Any) -> Any:
        """Create artifact tensor from RGB tensor.

        Args:
            rgb_224: torch.Tensor (3,224,224) in [0,1]

        Returns:
            torch.Tensor (C,224,224)
        """
        # Grayscale (1,1,224,224)
        gray = (0.299 * rgb_224[0] + 0.587 * rgb_224[1] + 0.114 * rgb_224[2]).unsqueeze(0).unsqueeze(0)

        k3, k5 = self._artifact_kernels
        k3 = k3.to(rgb_224.device)
        k5 = k5.to(rgb_224.device)

        # Scale 1 (224)
        f3 = self.F.conv2d(gray, k3, padding=1)
        f5 = self.F.conv2d(gray, k5, padding=2)
        feats_224 = self.torch.cat([f3, f5], dim=1)  # (1,5,224,224)

        # Scale 2 (112 then upsample)
        gray_112 = self.F.avg_pool2d(gray, kernel_size=2, stride=2)
        f3_s = self.F.conv2d(gray_112, k3, padding=1)
        f5_s = self.F.conv2d(gray_112, k5, padding=2)
        feats_112 = self.torch.cat([f3_s, f5_s], dim=1)  # (1,5,112,112)
        feats_112 = self.F.interpolate(feats_112, size=(224, 224), mode='bilinear', align_corners=False)

        fft_ch = self._compute_fft_channel(gray)

        # Wavelets operate in numpy; use CPU tensor to numpy
        gray_np = gray.squeeze(0).squeeze(0).detach().cpu().numpy().astype('float32')
        wave = self._compute_wavelet_channels(gray_np)

        artifact = self.torch.cat([feats_224, feats_112, wave.to(feats_224.device), fft_ch], dim=1)  # (1,14,224,224)

        # Per-channel standardization (per-image)
        mean = artifact.mean(dim=(-2, -1), keepdim=True)
        std = artifact.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        artifact = (artifact - mean) / std

        return artifact.squeeze(0)

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != 'image':
            self._log(f"[ConvNeXtLargeArtifact] Unsupported modality: {modality}")
            return None, []

        if not self._is_loaded:
            return None, []

        rgb_tensors = []
        artifact_tensors = []
        valid_indices: List[int] = []

        for idx, img in enumerate(inputs):
            try:
                if getattr(img, 'mode', None) != 'RGB':
                    img = img.convert('RGB')

                rgb = self._rgb_to_tensor(img)  # (3,224,224) in [0,1]
                artifact = self._artifact_from_rgb_tensor(rgb)
                rgb_norm = self._rgb_normalize(rgb)

                rgb_tensors.append(rgb_norm)
                artifact_tensors.append(artifact)
                valid_indices.append(idx)
            except Exception as exc:
                self._log(f"[ConvNeXtLargeArtifact] Preprocess failed for image {idx}: {exc}")
                continue

        if not valid_indices:
            return None, []

        rgb_batch = self.torch.stack(rgb_tensors, dim=0).to(self.device)
        artifact_batch = self.torch.stack(artifact_tensors, dim=0).to(self.device)

        return {'rgb': rgb_batch, 'artifact': artifact_batch}, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not self._is_loaded or batch_tensor is None:
            return []

        try:
            rgb = batch_tensor['rgb']
            artifact = batch_tensor['artifact']

            with self.torch.no_grad():
                logits = self.model(rgb, artifact)
                probs = self.torch.softmax(logits, dim=1)

            probs_np = probs.detach().cpu().numpy()
            ai_probs = probs_np[:, self._ai_class_index]
            self._log(
                f"[ConvNeXtLargeArtifactV2] Batch stats: n={len(ai_probs)}, ai_idx={self._ai_class_index}, "
                f"ai[min={float(ai_probs.min()):.4f}, max={float(ai_probs.max()):.4f}, mean={float(ai_probs.mean()):.4f}]"
            )
            return [float(x) for x in ai_probs]
        except Exception as exc:
            self._log(f"[ConvNeXtLargeArtifact] Inference failed: {exc}")
            n = 0
            try:
                n = int(batch_tensor['rgb'].shape[0])
            except Exception:
                pass
            return [-1.0] * n
