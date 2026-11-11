"""FvF Heimdall 02 — 20-Channel AI Image Classifier Plugin for Fire vs Fire.

Naming Convention:
  - Image models: Heimdall (`FvF Heimdall 02`)
  - Video models: Huginn
  - Audio models: Bragi
  - Text models:  Mímir

Self-contained `BaseClassifier` implementation supporting:
  1. GPU-Accelerated Image Preprocessing:
     - Transfers raw `uint8` image tensors directly to the GPU (`non_blocking=True`),
       where float normalization (`/ 255.0`), 1:1 native center-crop slicing (with
       reflect padding for small images), and bicubic antialiased global resizing
       (`torch.nn.functional.interpolate`) execute directly on CUDA Tensor Cores.
  2. Asymmetric Dual-Stream Inference:
     - Global Semantic View (256x256 Bicubic RGB -> ConvNeXt-V2 FCMAE)
     - Native-Grid 20-Channel Forensic View (1:1 un-resized 256x256 pixel crop ->
       pure-Conv2d 20-channel extractor + SRNet unfactored forensic CNN)
  3. Multi-Backend Consumer Hardware Acceleration:
     - NVIDIA CUDA (with automatic BF16/FP16 Tensor Core acceleration & GPU micro-batching)
     - ONNX Runtime Execution Providers (Qualcomm QNN NPU, Intel OpenVINO NPU/iGPU,
       Windows DirectML GPU/NPU, CUDA, CPU) when `fvf_heimdall_02_best.onnx` is present
     - PyTorch DirectML (AMD Radeon / Intel Arc GPUs on Windows)
     - CPU fallback
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .base_classifier import BaseClassifier
except Exception:  # pragma: no cover
    import importlib.util

    _base_path = Path(__file__).resolve().parent / "base_classifier.py"
    _spec = importlib.util.spec_from_file_location("classifier_harness_base", str(_base_path))
    if _spec is None or _spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


class FvfHeimdall02Classifier(BaseClassifier):
    """FvF Heimdall 02 (ConvNeXt-V2 + 20-Channel SRNet Forensic) image classifier."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__(model_path)
        self.torch = None
        self.F = None
        self.np = None
        self.Image = None
        self.ort_session = None

        self._device_info: Dict[str, Any] = {
            "device": "unknown",
            "name": "Not loaded",
            "backend": "N/A",
        }
        self._image_size: int = 256
        self._num_crops: int = 5
        self._ai_class_index: int = 1
        self._use_fp16: bool = False
        self._amp_dtype: Any = None
        self._gpu_micro_batch_size: int = 8

    def _log(self, message: str) -> None:
        self._log_to_stderr(f"[FvF Heimdall 02] {message}")

    def get_supported_modalities(self) -> Set[str]:
        return {"image"}

    def get_model_name(self) -> str:
        return "FvF Heimdall 02"

    def get_label(self) -> str:
        return "AI generated"

    def get_device_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {"device": "unknown", "name": "Not loaded", "backend": "N/A"}
        return self._device_info

    def _ensure_imports(self) -> bool:
        if self.torch is not None:
            return True
        try:
            import numpy as np
            import torch
            import torch.nn.functional as F
            from PIL import Image

            self.torch = torch
            self.F = F
            self.np = np
            self.Image = Image
            return True
        except Exception as exc:
            self._log(f"Failed to import core libraries: {exc}")
            return False

    def _find_checkpoint(self) -> Optional[Path]:
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)

        here = Path(__file__).resolve().parent
        repo_root = here.parents[1]
        candidates = [
            here / "fvf_heimdall_02_int8.onnx",
            here / "fvf_heimdall_02_best.onnx",
            here / "fvf_heimdall_02_best.pt",
            here / "fvf_ares_best.pt",
            repo_root / "checkpoints" / "fvf_heimdall_02_base" / "best_inference_fp16.pt",
            repo_root / "checkpoints" / "fvf_ares" / "best_inference_fp16.pt",
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _setup_device(self) -> None:
        torch = self.torch
        # 1. Prioritize NVIDIA CUDA first
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "NVIDIA CUDA GPU"
            self._use_fp16 = True
            self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._device_info = {"device": "cuda", "name": name, "backend": "CUDA"}
            self._log(f"Using GPU (CUDA): {name} (dtype={self._amp_dtype})")
            return

        # 2. Try Apple Silicon MPS (Metal Performance Shaders / Apple Neural Engine via MPSGraph)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                self.device = torch.device("mps")
                self._use_fp16 = True
                self._amp_dtype = torch.float16
                self._device_info = {"device": "mps", "name": "Apple Silicon (Metal/ANE)", "backend": "MPS"}
                self._log("Using Apple Silicon GPU/ANE (MPS)")
                return
            except Exception:
                pass

        # 3. Try DirectML (AMD / Intel / Windows GPU & NPU)
        try:
            import torch_directml

            self.device = torch_directml.device()
            name = torch_directml.device_name(0)
            self._use_fp16 = False
            self._device_info = {"device": "privateuseone:0", "name": name, "backend": "DirectML"}
            self._log(f"Using GPU/NPU (DirectML): {name}")
            return
        except Exception:
            pass

        # 3. CPU fallback
        self.device = torch.device("cpu")
        self._use_fp16 = False
        self._num_crops = 1
        self._gpu_micro_batch_size = 2
        try:
            import os
            self.torch.set_num_threads(min(8, os.cpu_count() or 4))
        except Exception:
            pass
        self._device_info = {"device": "cpu", "name": "CPU (Multithreaded)", "backend": "CPU"}
        self._log("Using CPU (Multithreaded, 1-crop mode)")

    def _try_load_onnx(self, onnx_path: Path) -> bool:
        try:
            import onnxruntime as ort

            avail = ort.get_available_providers()
            preferred = BaseClassifier.get_preferred_onnx_providers()
            chosen = [p for p in preferred if p in avail] or avail
            opts = BaseClassifier.get_onnx_provider_options(chosen)
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                self.ort_session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=chosen, provider_options=opts)
            except Exception:
                self.ort_session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=chosen)
            active_ep = self.ort_session.get_providers()[0]
            self._device_info = {
                "device": active_ep,
                "name": f"ONNX Runtime ({active_ep})",
                "backend": active_ep.replace("ExecutionProvider", ""),
            }
            self.model_path = onnx_path
            self._is_loaded = True
            self._log(f"Loaded ONNX model {onnx_path.name} via {active_ep}")
            return True
        except Exception as exc:
            self._log(f"ONNX load skipped/failed ({exc}); falling back to PyTorch")
            return False

    def load_model(self) -> Tuple[bool, Optional[str]]:
        if self._is_loaded:
            return True, None
        if not self._ensure_imports():
            return False, "import_failed"

        ckpt_path = self._find_checkpoint()
        if ckpt_path is None:
            return False, "FvF Heimdall 02 checkpoint not found (place fvf_heimdall_02_best.pt in native/classifiers/)"

        onnx_candidate = ckpt_path.with_suffix(".onnx")
        if ckpt_path.suffix.lower() == ".onnx":
            if self._try_load_onnx(ckpt_path):
                return True, None
            return False, "onnx_load_failed"
        elif not self.torch.cuda.is_available() and onnx_candidate.is_file():
            if self._try_load_onnx(onnx_candidate):
                return True, None

        self._setup_device()

        try:
            try:
                from ._heimdall_arch import FvfAresModel
            except Exception:
                try:
                    from _heimdall_arch import FvfAresModel
                except Exception:
                    repo_root = Path(__file__).resolve().parents[2]
                    training_dir = str(repo_root / "training")
                    if training_dir not in sys.path:
                        sys.path.insert(0, training_dir)
                    from model_fvf_ares import FvfAresModel

            ckpt = self.torch.load(ckpt_path, map_location="cpu", weights_only=False)
            backbone_name = ckpt.get("backbone_name", "convnextv2_tiny") if isinstance(ckpt, dict) else "convnextv2_tiny"
            self._image_size = int(ckpt.get("image_size", 256)) if isinstance(ckpt, dict) else 256
            self._ai_class_index = int(ckpt.get("ai_class_index", 1)) if isinstance(ckpt, dict) else 1

            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            has_multicrop = any("multi_crop_aggregator" in k for k in state.keys())
            has_crops = int(ckpt.get("num_crops", 5 if has_multicrop else 1)) if isinstance(ckpt, dict) else (5 if has_multicrop else 1)
            self._num_crops = 1 if self.device.type == "cpu" else has_crops

            model = FvfAresModel(backbone_name=backbone_name, pretrained=False, num_crops=max(1, self._num_crops))
            clean_state = {
                k: (v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
                for k, v in state.items()
            }
            model.load_state_dict(clean_state, strict=False)
            model.eval()
            model.to(self.device)

            # Warmup GPU kernels across both batch sizes (1 and _gpu_micro_batch_size) so cuDNN
            # never stalls during live browser classification!
            if self.device.type == "cuda":
                with self.torch.no_grad(), self.torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                    for bsz in (1, self._gpu_micro_batch_size):
                        dummy_g = self.torch.zeros(bsz, 3, self._image_size, self._image_size, device=self.device)
                        if self._num_crops > 1:
                            dummy_n = self.torch.zeros(bsz, self._num_crops, 3, self._image_size, self._image_size, device=self.device)
                        else:
                            dummy_n = dummy_g
                        _ = model(dummy_g, dummy_n)
                self.torch.cuda.synchronize()

            self.model = model
            self.model_path = ckpt_path
            self._is_loaded = True
            self._log(
                f"Loaded {ckpt_path.name} (backbone={backbone_name}, crops={self._num_crops}, epoch={ckpt.get('epoch', '?')}, "
                f"val_acc={ckpt.get('val_acc', 0.0):.4f})"
            )
            return True, None
        except Exception as exc:
            self._log(f"Failed to load FvF Heimdall 02 model: {exc}")
            return False, f"load_failed: {exc}"

    def _prepare_pair_gpu(self, img: Any) -> Tuple[Any, Any]:
        """Fast GPU-accelerated preprocessing:
        Transfers uint8 image directly to GPU (or target device), normalizes in-place on GPU,
        slices the 5 canonical 1:1 native crops (Center, Corners, Edges) on GPU, and runs
        bicubic antialiased resize on GPU.
        """
        s = self._image_size
        np = self.np
        torch = self.torch
        F = self.F

        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")

        # 1. Fast uint8 numpy array -> HWC uint8 tensor -> GPU transfer (3x smaller than float32)
        arr = np.array(img, dtype=np.uint8, copy=True)
        target_dev = self.device if self.ort_session is None else torch.device("cpu")
        t_u8 = torch.from_numpy(arr).to(target_dev, non_blocking=True)

        # Convert HWC uint8 -> NCHW float32 in [0, 1] directly on GPU
        x = t_u8.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
        _, _, h, w = x.shape

        # 2. Stream A (Global Semantic View): GPU Bicubic Antialiased Interpolation to (s, s)
        if h == s and w == s:
            glob_t = x.squeeze(0)
        else:
            glob_t = F.interpolate(
                x, size=(s, s), mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0.0, 1.0).squeeze(0)

        # 3. Stream B (Native 1:1 Pixel Crops): GPU Tensor Slicing (with reflect/replicate pad if smaller than s)
        if h < s or w < s:
            pad_h = max(0, s - h)
            pad_w = max(0, s - w)
            pad_mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            _, _, h, w = x.shape

        max_y = max(0, h - s)
        max_x = max(0, w - s)
        mid_y = max_y // 2
        mid_x = max_x // 2

        if self._num_crops <= 1:
            crop_t = x[0, :, mid_y : mid_y + s, mid_x : mid_x + s]
            return glob_t.contiguous(), crop_t.contiguous()

        # 5 Canonical Crops: [0: Center, 1: Top-Left Corner, 2: Bottom-Right Corner, 3: Top Edge, 4: Right Edge]
        coords = [
            (mid_y, mid_x),  # 0: Center (Middle)
            (0, 0),          # 1: Top-Left Corner
            (max_y, max_x),  # 2: Bottom-Right Corner
            (0, mid_x),      # 3: Top-Middle Edge
            (mid_y, max_x),  # 4: Middle-Right Edge
        ][: self._num_crops]

        crops = [x[0, :, top : top + s, left : left + s] for top, left in coords]
        crops_t = torch.stack(crops, dim=0)  # (K, 3, s, s)

        return glob_t.contiguous(), crops_t.contiguous()

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "image" or not self._is_loaded:
            return None, []

        global_list = []
        native_list = []
        valid_indices: List[int] = []

        with self.torch.no_grad():
            for idx, item in enumerate(inputs):
                try:
                    if item is None:
                        continue
                    glob_t, crop_t = self._prepare_pair_gpu(item)
                    global_list.append(glob_t)
                    native_list.append(crop_t)
                    valid_indices.append(idx)
                except Exception as exc:
                    self._log(f"Preprocess failed on image {idx}: {exc}")
                    continue

            if not valid_indices:
                return None, []

            glob_batch = self.torch.stack(global_list, dim=0)
            nat_batch = self.torch.stack(native_list, dim=0)

        return {"global_rgb": glob_batch, "native_rgb": nat_batch}, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not self._is_loaded or batch_tensor is None:
            return []

        glob_rgb = batch_tensor["global_rgb"]
        nat_rgb = batch_tensor["native_rgb"]
        n_total = int(glob_rgb.shape[0])

        try:
            if self.ort_session is not None:
                logits_np = self.ort_session.run(
                    None,
                    {
                        "global_rgb": glob_rgb.cpu().numpy(),
                        "native_rgb": nat_rgb.cpu().numpy(),
                    },
                )[0]
                logits = self.torch.from_numpy(logits_np)
                probs = self.torch.softmax(logits, dim=-1)[:, self._ai_class_index]
                return [float(x) for x in probs.tolist()]

            all_probs: List[float] = []
            mb = max(1, self._gpu_micro_batch_size)
            with self.torch.no_grad():
                for start in range(0, n_total, mb):
                    end = min(start + mb, n_total)
                    g_sub = glob_rgb[start:end]
                    n_sub = nat_rgb[start:end]
                    if self._use_fp16 and self.device.type == "cuda":
                        with self.torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype):
                            logits = self.model(g_sub, n_sub)
                    else:
                        logits = self.model(g_sub, n_sub)
                    probs = self.torch.softmax(logits.float(), dim=-1)[:, self._ai_class_index]
                    all_probs.extend(float(x) for x in probs.detach().cpu().tolist())

            return all_probs
        except Exception as exc:
            self._log(f"Inference failed: {exc}")
            return [-1.0] * n_total
