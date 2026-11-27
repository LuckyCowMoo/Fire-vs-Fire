"""FvF Bellows — Detached Hardware-Adaptive Classifier Trainer & Exporter.

Runs standalone as a background OS process (`python native/bellows_trainer.py --job-dir <path>`)
so training continues smoothly even if the user closes the browser window.

Features:
  - Prevents OS sleep while training (Windows SetThreadExecutionState / macOS caffeinate).
  - Auto-selects best available training hardware: CUDA (BF16/FP16) -> MPS -> DirectML -> CPU.
  - Polls battery/AC power status: if unplugged AND battery < 50%, saves best checkpoint & stops.
  - Supports 3 speed/accuracy tiers for Image (RGB MobileNetV3-Small / EfficientNet-B0 / ConvNeXt-Tiny)
    and Text (ELECTRA-Small / MiniLM-L6 / DistilRoBERTa-Base).
  - Balances user positives + weighted hard negatives + Hugging Face base web negatives with AFPL FP penalty.
  - Writes real-time telemetry (loss/acc curves, ETA, 2x2 confusion matrix) to `status.json`.
  - Exports INT8 ONNX (falling back to FP32 ONNX / PT if needed) + auto-generated BaseClassifier `.py`.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIERS_DIR = REPO_ROOT / "native" / "classifiers"
BELLOWS_CACHE_DIR = REPO_ROOT / "native" / "bellows_cache"
PRETRAINED_CACHE_DIR = BELLOWS_CACHE_DIR / "pretrained"

IMAGE_TIERS: Dict[str, Dict[str, Any]] = {
    "swift": {
        "name": "Swift (MobileNetV3-Small RGB)",
        "timm_name": "mobilenetv3_small_100",
        "img_size": 192,
        "epochs": 10,
        "batch_size": 32,
        "lr": 4e-4,
        "est_gpu_min": 3,
        "est_cpu_min": 28,
        "disk_mb": 450,
    },
    "balanced": {
        "name": "Balanced (EfficientNet-B0 RGB)",
        "timm_name": "efficientnet_b0",
        "img_size": 224,
        "epochs": 12,
        "batch_size": 24,
        "lr": 3e-4,
        "est_gpu_min": 7,
        "est_cpu_min": 75,
        "disk_mb": 520,
    },
    "thorough": {
        "name": "Thorough (ConvNeXt-Tiny RGB)",
        "timm_name": "convnext_tiny",
        "img_size": 224,
        "epochs": 10,
        "batch_size": 16,
        "lr": 1.2e-4,
        "est_gpu_min": 14,
        "est_cpu_min": 165,
        "disk_mb": 720,
    },
}

TEXT_TIERS: Dict[str, Dict[str, Any]] = {
    "swift": {
        "name": "Swift (ELECTRA-Small)",
        "hf_name": "google/electra-small-discriminator",
        "max_length": 256,
        "epochs": 4,
        "batch_size": 24,
        "lr": 5e-5,
        "est_gpu_min": 4,
        "est_cpu_min": 35,
        "disk_mb": 420,
    },
    "balanced": {
        "name": "Balanced (MiniLM-L6)",
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "epochs": 4,
        "batch_size": 24,
        "lr": 4e-5,
        "est_gpu_min": 6,
        "est_cpu_min": 55,
        "disk_mb": 500,
    },
    "thorough": {
        "name": "Thorough (DistilRoBERTa-Base)",
        "hf_name": "distilbert/distilroberta-base",
        "max_length": 256,
        "epochs": 3,
        "batch_size": 16,
        "lr": 2.5e-5,
        "est_gpu_min": 12,
        "est_cpu_min": 140,
        "disk_mb": 820,
    },
}


def get_power_status() -> Dict[str, Any]:
    is_plugged = True
    has_battery = False
    pct: Optional[int] = 100
    charging = False
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_byte),
                    ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            s = SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                is_plugged = s.ACLineStatus == 1
                has_battery = not bool(s.BatteryFlag & 128)
                pct = int(s.BatteryLifePercent) if 0 <= s.BatteryLifePercent <= 100 else None
                charging = bool(s.BatteryFlag & 8)
        elif sys.platform.startswith("linux"):
            pdir = Path("/sys/class/power_supply")
            if pdir.is_dir():
                for sup in pdir.iterdir():
                    tf = sup / "type"
                    if not tf.is_file():
                        continue
                    st = tf.read_text(encoding="utf-8").strip().lower()
                    if st == "mains" and (sup / "online").is_file():
                        is_plugged = (sup / "online").read_text(encoding="utf-8").strip() == "1"
                    elif st == "battery":
                        has_battery = True
                        if (sup / "capacity").is_file():
                            pct = int((sup / "capacity").read_text(encoding="utf-8").strip())
                        if (sup / "status").is_file():
                            charging = (sup / "status").read_text(encoding="utf-8").strip().lower() == "charging"
    except Exception:
        pass
    return {
        "is_plugged_in": is_plugged,
        "has_battery": has_battery,
        "battery_percent": pct,
        "charging": charging,
    }


def _should_abort_low_battery() -> bool:
    ps = get_power_status()
    return bool(
        ps.get("has_battery")
        and not ps.get("is_plugged_in", True)
        and ps.get("battery_percent") is not None
        and int(ps["battery_percent"]) < 50
    )


class _SleepInhibitor:
    """Prevent OS suspend while Bellows training runs."""

    def __init__(self) -> None:
        self._proc = None

    def __enter__(self) -> "_SleepInhibitor":
        try:
            if sys.platform == "win32":
                # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            elif sys.platform == "darwin":
                self._proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except Exception:
            pass
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            if sys.platform == "win32":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            elif self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass


def _write_status(status_path: Path, data: Dict[str, Any]) -> None:
    data["updated_at"] = time.time()
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "custom").lower()).strip("_")
    return s[:40] or "custom"


def _select_training_device(torch_mod):
    if torch_mod.cuda.is_available():
        try:
            dname = torch_mod.cuda.get_device_name(0)
        except Exception:
            dname = "NVIDIA CUDA GPU"
        use_bf16 = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
        return torch_mod.device("cuda"), {"device": "cuda", "name": dname, "backend": "CUDA"}, True, use_bf16
    if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
        return torch_mod.device("mps"), {"device": "mps", "name": "Apple Silicon (MPS)", "backend": "MPS"}, False, False
    try:
        import torch_directml

        dml = torch_directml.device()
        try:
            dname = torch_directml.device_name(0)
        except Exception:
            dname = "DirectML GPU"
        return dml, {"device": "privateuseone:0", "name": dname, "backend": "DirectML"}, False, False
    except Exception:
        pass
    try:
        torch_mod.set_num_threads(max(2, min(8, os.cpu_count() or 4)))
    except Exception:
        pass
    return torch_mod.device("cpu"), {"device": "cpu", "name": "CPU (Multithreaded)", "backend": "CPU"}, False, False


def _compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    total = max(1, tp + fp + tn + fn)
    acc = (tp + tn) / total
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = (2 * prec * tpr) / max(1e-8, prec + tpr)
    return {
        "acc": round(float(acc), 4),
        "tpr": round(float(tpr), 4),
        "fpr": round(float(fpr), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": round(float(threshold), 2),
    }


def _bellows_loss(logits, labels, weights, torch_mod, F_mod):
    """Weighted BCE + mild AFPL false-positive penalty on negatives."""
    bce = F_mod.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch_mod.sigmoid(logits)
    neg_mask = (1.0 - labels)
    fp_excess = torch_mod.clamp(probs - 0.35, min=0.0)
    afpl = 2.0 * neg_mask * (fp_excess ** 2)
    return torch_mod.mean((bce + afpl) * weights)


def _auto_calibrate_threshold(probs: np.ndarray, labels: np.ndarray, user_thresh: float) -> float:
    best_t = float(user_thresh)
    best_score = -1.0
    for t in np.linspace(0.35, 0.90, 23):
        m = _compute_metrics(probs, labels, float(t))
        score = m["f1"] - 0.65 * m["fpr"]
        if score > best_score:
            best_score = score
            best_t = float(t)
    return round(best_t, 2)


def _generate_classifier_python(
    out_py: Path,
    classifier_id: str,
    class_name: str,
    modality: str,
    model_name: str,
    label: str,
    suggested_threshold: float,
    onnx_filename: str,
    pt_filename: str,
    meta_filename: str,
) -> None:
    if modality == "image":
        code = f'''"""Bespoke FvF Bellows Image Classifier: {model_name}."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .base_classifier import BaseClassifier
except Exception:
    import importlib.util

    _base_path = Path(__file__).resolve().parent / "base_classifier.py"
    _spec = importlib.util.spec_from_file_location("bellows_base_classifier", str(_base_path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


class {class_name}(BaseClassifier):
    IS_BELLOWS_MODEL = True
    CLASSIFIER_ID = "{classifier_id}"
    REQUIRED_PACKAGES = ["onnxruntime", "pillow", "numpy"]

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path=model_path)
        self._ort_session = None
        self._torch_model = None
        self._torch = None
        self._meta: Dict[str, Any] = {{}}
        self._img_size: int = 224
        self._device_info: Dict[str, Any] = {{"device": "cpu", "name": "CPU", "backend": "CPU"}}
        self.model_path = model_path or self._find_checkpoint()

    def _find_checkpoint(self) -> Optional[Path]:
        here = Path(__file__).resolve().parent
        for cand in (here / "{onnx_filename}", here / "{pt_filename}"):
            if cand.is_file():
                return cand
        return None

    def get_supported_modalities(self) -> Set[str]:
        return {{"image"}}

    def get_model_name(self) -> str:
        return {json.dumps(model_name)}

    def get_label(self) -> str:
        return {json.dumps(label)}

    def get_suggested_threshold(self) -> Optional[float]:
        return {suggested_threshold:.4f}

    def get_device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def load_model(self) -> Tuple[bool, Optional[str]]:
        here = Path(__file__).resolve().parent
        meta_path = here / "{meta_filename}"
        if meta_path.is_file():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._img_size = int(self._meta.get("img_size", 224))

        onnx_path = here / "{onnx_filename}"
        if onnx_path.is_file():
            try:
                import onnxruntime as ort

                avail = ort.get_available_providers()
                preferred = [ep for ep in BaseClassifier.get_preferred_onnx_providers() if ep in avail] or ["CPUExecutionProvider"]
                opts = BaseClassifier.get_onnx_provider_options(preferred)
                try:
                    self._ort_session = ort.InferenceSession(str(onnx_path), providers=preferred, provider_options=opts)
                except Exception:
                    self._ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                active_ep = self._ort_session.get_providers()[0]
                self._device_info = {{
                    "device": active_ep,
                    "name": f"ONNX Runtime ({{active_ep}})",
                    "backend": active_ep.replace("ExecutionProvider", ""),
                }}
                self.model_path = onnx_path
                self._is_loaded = True
                return True, None
            except Exception as exc:
                self._log_to_stderr(f"[{classifier_id}] ONNX load warning: {{exc}}")

        pt_path = here / "{pt_filename}"
        if pt_path.is_file():
            try:
                import torch

                self._torch = torch
                dev = self._try_cuda_then_directml(torch) or torch.device("cpu")
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                self._torch_model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
                self._torch_model = self._torch_model.to(dev).eval()
                self.device = dev
                self.model_path = pt_path
                self._is_loaded = True
                return True, None
            except Exception as exc:
                return False, f"Failed loading Bellows classifier: {{exc}}"
        return False, "Missing Bellows model weights."

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "image" or not self._is_loaded:
            return None, []
        from PIL import Image, ImageOps

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        arrs, valid = [], []
        for idx, item in enumerate(inputs):
            try:
                im = item if isinstance(item, Image.Image) else Image.open(item)
                im = ImageOps.exif_transpose(im).convert("RGB").resize((self._img_size, self._img_size), Image.Resampling.BILINEAR)
                x = np.asarray(im, dtype=np.float32) / 255.0
                x = np.transpose(x, (2, 0, 1))[None, ...]
                x = (x - mean) / std
                arrs.append(x)
                valid.append(idx)
            except Exception:
                continue
        if not arrs:
            return None, []
        return np.concatenate(arrs, axis=0).astype(np.float32), valid

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if batch_tensor is None:
            return []
        if self._ort_session is not None:
            inp_name = self._ort_session.get_inputs()[0].name
            logits = np.asarray(self._ort_session.run(None, {{inp_name: batch_tensor}})[0]).reshape(-1)
        elif self._torch_model is not None and self._torch is not None:
            with self._torch.no_grad():
                t = self._torch.from_numpy(batch_tensor).to(self.device)
                logits = self._torch_model(t).detach().cpu().numpy().reshape(-1)
        else:
            return []
        logits = np.clip(logits.astype(np.float64), -30.0, 30.0)
        return [float(1.0 / (1.0 + np.exp(-x))) for x in logits]

    def unload_model(self) -> None:
        self._ort_session = None
        self._torch_model = None
        self._is_loaded = False
        gc.collect()
'''
    else:
        code = f'''"""Bespoke FvF Bellows Text Classifier: {model_name}."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from .base_classifier import BaseClassifier
except Exception:
    import importlib.util

    _base_path = Path(__file__).resolve().parent / "base_classifier.py"
    _spec = importlib.util.spec_from_file_location("bellows_base_classifier", str(_base_path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseClassifier = _mod.BaseClassifier


class {class_name}(BaseClassifier):
    IS_BELLOWS_MODEL = True
    CLASSIFIER_ID = "{classifier_id}"
    REQUIRED_PACKAGES = ["onnxruntime", "transformers", "numpy"]

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path=model_path)
        self._ort_session = None
        self._tokenizer = None
        self._meta: Dict[str, Any] = {{}}
        self._max_length: int = 256
        self._device_info: Dict[str, Any] = {{"device": "cpu", "name": "CPU", "backend": "CPU"}}
        self.model_path = model_path or self._find_checkpoint()

    def _find_checkpoint(self) -> Optional[Path]:
        here = Path(__file__).resolve().parent
        cand = here / "{onnx_filename}"
        return cand if cand.is_file() else None

    def get_supported_modalities(self) -> Set[str]:
        return {{"text"}}

    def get_model_name(self) -> str:
        return {json.dumps(model_name)}

    def get_label(self) -> str:
        return {json.dumps(label)}

    def get_suggested_threshold(self) -> Optional[float]:
        return {suggested_threshold:.4f}

    def get_device_info(self) -> Dict[str, Any]:
        return dict(self._device_info)

    def load_model(self) -> Tuple[bool, Optional[str]]:
        here = Path(__file__).resolve().parent
        meta_path = here / "{meta_filename}"
        if not meta_path.is_file():
            return False, "Missing Bellows metadata JSON."
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._max_length = int(self._meta.get("max_length", 256))
        onnx_path = here / "{onnx_filename}"
        if not onnx_path.is_file():
            return False, "Missing Bellows ONNX model."

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            tok_dir = here / (self._meta.get("tokenizer_dir") or "{classifier_id}_tokenizer")
            self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

            avail = ort.get_available_providers()
            preferred = [ep for ep in BaseClassifier.get_preferred_onnx_providers() if ep in avail] or ["CPUExecutionProvider"]
            opts = BaseClassifier.get_onnx_provider_options(preferred)
            try:
                self._ort_session = ort.InferenceSession(str(onnx_path), providers=preferred, provider_options=opts)
            except Exception:
                self._ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            active_ep = self._ort_session.get_providers()[0]
            self._device_info = {{
                "device": active_ep,
                "name": f"ONNX Runtime ({{active_ep}})",
                "backend": active_ep.replace("ExecutionProvider", ""),
            }}
            self.model_path = onnx_path
            self._is_loaded = True
            return True, None
        except Exception as exc:
            return False, f"Failed loading Bellows text classifier: {{exc}}"

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "text" or not self._is_loaded or self._tokenizer is None:
            return None, []
        valid_idx, valid_txt = [], []
        for idx, item in enumerate(inputs):
            if isinstance(item, str) and len(item.strip()) >= 40:
                valid_idx.append(idx)
                valid_txt.append(item.strip())
        if not valid_txt:
            return None, []
        tok = self._tokenizer(valid_txt, padding=True, truncation=True, max_length=self._max_length, return_tensors="np")
        return {{
            "input_ids": tok["input_ids"].astype(np.int64),
            "attention_mask": tok["attention_mask"].astype(np.int64),
        }}, valid_idx

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not batch_tensor or self._ort_session is None:
            return []
        logits = np.asarray(self._ort_session.run(None, batch_tensor)[0]).reshape(-1)
        logits = np.clip(logits.astype(np.float64), -30.0, 30.0)
        return [float(1.0 / (1.0 + np.exp(-x))) for x in logits]

    def unload_model(self) -> None:
        self._ort_session = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()
'''
    out_py.write_text(code, encoding="utf-8")


def _export_and_finalize(
    cfg: Dict[str, Any],
    best_model_state: Dict[str, Any],
    build_model_fn,
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    tokenizer,
    job_dir: Path,
    status: Dict[str, Any],
    status_path: Path,
    extra_meta: Dict[str, Any],
) -> None:
    import onnx
    import onnxruntime as ort
    import torch

    status["phase"] = "exporting"
    status["phase_detail"] = "Exporting quantized ONNX model & creating BaseClassifier..."
    _write_status(status_path, status)

    slug = cfg["slug"]
    classifier_id = f"bellows_{slug}"
    class_name = "Bellows" + "".join(w.capitalize() for w in slug.split("_")) + "Classifier"
    onnx_fp32 = job_dir / f"{classifier_id}_fp32.onnx"
    onnx_final_name = f"{classifier_id}_int8.onnx"
    pt_final_name = f"{classifier_id}_best.pt"
    meta_name = f"{classifier_id}_meta.json"
    onnx_final = CLASSIFIERS_DIR / onnx_final_name
    pt_final = CLASSIFIERS_DIR / pt_final_name
    meta_path = CLASSIFIERS_DIR / meta_name
    out_py = CLASSIFIERS_DIR / f"{classifier_id}_classifier.py"

    cpu_model = build_model_fn().cpu().eval()
    cpu_model.load_state_dict(best_model_state)

    if bool(cfg.get("auto_threshold", False)) and len(val_probs) >= 10:
        final_thresh = _auto_calibrate_threshold(val_probs, val_labels, float(cfg.get("suggested_threshold", 0.75)))
    else:
        final_thresh = float(cfg.get("suggested_threshold", 0.75))

    if cfg["modality"] == "image":
        sz = int(extra_meta["img_size"])
        dummy = torch.randn((2, 3, sz, sz), dtype=torch.float32)
        with torch.no_grad():
            pt_logits = cpu_model(dummy).numpy().reshape(-1)
        torch.onnx.export(
            cpu_model,
            dummy,
            str(onnx_fp32),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

            q = MatMulNBitsQuantizer(model=onnx.load(str(onnx_fp32)), bits=8, block_size=64, is_symmetric=True)
            q.process()
            q.model.save_model_to_file(str(onnx_final))
            sess = ort.InferenceSession(str(onnx_final), providers=["CPUExecutionProvider"])
            ort_out = np.asarray(sess.run(None, {"image": dummy.numpy()})[0]).reshape(-1)
            if float(np.mean(np.abs(pt_logits - ort_out))) > 0.25:
                raise RuntimeError("Quantization drift exceeded 0.25")
        except Exception:
            shutil.copyfile(onnx_fp32, onnx_final)
    else:
        max_len = int(extra_meta["max_length"])
        tok_dir = CLASSIFIERS_DIR / f"{classifier_id}_tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
        extra_meta["tokenizer_dir"] = tok_dir.name

        bt = tokenizer(["Hello world example sample.", "This is a longer sample paragraph used to verify padded dynamic axes during trace export."], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            pt_logits = cpu_model(bt["input_ids"], bt["attention_mask"]).numpy().reshape(-1)
        torch.onnx.export(
            cpu_model,
            (bt["input_ids"], bt["attention_mask"]),
            str(onnx_fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

            q = MatMulNBitsQuantizer(model=onnx.load(str(onnx_fp32)), bits=8, block_size=64, is_symmetric=True, accuracy_level=4)
            q.process()
            q.model.save_model_to_file(str(onnx_final))
            sess = ort.InferenceSession(str(onnx_final), providers=["CPUExecutionProvider"])
            ort_out = np.asarray(sess.run(None, {"input_ids": bt["input_ids"].numpy().astype(np.int64), "attention_mask": bt["attention_mask"].numpy().astype(np.int64)})[0]).reshape(-1)
            if float(np.mean(np.abs(pt_logits - ort_out))) > 0.25:
                raise RuntimeError("Quantization drift exceeded 0.25")
        except Exception:
            shutil.copyfile(onnx_fp32, onnx_final)

    onnx_fp32.unlink(missing_ok=True)

    meta_payload = {
        "classifier_id": classifier_id,
        "name": cfg["model_name"],
        "label": cfg["label"],
        "modality": cfg["modality"],
        "suggested_threshold": final_thresh,
        "tier": cfg["tier"],
        "is_bellows": True,
        **extra_meta,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    _generate_classifier_python(
        out_py=out_py,
        classifier_id=classifier_id,
        class_name=class_name,
        modality=cfg["modality"],
        model_name=cfg["model_name"],
        label=cfg["label"],
        suggested_threshold=final_thresh,
        onnx_filename=onnx_final_name,
        pt_filename=pt_final_name,
        meta_filename=meta_name,
    )

    status["classifier_id"] = classifier_id
    status["suggested_threshold"] = final_thresh
    status["confusion_matrix"] = _compute_metrics(val_probs, val_labels, final_thresh)


def _run_image_training(cfg: Dict[str, Any], job_dir: Path, status: Dict[str, Any], status_path: Path) -> None:
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    tier_cfg = IMAGE_TIERS[cfg.get("tier", "balanced")]
    img_size = int(tier_cfg["img_size"])
    epochs = int(cfg.get("epochs_override") or tier_cfg["epochs"])
    batch_size = int(tier_cfg["batch_size"])
    lr = float(tier_cfg["lr"])
    user_thresh = float(cfg.get("suggested_threshold", 0.75))

    device, dev_info, use_amp, use_bf16 = _select_training_device(torch)
    status.update({"device": dev_info["device"], "device_name": dev_info["name"], "backend": dev_info["backend"]})

    # Assemble positive, hard_negative, and HF base web negative image records
    pos_dir = job_dir / "dataset" / "positives"
    hn_dir = job_dir / "dataset" / "hard_negatives"
    pos_files = [p for p in pos_dir.glob("*") if p.is_file()]
    hn_files = [p for p in hn_dir.glob("*") if p.is_file()]

    base_zip = BELLOWS_CACHE_DIR / "base_datasets" / "base_images_v1.zip"
    extract_dir = BELLOWS_CACHE_DIR / "base_datasets" / "images_unpacked"
    if base_zip.is_file() and not (extract_dir / "manifest.csv").is_file():
        status["phase"] = "unpacking_dataset"
        status["phase_detail"] = "Extracting base web negative images..."
        _write_status(status_path, status)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(base_zip, "r") as zf:
            zf.extractall(extract_dir)

    base_files = [p for p in (extract_dir / "images").glob("*.jpg")] if (extract_dir / "images").is_dir() else []
    rng = random.Random(42)
    rng.shuffle(pos_files)
    rng.shuffle(hn_files)
    rng.shuffle(base_files)

    # Cap negatives relative to positives so epochs finish cleanly in reasonable time
    max_base = min(len(base_files), max(350, min(2000, len(pos_files) * 6)))
    base_files = base_files[:max_base]

    def split_list(items: List[Path], val_frac: float = 0.18, min_val: int = 2) -> Tuple[List[Path], List[Path]]:
        if len(items) <= 2:
            return items, items[:1]
        nv = max(min_val, int(round(len(items) * val_frac)))
        nv = min(nv, max(1, len(items) - 1))
        return items[nv:], items[:nv]

    tr_pos, va_pos = split_list(pos_files, 0.18, min_val=2)
    tr_hn, va_hn = split_list(hn_files, 0.18, min_val=0) if hn_files else ([], [])
    tr_base, va_base = split_list(base_files, 0.18, min_val=20) if base_files else ([], [])

    train_records = [(p, 1.0, 1.0) for p in tr_pos] + [(p, 0.0, 3.0) for p in tr_hn] + [(p, 0.0, 1.0) for p in tr_base]
    val_records = [(p, 1.0, 1.0) for p in va_pos] + [(p, 0.0, 2.0) for p in va_hn] + [(p, 0.0, 1.0) for p in va_base]

    status["dataset_summary"] = {
        "positives": len(pos_files),
        "hard_negatives": len(hn_files),
        "base_negatives": len(base_files),
        "train_count": len(train_records),
        "val_count": len(val_records),
    }

    mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    class ImgDS(Dataset):
        def __init__(self, recs: List[Tuple[Path, float, float]], is_train: bool):
            self.recs = recs
            self.is_train = is_train

        def __len__(self) -> int:
            return len(self.recs)

        def __getitem__(self, idx: int):
            p, lbl, w = self.recs[idx]
            try:
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    if self.is_train and random.random() < 0.5:
                        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    im = im.resize((img_size, img_size), Image.Resampling.BILINEAR)
                    arr = np.transpose(np.asarray(im, dtype=np.float32) / 255.0, (2, 0, 1))
                    arr = (arr - mean_np) / std_np
            except Exception:
                arr = np.zeros((3, img_size, img_size), dtype=np.float32)
            return torch.from_numpy(arr), torch.tensor(lbl, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)

    n_pos_tr = max(1, len(tr_pos))
    n_neg_tr = max(1, len(tr_hn) + len(tr_base))
    sample_weights = [
        (n_neg_tr / n_pos_tr if lbl > 0.5 else (2.5 if w > 1.5 else 1.0))
        for _, lbl, w in train_records
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_records), replacement=True)
    train_loader = DataLoader(ImgDS(train_records, True), batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(ImgDS(val_records, False), batch_size=batch_size, shuffle=False, num_workers=0)

    status["phase"] = "downloading_backbone"
    status["phase_detail"] = f"Loading pretrained RGB backbone ({tier_cfg['name']})..."
    _write_status(status_path, status)

    PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(PRETRAINED_CACHE_DIR))

    def build_model():
        return timm.create_model(tier_cfg["timm_name"], pretrained=True, num_classes=1)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    best_score = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_probs = np.zeros(len(val_records), dtype=np.float32)
    best_labels = np.array([r[1] for r in val_records], dtype=np.float32)

    t_start = time.time()
    global_step = 0
    stop_reason = "completed"
    stop_flag = job_dir / "stop.flag"

    status.update({"state": "running", "phase": "training", "total_epochs": epochs, "total_steps": total_steps})
    _write_status(status_path, status)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss_sum = 0.0
        ep_batches = 0

        for x, y, w in train_loader:
            if stop_flag.exists():
                stop_reason = "stopped_early"
                break
            if global_step % 10 == 0 and _should_abort_low_battery():
                stop_reason = "stopped_low_battery"
                break

            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(x).reshape(-1)
                    loss = _bellows_loss(logits, y, w, torch, F)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).reshape(-1)
                loss = _bellows_loss(logits, y, w, torch, F)
                loss.backward()
                optimizer.step()

            scheduler.step()
            ep_loss_sum += float(loss.detach().cpu().item())
            ep_batches += 1
            global_step += 1

            if global_step % 5 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                rate = global_step / max(0.5, elapsed)
                eta = max(0.0, (total_steps - global_step) / max(1e-4, rate))
                status.update({
                    "epoch": ep,
                    "step": global_step,
                    "progress_pct": round(min(96.0, 100.0 * global_step / total_steps), 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1),
                    "power_status": get_power_status(),
                    "phase_detail": f"Epoch {ep}/{epochs} • Step {global_step}/{total_steps} • loss={ep_loss_sum / max(1, ep_batches):.4f}",
                })
                _write_status(status_path, status)

        if stop_reason != "completed" and ep_batches == 0:
            break

        # Validation pass
        model.eval()
        val_logits_list, val_y_list, val_losses = [], [], []
        with torch.no_grad():
            for vx, vy, vw in val_loader:
                vx = vx.to(device)
                vy_d = vy.to(device)
                vw_d = vw.to(device)
                vl = model(vx).reshape(-1)
                val_losses.append(float(_bellows_loss(vl, vy_d, vw_d, torch, F).cpu().item()))
                val_logits_list.append(torch.sigmoid(vl).cpu().numpy())
                val_y_list.append(vy.numpy())

        v_probs = np.concatenate(val_logits_list) if val_logits_list else np.array([0.5])
        v_labels = np.concatenate(val_y_list) if val_y_list else np.array([0.0])
        cm = _compute_metrics(v_probs, v_labels, user_thresh)
        tr_loss = round(ep_loss_sum / max(1, ep_batches), 4)
        va_loss = round(float(np.mean(val_losses)) if val_losses else tr_loss, 4)

        status.setdefault("history", []).append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_acc": cm["acc"],
            "val_fpr": cm["fpr"],
            "val_tpr": cm["tpr"],
            "val_f1": cm["f1"],
        })
        status["confusion_matrix"] = cm

        sel = cm["f1"] - 0.5 * cm["fpr"]
        if sel >= best_score or ep == 1:
            best_score = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = v_probs
            best_labels = v_labels
            status["best_metrics"] = cm

        _write_status(status_path, status)

        if stop_reason != "completed":
            break

    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def build_export_model():
        return timm.create_model(tier_cfg["timm_name"], pretrained=False, num_classes=1)

    _export_and_finalize(
        cfg=cfg,
        best_model_state=best_state,
        build_model_fn=build_export_model,
        val_probs=best_probs,
        val_labels=best_labels,
        tokenizer=None,
        job_dir=job_dir,
        status=status,
        status_path=status_path,
        extra_meta={"img_size": img_size, "backbone": tier_cfg["timm_name"]},
    )

    status["state"] = stop_reason
    status["phase"] = "finished"
    status["progress_pct"] = 100.0
    status["eta_seconds"] = 0.0
    status["elapsed_seconds"] = round(time.time() - t_start, 1)
    status["power_status"] = get_power_status()
    if stop_reason == "stopped_low_battery":
        status["phase_detail"] = "Unplugged below 50% battery — saved and exported best checkpoint cleanly."
    elif stop_reason == "stopped_early":
        status["phase_detail"] = "Stopped early — saved and exported best checkpoint cleanly."
    else:
        status["phase_detail"] = "Training complete! Bespoke classifier installed and ready."
    _write_status(status_path, status)


def _run_text_training(cfg: Dict[str, Any], job_dir: Path, status: Dict[str, Any], status_path: Path) -> None:
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModel, AutoTokenizer

    tier_cfg = TEXT_TIERS[cfg.get("tier", "balanced")]
    max_len = int(tier_cfg["max_length"])
    epochs = int(cfg.get("epochs_override") or tier_cfg["epochs"])
    batch_size = int(tier_cfg["batch_size"])
    lr = float(tier_cfg["lr"])
    user_thresh = float(cfg.get("suggested_threshold", 0.75))

    device, dev_info, use_amp, use_bf16 = _select_training_device(torch)
    status.update({"device": dev_info["device"], "device_name": dev_info["name"], "backend": dev_info["backend"]})

    pos_path = job_dir / "dataset" / "text_positives.jsonl"
    hn_path = job_dir / "dataset" / "text_hard_negatives.jsonl"
    pos_texts = [json.loads(ln)["text"] for ln in pos_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if pos_path.is_file() else []
    hn_texts = [json.loads(ln)["text"] for ln in hn_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if hn_path.is_file() else []

    base_parquet = BELLOWS_CACHE_DIR / "base_datasets" / "base_text_v1.parquet"
    base_texts: List[str] = []
    if base_parquet.is_file():
        df_b = pd.read_parquet(base_parquet)
        base_texts = df_b["text"].dropna().astype(str).tolist()

    rng = random.Random(42)
    rng.shuffle(pos_texts)
    rng.shuffle(hn_texts)
    rng.shuffle(base_texts)
    base_texts = base_texts[: min(len(base_texts), max(400, min(2000, len(pos_texts) * 6)))]

    def split_t(items: List[str], val_frac: float = 0.18, min_val: int = 2) -> Tuple[List[str], List[str]]:
        if len(items) <= 2:
            return items, items[:1]
        nv = min(max(1, len(items) - 1), max(min_val, int(round(len(items) * val_frac))))
        return items[nv:], items[:nv]

    tr_pos, va_pos = split_t(pos_texts, 0.18, 2)
    tr_hn, va_hn = split_t(hn_texts, 0.18, 0) if hn_texts else ([], [])
    tr_base, va_base = split_t(base_texts, 0.18, 20) if base_texts else ([], [])

    train_records = [(t, 1.0, 1.0) for t in tr_pos] + [(t, 0.0, 3.0) for t in tr_hn] + [(t, 0.0, 1.0) for t in tr_base]
    val_records = [(t, 1.0, 1.0) for t in va_pos] + [(t, 0.0, 2.0) for t in va_hn] + [(t, 0.0, 1.0) for t in va_base]

    status["dataset_summary"] = {
        "positives": len(pos_texts),
        "hard_negatives": len(hn_texts),
        "base_negatives": len(base_texts),
        "train_count": len(train_records),
        "val_count": len(val_records),
    }

    status["phase"] = "downloading_backbone"
    status["phase_detail"] = f"Loading pretrained language encoder ({tier_cfg['name']})..."
    _write_status(status_path, status)

    PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hf_name = tier_cfg["hf_name"]
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=str(PRETRAINED_CACHE_DIR))

    class BellowsTextNet(nn.Module):
        def __init__(self, pretrained: bool = True):
            super().__init__()
            if pretrained:
                self.encoder = AutoModel.from_pretrained(hf_name, cache_dir=str(PRETRAINED_CACHE_DIR), attn_implementation="eager")
            else:
                from transformers import AutoConfig

                cfg_hf = AutoConfig.from_pretrained(hf_name, cache_dir=str(PRETRAINED_CACHE_DIR))
                cfg_hf._attn_implementation = "eager"
                self.encoder = AutoModel.from_config(cfg_hf)
            hid = self.encoder.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hid, 1))

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            return self.head(pooled).squeeze(-1)

    class TextDS(Dataset):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            return self.recs[idx]

    def collate(batch):
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        tok = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return tok["input_ids"], tok["attention_mask"], labels, weights

    n_pos_tr = max(1, len(tr_pos))
    n_neg_tr = max(1, len(tr_hn) + len(tr_base))
    sample_weights = [(n_neg_tr / n_pos_tr if l > 0.5 else (2.5 if w > 1.5 else 1.0)) for _, l, w in train_records]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_records), replacement=True)
    train_loader = DataLoader(TextDS(train_records), batch_size=batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(TextDS(val_records), batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = BellowsTextNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    best_score = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_probs = np.zeros(len(val_records), dtype=np.float32)
    best_labels = np.array([r[1] for r in val_records], dtype=np.float32)

    t_start = time.time()
    global_step = 0
    stop_reason = "completed"
    stop_flag = job_dir / "stop.flag"

    status.update({"state": "running", "phase": "training", "total_epochs": epochs, "total_steps": total_steps})
    _write_status(status_path, status)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss_sum, ep_batches = 0.0, 0
        for ids, mask, y, w in train_loader:
            if stop_flag.exists():
                stop_reason = "stopped_early"
                break
            if global_step % 8 == 0 and _should_abort_low_battery():
                stop_reason = "stopped_low_battery"
                break

            ids, mask, y, w = ids.to(device), mask.to(device), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(ids, mask)
                    loss = _bellows_loss(logits, y, w, torch, F)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(ids, mask)
                loss = _bellows_loss(logits, y, w, torch, F)
                loss.backward()
                optimizer.step()
            scheduler.step()
            ep_loss_sum += float(loss.detach().cpu().item())
            ep_batches += 1
            global_step += 1

            if global_step % 4 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                rate = global_step / max(0.5, elapsed)
                eta = max(0.0, (total_steps - global_step) / max(1e-4, rate))
                status.update({
                    "epoch": ep,
                    "step": global_step,
                    "progress_pct": round(min(96.0, 100.0 * global_step / total_steps), 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(eta, 1),
                    "power_status": get_power_status(),
                    "phase_detail": f"Epoch {ep}/{epochs} • Step {global_step}/{total_steps} • loss={ep_loss_sum / max(1, ep_batches):.4f}",
                })
                _write_status(status_path, status)

        if stop_reason != "completed" and ep_batches == 0:
            break

        model.eval()
        val_probs_l, val_y_l, val_losses = [], [], []
        with torch.no_grad():
            for ids, mask, vy, vw in val_loader:
                vl = model(ids.to(device), mask.to(device))
                val_losses.append(float(_bellows_loss(vl, vy.to(device), vw.to(device), torch, F).cpu().item()))
                val_probs_l.append(torch.sigmoid(vl).cpu().numpy())
                val_y_l.append(vy.numpy())

        v_probs = np.concatenate(val_probs_l) if val_probs_l else np.array([0.5])
        v_labels = np.concatenate(val_y_l) if val_y_l else np.array([0.0])
        cm = _compute_metrics(v_probs, v_labels, user_thresh)
        tr_loss = round(ep_loss_sum / max(1, ep_batches), 4)
        va_loss = round(float(np.mean(val_losses)) if val_losses else tr_loss, 4)

        status.setdefault("history", []).append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_acc": cm["acc"],
            "val_fpr": cm["fpr"],
            "val_tpr": cm["tpr"],
            "val_f1": cm["f1"],
        })
        status["confusion_matrix"] = cm

        sel = cm["f1"] - 0.5 * cm["fpr"]
        if sel >= best_score or ep == 1:
            best_score = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = v_probs
            best_labels = v_labels
            status["best_metrics"] = cm
        _write_status(status_path, status)

        if stop_reason != "completed":
            break

    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _export_and_finalize(
        cfg=cfg,
        best_model_state=best_state,
        build_model_fn=lambda: BellowsTextNet(pretrained=False),
        val_probs=best_probs,
        val_labels=best_labels,
        tokenizer=tokenizer,
        job_dir=job_dir,
        status=status,
        status_path=status_path,
        extra_meta={"max_length": max_len, "backbone": hf_name},
    )

    status["state"] = stop_reason
    status["phase"] = "finished"
    status["progress_pct"] = 100.0
    status["eta_seconds"] = 0.0
    status["elapsed_seconds"] = round(time.time() - t_start, 1)
    status["power_status"] = get_power_status()
    status["phase_detail"] = (
        "Unplugged below 50% battery — saved best checkpoint cleanly."
        if stop_reason == "stopped_low_battery"
        else ("Stopped early — saved best checkpoint cleanly." if stop_reason == "stopped_early" else "Training complete! Bespoke classifier installed and ready.")
    )
    _write_status(status_path, status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir).resolve()
    cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    status_path = job_dir / "status.json"
    status: Dict[str, Any] = {
        "job_id": cfg["job_id"],
        "pid": os.getpid(),
        "state": "initializing",
        "phase": "initializing",
        "phase_detail": "Starting Bellows training worker...",
        "modality": cfg["modality"],
        "model_name": cfg["model_name"],
        "label": cfg["label"],
        "tier": cfg["tier"],
        "progress_pct": 0.0,
        "epoch": 0,
        "total_epochs": 1,
        "step": 0,
        "total_steps": 1,
        "history": [],
        "power_status": get_power_status(),
    }
    _write_status(status_path, status)

    with _SleepInhibitor():
        try:
            if cfg["modality"] == "image":
                _run_image_training(cfg, job_dir, status, status_path)
            else:
                _run_text_training(cfg, job_dir, status, status_path)
            return 0
        except Exception as exc:
            import traceback

            status["state"] = "failed"
            status["phase"] = "failed"
            status["error"] = f"{exc}\n{traceback.format_exc()}"
            status["phase_detail"] = f"Training failed: {exc}"
            _write_status(status_path, status)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
