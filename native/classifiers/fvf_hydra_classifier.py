"""FvF Image Hydra — forensic-channel real vs AI classifier (BaseClassifier plugin).

Loads FvfHydra weights trained in the separate Hydra Mk 1 repo. Genre folders are
not used at inference; only the binary head runs on the 19-channel stack.

Set environment variable FVF_HYDRA_ROOT to the Hydra project root (contains src/
and configs/). If unset, common install paths are tried.

Checkpoint: pass model_path, or place last.pt under
  <FVF_HYDRA_ROOT>/checkpoints/hydra/last.pt
Weights may also embed the training config (preferred).
"""

from __future__ import annotations

import os
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


# Default channel/crop settings (match configs/default.yaml in Hydra Mk 1).
_DEFAULT_CHANNELS_CFG: Dict[str, Any] = {
    "ycbcr": True,
    "residuals_native": True,
    "residuals_half": True,
    "wavelet": True,
    "fft": True,
    "prnu": True,
    "lbp": True,
    "wavelet_name": "db4",
    "fft_log": True,
    "prnu_denoiser": "wavelet",
    "prnu_wavelet_levels": 4,
    "lbp_radius": 1,
    "lbp_neighbors": 8,
    "lbp_method": "uniform",
    "per_channel_standardize": True,
}

_DEFAULT_CROP_CFG: Dict[str, Any] = {
    "inference_default": 5,
    "adaptive": [
        {"max_min_dim": 319, "count": 1},
        {"max_min_dim": 511, "count": 3},
        {"max_min_dim": None, "count": 5},
    ],
    "three_crop_corners": ["center", "top_left", "bottom_right"],
}


def _resolve_hydra_root() -> Optional[Path]:
    env = os.environ.get("FVF_HYDRA_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "src" / "fvf_hydra").is_dir():
            return p
    here = Path(__file__).resolve()
    candidates = [
        Path(r"C:\Stuff\coding\FvF Img hydra Mk 1"),
        here.parents[3] / "FvF Img hydra Mk 1",
        here.parents[4] / "FvF Img hydra Mk 1",
    ]
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if (p / "src" / "fvf_hydra").is_dir():
            return p
    return None


def _import_hydra_modules(hydra_root: Path):
    src = str(hydra_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from fvf_hydra.channels import ChannelPipeline, channel_layout
    from fvf_hydra.data.crops import CropPolicy, take_crops
    from fvf_hydra.models import FvfHydra

    return ChannelPipeline, channel_layout, CropPolicy, take_crops, FvfHydra


class FvfHydraClassifier(BaseClassifier):
    """MobileNetV4-Hydra binary classifier on 19 forensic channels."""

    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(model_path)
        self.torch = None
        self._device_info: Dict[str, Any] = {
            "device": "unknown",
            "name": "Not loaded",
            "backend": "N/A",
        }
        self._hydra_root: Optional[Path] = None
        self._cfg: Dict[str, Any] = {}
        self._pipeline = None
        self._crop_policy = None
        self._take_crops = None
        self._batch_row_groups: List[List[int]] = []
        self._ai_class_index = 1
        self._inference_crops = int(os.environ.get("FVF_HYDRA_CROPS", "5"))
        self._score_agg = os.environ.get("FVF_HYDRA_AGG", "max").strip().lower()

    def _log(self, message: str) -> None:
        self._log_to_stderr(f"[FvfHydra] {message}")

    def _ensure_imports(self) -> bool:
        if self.torch is not None:
            return True
        try:
            import torch

            self.torch = torch
            return True
        except Exception as exc:
            self._log(f"Failed to import torch: {exc}")
            return False

    def _setup_device(self) -> None:
        # Prefer CUDA when available (Hydra training target); else base helper.
        if self.torch.cuda.is_available():
            self.device = self.torch.device("cuda")
            try:
                name = self.torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA GPU"
            self._device_info = {"device": "cuda", "name": name, "backend": "CUDA"}
            self._log(f"Using GPU (CUDA): {name}")
            return
        dev = self._try_cuda_then_directml(self.torch)
        if dev is not None:
            self.device = dev
            if hasattr(self, "_device_info") and self._device_info.get("device") != "unknown":
                pass
            else:
                self._device_info = {"device": str(dev), "name": str(dev), "backend": "CPU/DML"}
        else:
            self.device = self.torch.device("cpu")
            self._device_info = {"device": "cpu", "name": "CPU", "backend": "CPU"}

    def _find_checkpoint(self) -> Optional[Path]:
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)
        root = self._hydra_root
        if root is None:
            return None
        # Default: FVF-trained weights (work on FVF-style data). Set FVF_HYDRA_CHECKPOINT=webscrape3
        # only after web-scrape training validates (current webscrape3 run collapsed ~50% acc).
        prefer = os.environ.get("FVF_HYDRA_CHECKPOINT", "fvf").strip().lower()
        if prefer in ("webscrape3", "web_scrape", "web-scrape"):
            rels = (
                "checkpoints/hydra_webscrape/best.pt",
                "checkpoints/hydra_webscrape/last.pt",
                "checkpoints/hydra_webscrape3_v2/best.pt",
                "checkpoints/hydra_webscrape3/best.pt",
                "checkpoints/hydra/last.pt",
            )
        else:
            rels = (
                "checkpoints/hydra/last.pt",
                "checkpoints/hydra/epoch_001.pt",
                "checkpoints/hydra_webscrape3/last.pt",
            )
        for rel in rels:
            p = root / rel
            if p.is_file():
                return p
        return None

    def _load_config(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(ckpt.get("config"), dict):
            return ckpt["config"]
        root = self._hydra_root
        if root is not None:
            yaml_path = root / "configs" / "default.yaml"
            if yaml_path.is_file():
                try:
                    import yaml

                    with yaml_path.open(encoding="utf-8") as f:
                        return yaml.safe_load(f)
                except Exception as exc:
                    self._log(f"Could not load {yaml_path}: {exc}")
        return {
            "channels": _DEFAULT_CHANNELS_CFG,
            "crops": _DEFAULT_CROP_CFG,
            "input": {"crop_size": 256},
            "model": {
                "backbone": "mobilenetv4_hybrid_medium",
                "binary_hidden": 256,
                "dropout": 0.2,
                "seed_ycbcr_from_rgb": True,
                "init_extra_channels": "kaiming",
            },
            "genres": {"ids": []},
        }

    def get_supported_modalities(self) -> Set[str]:
        return {"image"}

    def get_model_name(self) -> str:
        return "FvF-Image-Hydra"

    def get_label(self) -> str:
        return "AI generated"

    def get_device_info(self) -> Dict[str, Any]:
        if not self._is_loaded:
            return {"device": "unknown", "name": "Not loaded", "backend": "N/A"}
        return self._device_info

    def load_model(self) -> Tuple[bool, Optional[str]]:
        if self._is_loaded:
            return True, None
        if not self._ensure_imports():
            return False, "torch not available"
        self._hydra_root = _resolve_hydra_root()
        if self._hydra_root is None:
            return (
                False,
                "FVF_HYDRA_ROOT not set and Hydra Mk 1 src/ not found (set env to project root)",
            )
        try:
            (
                ChannelPipeline,
                channel_layout,
                CropPolicy,
                take_crops_fn,
                FvfHydra,
            ) = _import_hydra_modules(self._hydra_root)
        except Exception as exc:
            return False, f"Could not import fvf_hydra from {self._hydra_root}: {exc}"

        ckpt_path = self._find_checkpoint()
        if ckpt_path is None:
            return False, "Hydra checkpoint not found (set model_path or train last.pt)"

        try:
            ckpt = self.torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return False, f"Failed to load checkpoint: {exc}"

        self._cfg = self._load_config(ckpt)
        ch_cfg = self._cfg.get("channels", _DEFAULT_CHANNELS_CFG)
        self._pipeline = ChannelPipeline(channels_cfg=ch_cfg)
        self._take_crops = take_crops_fn

        crop_cfg = self._cfg.get("crops", _DEFAULT_CROP_CFG)
        input_cfg = self._cfg.get("input", {})
        crop_size = int(input_cfg.get("crop_size", 256))
        adaptive_table = [
            (entry.get("max_min_dim"), int(entry["count"]))
            for entry in crop_cfg.get("adaptive", _DEFAULT_CROP_CFG["adaptive"])
        ]
        self._crop_policy = CropPolicy(
            crop_size=crop_size,
            inference_default=int(crop_cfg.get("inference_default", 5)),
            three_crop_corners=tuple(
                crop_cfg.get("three_crop_corners", _DEFAULT_CROP_CFG["three_crop_corners"])
            ),
            adaptive_table=tuple(adaptive_table),
        )

        _, group_slices = channel_layout(ch_cfg)
        num_ch = len(channel_layout(ch_cfg)[0])
        rgb_sl = group_slices.get("rgb")
        rgb_stem_slice = (
            (rgb_sl.start, rgb_sl.stop) if rgb_sl is not None and rgb_sl.start > 0 else None
        )
        mcfg = self._cfg.get("model", {})
        genre_ids = list(self._cfg.get("genres", {}).get("ids", []))
        n_genres = max(2, len(genre_ids) or 4)
        self.model = FvfHydra(
            num_input_channels=num_ch,
            num_genres=n_genres,
            backbone_name=mcfg.get("backbone", "mobilenetv4_hybrid_medium"),
            pretrained=False,
            seed_ycbcr_from_rgb=bool(mcfg.get("seed_ycbcr_from_rgb", True)),
            init_extra_channels=mcfg.get("init_extra_channels", "kaiming"),
            genre_hidden=int(mcfg.get("genre_hidden", 256)),
            binary_hidden=int(mcfg.get("binary_hidden", 256)),
            dropout=float(mcfg.get("dropout", 0.2)),
            aggregation_residual=bool(mcfg.get("aggregation_residual", True)),
            rgb_stem_slice=rgb_stem_slice,
        )
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)

        self._setup_device()
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        self._log(f"Loaded {ckpt_path.name} from {ckpt_path.parent} (epoch {ckpt.get('epoch', '?')})")
        return True, None

    def _pil_to_rgb_tensor(self, img) -> Any:
        import numpy as np

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return self.torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def preprocess_batch(self, inputs: List[Any], modality: str) -> Tuple[Any, List[int]]:
        if modality != "image" or not self._is_loaded:
            return None, []

        tensors: List[Any] = []
        valid_indices: List[int] = []
        self._batch_row_groups = []

        for idx, img in enumerate(inputs):
            try:
                if getattr(img, "mode", None) != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if min(w, h) < 64:
                    continue
                crops = self._take_crops(
                    img,
                    self._crop_policy,
                    requested=self._inference_crops,
                    training_random_one=False,
                )
                row_idxs: List[int] = []
                for _spec, crop in crops:
                    rgb = self._pil_to_rgb_tensor(crop)
                    ch = self._pipeline(rgb)
                    if ch.dim() == 4:
                        ch = ch.squeeze(0)
                    tensors.append(ch)
                    row_idxs.append(len(tensors) - 1)
                if row_idxs:
                    self._batch_row_groups.append(row_idxs)
                    valid_indices.append(idx)
            except Exception as exc:
                self._log(f"Preprocess failed image {idx}: {exc}")
                continue

        if not tensors:
            return None, []

        batch = self.torch.stack(tensors, dim=0).to(self.device)
        return batch, valid_indices

    def classify_batch(self, batch_tensor: Any) -> List[float]:
        if not self._is_loaded or batch_tensor is None:
            return []
        try:
            with self.torch.no_grad():
                out = self.model(batch_tensor)
                probs = self.torch.softmax(out.binary_logits, dim=-1)[:, self._ai_class_index]
            probs_cpu = probs.detach().cpu()
            scores: List[float] = []
            for row_idxs in self._batch_row_groups:
                if not row_idxs:
                    scores.append(-1.0)
                    continue
                p = probs_cpu[row_idxs]
                if self._score_agg == "mean":
                    scores.append(float(p.mean().item()))
                else:
                    scores.append(float(p.max().item()))
            return scores
        except Exception as exc:
            self._log(f"Inference failed: {exc}")
            return [-1.0] * len(self._batch_row_groups)