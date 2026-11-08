"""Self-contained architecture definition for FvF Heimdall 02.
Bakes in the 20-channel forensic extractor and multi-crop fusion backbone
so native/classifiers/ runs completely standalone without external training dependencies.
"""

"""Pure-PyTorch 20-Channel Forensic Feature Extractor for Fire vs Fire.

All 20 channels are computed using vectorized tensor operations and fixed
`nn.Conv2d` filter banks (no CPU `pywt`, `skimage`, or non-exportable `fft2`
calls). This allows:
  1. Sub-millisecond batch execution on CUDA, DirectML, NPU (ONNX), and CPU.
  2. Full spatial alignment across all 20 channels (every channel is in (x, y)
     image coordinates).
  3. Clean export to ONNX for Qualcomm QNN, Intel OpenVINO, and DirectML NPUs.

Channel Layout (20 Channels Total):
  [0..2]   YCbCr color channels (Y, Cb, Cr)
  [3..7]   Native-scale spatial residuals (Laplacian, Sobel-X, Sobel-Y, SRM-KV, SRM-Square)
  [8..12]  Half-scale spatial residuals (same 5 filters at 1/2 resolution, upsampled)
  [13..15] Daubechies-4 (db4) 2D wavelet detail subbands (LH, HL, HH via fixed stride-2 Conv2d)
  [16]     PRNU / Camera sensor noise residual (multi-scale Wiener-like high-pass residual)
  [17]     Local micro-texture contrast (vectorized 8-neighbor LBP / directional variance)
  [18]     Localized 8x8 DCT Axial High-Frequency Spectral Energy (horizontal/vertical grid artifacts)
  [19]     Localized 8x8 DCT Diagonal/Nyquist Spectral Energy (checkerboard upsampling artifacts)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNEL_NAMES: Tuple[str, ...] = (
    "ycbcr_Y",
    "ycbcr_Cb",
    "ycbcr_Cr",
    "res_native_laplacian",
    "res_native_sobel_x",
    "res_native_sobel_y",
    "res_native_srm_kv",
    "res_native_srm_sq5",
    "res_half_laplacian",
    "res_half_sobel_x",
    "res_half_sobel_y",
    "res_half_srm_kv",
    "res_half_srm_sq5",
    "wavelet_db4_LH",
    "wavelet_db4_HL",
    "wavelet_db4_HH",
    "prnu_noise_residual",
    "lbp_microtexture",
    "dct8_axial_hf_energy",
    "dct8_diag_checker_energy",
)


def _build_srm_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 3x3 (Laplacian, Sobel-X, Sobel-Y) and 5x5 (SRM-KV, SRM-Square) kernels."""
    lap = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    ) / 4.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
    ) / 4.0

    srm_kv = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 2.0, -4.0, 2.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ) / 4.0

    srm_sq5 = torch.tensor(
        [
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-2.0, 8.0, -12.0, 8.0, -2.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-1.0, 2.0, -2.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    ) / 12.0

    k3 = torch.stack([lap, sobel_x, sobel_y], dim=0).unsqueeze(1)  # (3, 1, 3, 3)
    k5 = torch.stack([srm_kv, srm_sq5], dim=0).unsqueeze(1)  # (2, 1, 5, 5)
    return k3, k5


def _build_db4_wavelet_kernels() -> torch.Tensor:
    """Build exact 2D Daubechies-4 (db4) detail filters (LH, HL, HH) of shape (3, 1, 8, 8)."""
    # Orthogonal Daubechies-4 (8-tap) decomposition low-pass filter h
    h = torch.tensor(
        [
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909309,
            -0.027983769416859854,
            0.6308807679298589,
            0.7148465705529157,
            0.2303778133088965,
        ],
        dtype=torch.float32,
    )
    # Quadrature mirror high-pass filter: g[n] = (-1)^n * h[7 - n]
    idx = torch.arange(8, dtype=torch.float32)
    signs = torch.pow(-1.0, idx)
    g = signs * torch.flip(h, dims=[0])

    # 2D outer products: LH (low row, high col), HL (high row, low col), HH (high row, high col)
    lh = torch.outer(h, g)
    hl = torch.outer(g, h)
    hh = torch.outer(g, g)
    return torch.stack([lh, hl, hh], dim=0).unsqueeze(1)  # (3, 1, 8, 8)


def _build_lbp_neighbor_kernels() -> torch.Tensor:
    """Build 8-neighbor difference kernels of shape (8, 1, 3, 3)."""
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    kernels = []
    for dr, dc in offsets:
        k = torch.zeros((3, 3), dtype=torch.float32)
        k[1, 1] = -1.0
        k[1 + dr, 1 + dc] = 1.0
        kernels.append(k)
    return torch.stack(kernels, dim=0).unsqueeze(1)  # (8, 1, 3, 3)


def _build_dct8_hf_kernels() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 8x8 2D DCT-II basis kernels for axial high-freq vs diagonal/checkerboard high-freq."""
    axial_kernels: List[torch.Tensor] = []
    diag_kernels: List[torch.Tensor] = []

    coords = torch.arange(8, dtype=torch.float32)
    basis_1d = []
    for u in range(8):
        alpha = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        vec = alpha * torch.cos((2.0 * coords + 1.0) * u * math.pi / 16.0)
        basis_1d.append(vec)

    for u in range(8):
        for v in range(8):
            k2d = torch.outer(basis_1d[u], basis_1d[v])
            # Axial high-frequency modes: strong horizontal or vertical periodicity (grid lines)
            if (u >= 4 and v <= 1) or (v >= 4 and u <= 1):
                axial_kernels.append(k2d)
            # Diagonal & Nyquist high-frequency modes: checkerboard / upsampling artifacts
            elif u >= 4 and v >= 4:
                diag_kernels.append(k2d)

    k_axial = torch.stack(axial_kernels, dim=0).unsqueeze(1)  # (16, 1, 8, 8)
    k_diag = torch.stack(diag_kernels, dim=0).unsqueeze(1)  # (16, 1, 8, 8)
    return k_axial, k_diag


class ForensicChannelExtractor20(nn.Module):
    """Extracts 20 forensic channels from an RGB tensor in [0, 1] using pure Conv2d ops."""

    def __init__(self, standardize_forensic: bool = True) -> None:
        super().__init__()
        self.standardize_forensic = standardize_forensic
        self.num_channels = 20

        k3, k5 = _build_srm_kernels()
        k_wave = _build_db4_wavelet_kernels()
        k_lbp = _build_lbp_neighbor_kernels()
        k_dct_axial, k_dct_diag = _build_dct8_hf_kernels()

        # Register as non-trainable buffers so they move with .to(device) and export to ONNX
        self.register_buffer("k3", k3, persistent=False)
        self.register_buffer("k5", k5, persistent=False)
        self.register_buffer("k_wave", k_wave, persistent=False)
        self.register_buffer("k_lbp", k_lbp, persistent=False)
        self.register_buffer("k_dct_axial", k_dct_axial, persistent=False)
        self.register_buffer("k_dct_diag", k_dct_diag, persistent=False)

        # Bit weights for soft-LBP pattern encoding (normalized to [0, 1])
        lbp_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32) / 255.0
        self.register_buffer("lbp_weights", lbp_weights.view(1, 8, 1, 1), persistent=False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute 20-channel forensic stack.

        Args:
            rgb: Tensor of shape (B, 3, H, W) or (3, H, W) with values in [0, 1].

        Returns:
            Tensor of shape (B, 20, H, W).
        """
        squeeze_batch = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            squeeze_batch = True

        h, w = int(rgb.shape[-2]), int(rgb.shape[-1])
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        # 1. YCbCr color channels (3 channels: 0, 1, 2)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.500 * b
        cr = 0.500 * r - 0.418688 * g - 0.081312 * b
        ycbcr = torch.cat([y, cb, cr], dim=1)  # (B, 3, H, W)

        # 2. Native-scale SRM & edge residuals (5 channels: 3..7)
        f3_native = F.conv2d(y, self.k3, padding=1)
        f5_native = F.conv2d(y, self.k5, padding=2)
        res_native = torch.cat([f3_native, f5_native], dim=1)  # (B, 5, H, W)

        # 3. Half-scale SRM & edge residuals (5 channels: 8..12)
        y_half = F.avg_pool2d(y, kernel_size=2, stride=2)
        f3_half = F.conv2d(y_half, self.k3, padding=1)
        f5_half = F.conv2d(y_half, self.k5, padding=2)
        res_half = torch.cat([f3_half, f5_half], dim=1)
        res_half = F.interpolate(res_half, size=(h, w), mode="bilinear", align_corners=False)

        # 4. Daubechies-4 (db4) 2D wavelet subbands LH, HL, HH (3 channels: 13..15)
        y_pad_wave = F.pad(y, (3, 3, 3, 3), mode="reflect")
        wave_sub = F.conv2d(y_pad_wave, self.k_wave, stride=2)
        wave = F.interpolate(wave_sub, size=(h, w), mode="bilinear", align_corners=False)

        # 5. PRNU / Sensor Noise Residual (1 channel: 16)
        # Combines high-order SRM residual with local edge-masked Wiener shrinkage
        edge_mag = (f3_native[:, 1:2].abs() + f3_native[:, 2:3].abs()).clamp_min(1e-5)
        edge_gate = 1.0 / (1.0 + 15.0 * edge_mag)  # Suppress strong object edges, keep flat-region sensor noise
        prnu = f5_native[:, 1:2] * edge_gate  # (B, 1, H, W)

        # 6. Differentiable 8-Neighbor LBP / Micro-Texture Contrast (1 channel: 17)
        lbp_diffs = F.conv2d(y, self.k_lbp, padding=1)  # (B, 8, H, W)
        # Soft threshold around local contrast + bit-weighted texture code
        lbp_bits = torch.sigmoid(lbp_diffs * 64.0)
        lbp = (lbp_bits * self.lbp_weights).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 7. Localized 8x8 DCT High-Frequency Spectral Energy (2 channels: 18, 19)
        y_pad_dct = F.pad(y, (2, 2, 2, 2), mode="reflect")
        dct_ax = F.conv2d(y_pad_dct, self.k_dct_axial, stride=2)  # (B, 16, H/2, W/2)
        dct_dg = F.conv2d(y_pad_dct, self.k_dct_diag, stride=2)  # (B, 16, H/2, W/2)

        energy_axial = torch.log1p(dct_ax.abs().mean(dim=1, keepdim=True) * 16.0)
        energy_diag = torch.log1p(dct_dg.abs().mean(dim=1, keepdim=True) * 16.0)
        dct_channels = F.interpolate(
            torch.cat([energy_axial, energy_diag], dim=1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )  # (B, 2, H, W)

        # Stack forensic channels (17 channels) and standardize per-image per-channel
        forensic_17 = torch.cat(
            [res_native, res_half, wave, prnu, lbp, dct_channels], dim=1
        )  # (B, 17, H, W)

        if self.standardize_forensic:
            mean = forensic_17.mean(dim=(-2, -1), keepdim=True)
            std = forensic_17.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            forensic_17 = ((forensic_17 - mean) / std).clamp(-6.0, 6.0)

        # Center YCbCr to roughly [-1, 1]
        y_norm = (y - 0.5) * 2.0
        cb_norm = cb * 2.0
        cr_norm = cr * 2.0
        ycbcr_norm = torch.cat([y_norm, cb_norm, cr_norm], dim=1)

        out = torch.cat([ycbcr_norm, forensic_17], dim=1)  # (B, 20, H, W)
        if squeeze_batch:
            return out.squeeze(0)
        return out


"""FvF-Ares: Asymmetric Dual-Resolution 20-Channel Forensic Classifier.

Architecture overview:
  1. Stream A (Global Semantic Branch):
     - Input: Full image resized to (256, 256), 3-channel RGB (ImageNet normalized).
     - Backbone: Pretrained `convnextv2_tiny.fcmae_ft_in22k_in1k` (or `convnextv2_nano`,
       `fastvit_sa24`, etc. via `timm`).
     - Captures global structural coherence, anatomy, lighting consistency, and
       high-level semantic anomalies.

  2. Stream B (Native-Grid 20-Channel Forensic Branch):
     - Input: Un-resized 1:1 pixel crop(s) of size (256, 256), passed through
       `ForensicChannelExtractor20` to produce 20 spatially aligned channels:
         [0..2]   YCbCr color channels
         [3..7]   Native-scale SRM & edge residuals (Laplacian, Sobel-X/Y, SRM-KV, SRM-Sq5)
         [8..12]  Half-scale SRM & edge residuals
         [13..15] Daubechies-4 (db4) wavelet detail subbands (LH, HL, HH)
         [16]     PRNU / sensor noise residual
         [17]     Differentiable 8-neighbor LBP / micro-texture contrast
         [18..19] Localized 8x8 DCT high-frequency axial & diagonal/checkerboard energy
     - Backbone: SRNet-inspired Unfactored Residual Forensic CNN:
         * Stage 1 operates at full 256x256 resolution (`stride=1`, unfactored 3x3
           convolutions) so cross-channel noise correlations across all 20 channels
           are captured before any spatial downsampling.
         * Stages 2-4 extract hierarchical forensic patterns with Squeeze-and-Excitation
           (SE) channel attention and residual connections.
         * Statistical Moment Pooling (`Mean + Std + Max`) captures both average
           response, spatial noise variance (energy), and localized peak anomalies.

  3. Zero-Gated Cross-Modality Fusion Head + Temperature Calibration:
     - Uses a learnable gate `alpha` (initialized to 0.0) on the cross-interaction term
       so pretrained semantic features are 100% preserved at step 0 of training.
     - Includes a non-trainable `temperature` buffer `T` for post-training probability
       calibration so the Fire vs Fire GUI threshold slider (0%-100%) behaves smoothly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ForensicChannelExtractor20 defined above in this file


class SqueezeExcite(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention for forensic feature reweighting."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(16, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ForensicResidualBlock(nn.Module):
    """Unfactored 3x3 residual block with SE attention (avoids depthwise bottleneck loss)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcite(out_ch, reduction=4)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)


class SRNetForensicBranch20(nn.Module):
    """20-Channel Forensic CNN with stride-1 native-grid stem and Statistical Moment Pooling."""

    def __init__(self, in_channels: int = 20, out_dim: int = 768) -> None:
        super().__init__()
        # Stage 1: NO downsampling (stride=1) to preserve 1:1 pixel-grid correlations
        # across all 20 forensic channels before any spatial pooling occurs.
        self.stem_native = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            SqueezeExcite(64, reduction=4),
        )

        # Stage 2..4: Hierarchical unfactored residual stages (256 -> 128 -> 64 -> 32 -> 16)
        self.stage2 = ForensicResidualBlock(64, 96, stride=2)
        self.stage3 = ForensicResidualBlock(96, 160, stride=2)
        self.stage4 = ForensicResidualBlock(160, 256, stride=2)
        self.stage5 = ForensicResidualBlock(256, 256, stride=2)

        # Statistical Moment Pooling concatenates [Mean, Std, Max] (256 * 3 = 768)
        self.proj = nn.Sequential(
            nn.Linear(256 * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _forward_stages(self, x20: torch.Tensor) -> torch.Tensor:
        x = self.stem_native(x20)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x20: torch.Tensor) -> torch.Tensor:
        x = self._forward_stages(x20)

        # Statistical Moment Pooling: Mean + Spatial Std (Noise Energy) + Max (Peak Artifact)
        feat_mean = x.mean(dim=(-2, -1))
        feat_std = x.var(dim=(-2, -1), unbiased=False).add(1e-5).sqrt()
        feat_max = x.amax(dim=(-2, -1))

        moments = torch.cat([feat_mean, feat_std, feat_max], dim=1)  # (B, 768)
        return self.proj(moments)


BACKBONE_ALIASES: Dict[str, str] = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
    "fastvit_sa24": "fastvit_sa24.apple_dist_in1k",
    "caformer_s18": "caformer_s18.sail_in22k_ft_in1k",
}


class MultiCropForensicAggregator(nn.Module):
    """Aggregates K native-grid forensic crop embeddings (Center, Corners, Edges) into one representation.

    Computes:
      1. Role-aware Attention Pooling (learns to ignore flat/sky corners and weight textured regions).
      2. Center-vs-Periphery Residual (`Center - mean(Corners, Edges)`), capturing optical lens falloff
         and vignetting vs uniform synthetic AI generation.
      3. Cross-Crop Feature Dispersion (`Std` across the K crops) to detect regional inconsistencies.
      4. Peak Anomaly Pooling (`Max` across the K crops) to catch localized AI artifacts in any single crop.
    """

    def __init__(self, forensic_dim: int = 768, num_crops: int = 5) -> None:
        super().__init__()
        self.forensic_dim = int(forensic_dim)
        self.num_crops = int(num_crops)

        # Learnable spatial role embedding for [Center, TopLeft_Corner, BottomRight_Corner, Top_Edge, Right_Edge]
        self.crop_role_embed = nn.Parameter(torch.randn(1, self.num_crops, self.forensic_dim) * 0.02)

        # Attention network scoring how informative each crop is
        mid = max(64, self.forensic_dim // 4)
        self.crop_attn = nn.Sequential(
            nn.Linear(self.forensic_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

        # Combines [attn_pool, center_vs_periphery, crop_std, crop_max] -> forensic_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.forensic_dim * 4, self.forensic_dim),
            nn.LayerNorm(self.forensic_dim),
            nn.GELU(),
        )

    def forward(self, crop_feats: torch.Tensor) -> torch.Tensor:
        """Aggregate crop features of shape (B, K, D) -> (B, D)."""
        b, k, d = crop_feats.shape
        if k == 1:
            single = crop_feats[:, 0, :]
            zeros = torch.zeros_like(single)
            return self.fusion_proj(torch.cat([single, zeros, zeros, single], dim=-1))

        if k <= self.num_crops:
            role_emb = self.crop_role_embed[:, :k, :]
        else:
            role_emb = F.interpolate(
                self.crop_role_embed.permute(0, 2, 1), size=k, mode="linear", align_corners=False
            ).permute(0, 2, 1)

        x = crop_feats + role_emb  # (B, K, D)

        # 1. Learned importance-weighted crop pooling
        attn_logits = self.crop_attn(x)  # (B, K, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_pool = (x * attn_weights).sum(dim=1)  # (B, D)

        # 2. Center (crop 0) vs Periphery (corners & edges, crops 1..K-1) discrepancy
        center_feat = x[:, 0, :]
        periphery_feat = x[:, 1:, :].mean(dim=1)
        center_vs_periphery = center_feat - periphery_feat  # (B, D)

        # 3. Cross-crop variance/std & peak anomaly max
        crop_std = x.var(dim=1, unbiased=False).add(1e-5).sqrt()  # (B, D)
        crop_max = x.amax(dim=1)  # (B, D)

        combined = torch.cat([attn_pool, center_vs_periphery, crop_std, crop_max], dim=-1)  # (B, 4*D)
        return self.fusion_proj(combined)


class FvfAresModel(nn.Module):
    """Dual-Stream Asymmetric AI Image Detector (Global Semantic + Multi-Crop 20-Channel Native Forensic)."""

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny",
        pretrained: bool = True,
        forensic_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        drop_path_rate: float = 0.0,
        num_classes: int = 2,
        num_crops: int = 5,
        semantic_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        import timm

        resolved_name = BACKBONE_ALIASES.get(backbone_name, backbone_name)
        self.backbone_name = backbone_name
        self.resolved_backbone_name = resolved_name
        self.num_crops = int(num_crops)
        self.semantic_dropout_prob = float(semantic_dropout_prob)

        # 20-channel pure-Conv2d feature extractor (baked into model for single-pass GPU/ONNX execution)
        self.channel_extractor = ForensicChannelExtractor20(standardize_forensic=True)

        # ImageNet normalization buffers for Stream A (Global Semantic RGB)
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False
        )

        # Stream A: Global Semantic Backbone
        try:
            self.semantic_backbone = timm.create_model(
                resolved_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
            )
        except Exception:
            self.semantic_backbone = timm.create_model(
                resolved_name, pretrained=False, num_classes=0
            )

        semantic_dim = int(self.semantic_backbone.num_features)
        self.semantic_norm = nn.LayerNorm(semantic_dim)

        # Stream B: 20-Channel Native-Grid Forensic Backbone + Multi-Crop Aggregator
        self.forensic_backbone = SRNetForensicBranch20(
            in_channels=20, out_dim=forensic_dim
        )
        self.multi_crop_aggregator = MultiCropForensicAggregator(
            forensic_dim=forensic_dim, num_crops=self.num_crops
        )

        # Auxiliary forensic classifier head (supervises Stream B directly during training)
        self.forensic_aux_head = nn.Linear(forensic_dim, num_classes)

        # Cross-stream alignment & Gated Fusion into One Final Head
        self.forensic_to_sem = nn.Linear(forensic_dim, semantic_dim)

        # Initialized to 0.5 so cross-stream gradients flow actively from step 1
        self.fusion_gate = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        fusion_in_dim = semantic_dim + forensic_dim + semantic_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Post-training temperature scaling parameter (default 1.0)
        self.register_buffer("temperature", torch.ones(1, dtype=torch.float32))

    def set_backbone_trainable(self, trainable: bool, unfreeze_last_stage_only: bool = False) -> None:
        """Freeze or unfreeze the semantic backbone for two-phase training."""
        for name, param in self.semantic_backbone.named_parameters():
            if not trainable:
                param.requires_grad = False
            elif unfreeze_last_stage_only:
                # In ConvNeXt-V2 / timm backbones, stages.3 or stages.2+3 and norm_pre/head
                is_late = any(k in name for k in ("stages.3", "stages.2", "norm", "head", "final"))
                param.requires_grad = is_late
            else:
                param.requires_grad = True

    def _forward_single_crop(self, rgb_i: torch.Tensor) -> torch.Tensor:
        ch20 = self.channel_extractor(rgb_i)
        return self.forensic_backbone(ch20)

    def extract_streams(
        self, global_rgb: torch.Tensor, native_crop_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract global semantic embedding and multi-crop 20-channel native forensic embedding.

        Args:
            global_rgb: (B, 3, H, W) resized full image in [0, 1].
            native_crop_rgb: Either (B, K, 3, H, W) multi-crop stack in [0, 1]
                             or (B, 3, H, W) single crop in [0, 1].
        """
        # Stream A: Normalize global RGB and run semantic backbone
        rgb_norm = (global_rgb - self.rgb_mean) / self.rgb_std
        sem_feat = self.semantic_backbone(rgb_norm)
        if sem_feat.dim() > 2:
            sem_feat = sem_feat.mean(dim=(-2, -1))
        sem_feat = self.semantic_norm(sem_feat)

        # Stream B: Multi-crop 20-channel forensic feature extraction + aggregation
        if native_crop_rgb.dim() == 5:
            b, k, c, h, w = native_crop_rgb.shape
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                crop_feat_list = []
                for i in range(k):
                    rgb_i = native_crop_rgb[:, i].detach().requires_grad_(True)
                    crop_feat_list.append(checkpoint(self._forward_single_crop, rgb_i, use_reentrant=False))
                crop_feats = torch.stack(crop_feat_list, dim=1)  # (B, K, D)
            else:
                flat_crops = native_crop_rgb.reshape(b * k, c, h, w)
                with torch.no_grad():
                    ch20 = self.channel_extractor(flat_crops)
                crop_feats = self.forensic_backbone(ch20).reshape(b, k, -1)
        else:
            with torch.no_grad():
                ch20 = self.channel_extractor(native_crop_rgb)
            crop_feats = self.forensic_backbone(ch20).unsqueeze(1)  # (B, 1, D)

        for_feat = self.multi_crop_aggregator(crop_feats)
        return sem_feat, for_feat

    def _fuse_and_classify(self, sem_feat: torch.Tensor, for_feat: torch.Tensor) -> torch.Tensor:
        """Fuse Stream A and multi-crop Stream B into the One Final Head."""
        # During training, randomly drop out the semantic stream on a fraction of samples
        # so the final head is forced to rely heavily on the multi-crop forensic representation.
        if self.training and self.semantic_dropout_prob > 0.0:
            keep_mask = (
                torch.rand((sem_feat.size(0), 1), device=sem_feat.device) >= self.semantic_dropout_prob
            ).to(sem_feat.dtype)
            sem_feat = sem_feat * keep_mask

        gate = torch.tanh(self.fusion_gate)
        for_projected = self.forensic_to_sem(for_feat)
        sem_modulated = sem_feat + gate * for_projected
        cross_interaction = sem_feat * torch.sigmoid(for_projected)

        fused = torch.cat([sem_modulated, for_feat, cross_interaction], dim=1)
        logits = self.head(fused)

        temp = self.temperature.clamp_min(0.05)
        return logits / temp

    def forward_with_aux(
        self, global_rgb: torch.Tensor, native_crop_rgb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (final_head_logits, forensic_aux_logits) for training."""
        if native_crop_rgb is None:
            native_crop_rgb = global_rgb
        sem_feat, for_feat = self.extract_streams(global_rgb, native_crop_rgb)
        logits = self._fuse_and_classify(sem_feat, for_feat)
        aux_logits = self.forensic_aux_head(for_feat)
        return logits, aux_logits

    def forward(
        self, global_rgb: torch.Tensor, native_crop_rgb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returning calibrated logits of shape (B, 2).

        If `native_crop_rgb` is None, `global_rgb` is used for both streams.
        """
        if native_crop_rgb is None:
            native_crop_rgb = global_rgb

        sem_feat, for_feat = self.extract_streams(global_rgb, native_crop_rgb)
        return self._fuse_and_classify(sem_feat, for_feat)


# Alias for naming consistency
FvfHeimdall02Model = FvfAresModel
