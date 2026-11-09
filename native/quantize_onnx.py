"""Dynamic INT8 Quantization Utility for Fire vs Fire Classifiers.

Quantizes FP32 ONNX neural network models to 8-bit integer (INT8) weights.
Benefits:
  - 4x reduction in model file size (e.g. 385 MB -> 98 MB)
  - 4x reduction in memory bandwidth
  - Unlocks peak matrix performance on Qualcomm Hexagon NPU (45 TOPS) and CPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("[ERROR] onnxruntime is required for quantization: pip install onnxruntime")
    sys.exit(1)


def quantize_model(input_path: Path, output_path: Optional[Path] = None, weight_type: QuantType = QuantType.QUInt8) -> bool:
    if not input_path.is_file():
        print(f"[ERROR] Input model not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_int8{input_path.suffix}")

    orig_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"\n[Quantize] Input: {input_path.name} ({orig_size_mb:.1f} MB)")
    print(f"[Quantize] Output: {output_path.name}")
    print("[Quantize] Target: Dynamic INT8 (QUInt8 / QInt8) quantization...")

    t0 = time.time()
    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=weight_type,
            per_channel=True,
            reduce_range=False,
        )
        duration_s = time.time() - t0
        new_size_mb = output_path.stat().st_size / (1024 * 1024)
        ratio = (1.0 - (new_size_mb / orig_size_mb)) * 100.0

        print(f"[SUCCESS] Quantized {output_path.name} in {duration_s:.2f}s")
        print(f"  -> Size: {orig_size_mb:.1f} MB -> {new_size_mb:.1f} MB ({ratio:.1f}% reduction)")
        return True
    except Exception as exc:
        print(f"[ERROR] Quantization failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quantize Fire vs Fire ONNX models to INT8.")
    parser.add_argument("--model", type=str, help="Path to input .onnx model")
    parser.add_argument("--output", type=str, default=None, help="Path for quantized .onnx output")
    args = parser.parse_args()

    if args.model:
        quantize_model(Path(args.model), Path(args.output) if args.output else None)
    else:
        # Auto-discover and quantize all FP32 models in native/classifiers
        classifiers_dir = Path(__file__).resolve().parent / "classifiers"
        models = [
            classifiers_dir / "fvf_heimdall_02_best.onnx",
            classifiers_dir / "spider_detector_resnet50.onnx",
        ]
        found = False
        for m in models:
            if m.is_file():
                found = True
                out = m.with_name(f"{m.stem.replace('_best', '')}_int8.onnx") if "fvf" in m.name else m.with_name(f"{m.stem}_int8.onnx")
                quantize_model(m, out)

        if not found:
            print("[INFO] No FP32 .onnx models found to quantize.")


if __name__ == "__main__":
    main()
