#!/usr/bin/env bash
# =============================================================================
# 07_export_torchscript.sh
# Export trained YOLOv5s to TorchScript and ONNX
#
# WHY: Vitis-AI's vai_q_pytorch quantizer works on a traced TorchScript model.
#      ONNX is useful for intermediate inspection with Netron.
#
# After this step, Stage 2 (quantization) uses:
#   vai_q_pytorch --model runs/export/ship_yolov5s.torchscript ...
# =============================================================================

set -euo pipefail

YOLOV5_DIR="../yolov5"
WEIGHTS="$(pwd)/runs/train/ship_yolov5s/weights/best.pt"
IMG_SIZE=640
BATCH=1
OUT_DIR="$(pwd)/runs/export"

mkdir -p "$OUT_DIR"

echo "==> Exporting to TorchScript (for Vitis-AI vai_q_pytorch)..."
python "$YOLOV5_DIR/export.py" \
    --weights "$WEIGHTS" \
    --img     "$IMG_SIZE" \
    --batch   "$BATCH" \
    --include torchscript \
    --optimize   # fuse Conv+BN layers for faster inference

echo ""
echo "==> Exporting to ONNX (for Netron visualization / onnxruntime check)..."
python "$YOLOV5_DIR/export.py" \
    --weights "$WEIGHTS" \
    --img     "$IMG_SIZE" \
    --batch   "$BATCH" \
    --include onnx \
    --opset 13 \
    --simplify

# Move exports to our output dir
WEIGHTS_DIR="$(dirname "$WEIGHTS")"
for ext in ".torchscript" ".onnx"; do
    src="${WEIGHTS_DIR}/best${ext}"
    dst="${OUT_DIR}/ship_yolov5s${ext}"
    [ -f "$src" ] && cp "$src" "$dst" && echo "  Saved: $dst"
done

echo ""
echo "==> Export complete."
echo ""
echo "── Vitis-AI Stage 2 Checklist ──────────────────────────────────────"
echo "  1. Install Vitis-AI Docker: xilinx/vitis-ai:3.0-pytorch"
echo "  2. Calibration dataset    : use 100-200 images from dataset/images/val"
echo "  3. Quantize               :"
echo "       vai_q_pytorch \\"
echo "           --quant_mode  calib \\"
echo "           --model       runs/export/ship_yolov5s.torchscript \\"
echo "           --calib_datadir dataset/images/val \\"
echo "           --output_dir  runs/quantized"
echo "  4. Evaluate quantized mAP vs float mAP (target: <1% drop)"
echo "  5. Compile with Vitis-AI compiler for DPUCVDX8G_ISA3_C32B6"
echo "─────────────────────────────────────────────────────────────────────"
