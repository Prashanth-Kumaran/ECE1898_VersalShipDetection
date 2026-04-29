#!/usr/bin/env bash
# =============================================================================
# 04_train.sh
# Launch YOLOv5s training for ship detection
# =============================================================================
# Prerequisites:
#   - YOLOv5 cloned at ../yolov5 (relative to this project root)
#   - Dataset split complete (scripts 02 + 03 run)
#   - GPU with CUDA available (recommended: RTX 3070 or better)
#
# Adjust BATCH_SIZE and IMG_SIZE to fit your GPU VRAM:
#   8GB  VRAM → batch 16  @ 640
#   12GB VRAM → batch 32  @ 640
#   24GB VRAM → batch 64  @ 640
#
# Usage:
#   cd ship_detection/
#   bash scripts/04_train.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
YOLOV5_DIR="../yolov5"          # Path to your YOLOv5 clone
DATA_YAML="configs/ship.yaml"   # Relative to YOLOV5_DIR — see note below
MODEL="yolov5s.pt"              # Pretrained COCO weights (auto-downloaded)
IMG_SIZE=640                    # Keep at 640 for Vitis-AI compatibility
BATCH_SIZE=32                   # Adjust for your GPU
EPOCHS=200                      # Start here; early stopping will kick in
WORKERS=8                       # DataLoader workers (= num CPU cores / 2)
PROJECT="runs/train"
NAME="ship_yolov5s"
PATIENCE=50                     # Early stopping patience (epochs)

# ── Vitis-AI friendly hyperparameters ────────────────────────────────────────
# Using scratch-low hyps: moderate augmentation, good generalization
HYP="$YOLOV5_DIR/data/hyps/hyp.scratch-low.yaml"

# ─────────────────────────────────────────────────────────────────────────────

echo "==> Checking YOLOv5 directory..."
if [ ! -d "$YOLOV5_DIR" ]; then
    echo "ERROR: YOLOv5 not found at $YOLOV5_DIR"
    echo "       Run: git clone https://github.com/ultralytics/yolov5 $YOLOV5_DIR"
    exit 1
fi

# configs/ship.yaml uses a path relative to itself — copy to yolov5/data/
echo "==> Copying dataset config to yolov5/data/..."
cp "$DATA_YAML" "$YOLOV5_DIR/data/ship.yaml"

echo "==> Starting training..."
echo "    Model    : $MODEL"
echo "    Img size : $IMG_SIZE"
echo "    Batch    : $BATCH_SIZE"
echo "    Epochs   : $EPOCHS (patience=$PATIENCE)"
echo ""

python "$YOLOV5_DIR/train.py" \
    --data    data/ship.yaml \
    --weights "$MODEL" \
    --cfg     models/yolov5s.yaml \
    --hyp     "$HYP" \
    --img     "$IMG_SIZE" \
    --batch   "$BATCH_SIZE" \
    --epochs  "$EPOCHS" \
    --workers "$WORKERS" \
    --project "$(pwd)/$PROJECT" \
    --name    "$NAME" \
    --patience "$PATIENCE" \
    --save-period 10 \
    --exist-ok \
    --cache ram

# ── Post-training summary ─────────────────────────────────────────────────────
echo ""
echo "==> Training complete!"
echo "==> Best weights : $PROJECT/$NAME/weights/best.pt"
echo "==> Last weights : $PROJECT/$NAME/weights/last.pt"
echo "==> Results      : $PROJECT/$NAME/results.csv"
echo ""
echo "==> Next step: bash scripts/05_validate.sh"
