#!/usr/bin/env bash
# =============================================================================
# 05_validate.sh
# Run YOLOv5 validation on the val and test sets
# Reports: mAP@0.5, mAP@0.5:0.95, Precision, Recall
# =============================================================================

set -euo pipefail

YOLOV5_DIR="../yolov5"
WEIGHTS="runs/train/ship_yolov5s/weights/best.pt"
IMG_SIZE=640
BATCH_SIZE=32
CONF_THRES=0.001   # Low threshold for mAP calculation (sweep the full PR curve)
IOU_THRES=0.60     # IoU threshold for NMS

echo "========================================"
echo " Validation — Val Set"
echo "========================================"
python "$YOLOV5_DIR/val.py" \
    --data    data/ship.yaml \
    --weights "$(pwd)/$WEIGHTS" \
    --img     "$IMG_SIZE" \
    --batch   "$BATCH_SIZE" \
    --conf    "$CONF_THRES" \
    --iou     "$IOU_THRES" \
    --task    val \
    --project "$(pwd)/runs/val" \
    --name    "ship_val" \
    --exist-ok \
    --verbose \
    --save-txt \
    --save-conf

echo ""
echo "========================================"
echo " Validation — Test Set"
echo "========================================"
python "$YOLOV5_DIR/val.py" \
    --data    data/ship.yaml \
    --weights "$(pwd)/$WEIGHTS" \
    --img     "$IMG_SIZE" \
    --batch   "$BATCH_SIZE" \
    --conf    "$CONF_THRES" \
    --iou     "$IOU_THRES" \
    --task    test \
    --project "$(pwd)/runs/test" \
    --name    "ship_test" \
    --exist-ok \
    --verbose \
    --save-txt \
    --save-conf

echo ""
echo "==> Validation complete."
echo "==> Results in runs/val/ship_val/ and runs/test/ship_test/"
echo ""
echo "Interpretation guide:"
echo "  mAP@0.5 > 0.70  → good for single-class aerial ship detection"
echo "  mAP@0.5 > 0.80  → excellent — ready for quantization trials"
echo "  mAP@0.5 < 0.60  → consider more data or hyperparameter tuning"
