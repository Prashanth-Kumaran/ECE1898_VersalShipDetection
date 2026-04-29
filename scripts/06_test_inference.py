#!/usr/bin/env python3
"""
06_test_inference.py
====================
Run inference on test images and save visual results.
Also reports per-image timing for performance benchmarking
(useful to compare with post-quantization latency on VCK-190).

Usage:
    python scripts/06_test_inference.py \
        --weights runs/train/ship_yolov5s/weights/best.pt \
        --source  dataset/images/test \
        --output  runs/inference/ship_test \
        --conf    0.25 \
        --iou     0.45 \
        --n       20
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

YOLOV5_DIR = Path("../yolov5")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/train/ship_yolov5s/weights/best.pt")
    p.add_argument("--source",  default="dataset/images/test",
                   help="Directory of test images or single image path")
    p.add_argument("--output",  default="runs/inference/ship_test")
    p.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--img-size",type=int,   default=640)
    p.add_argument("--n",       type=int,   default=20,
                   help="Max images to process (0 = all)")
    p.add_argument("--device",  default="", help="cuda:0 or cpu")
    return p.parse_args()


def load_model(weights_path, device):
    """Load YOLOv5 model using torch.hub or local clone."""
    weights = str(Path(weights_path).resolve())

    # Try local clone first, fall back to torch.hub
    if YOLOV5_DIR.exists():
        sys.path.insert(0, str(YOLOV5_DIR))
        from models.experimental import attempt_load
        model = attempt_load(weights, device=device)
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom",
                               path=weights, verbose=False)
        model.to(device)

    model.eval()
    return model


def draw_detections(img_bgr, detections, conf_thresh=0.25):
    """Draw bounding boxes on image. detections: [[x1,y1,x2,y2,conf,cls], ...]"""
    COLOR = (0, 220, 60)  # bright green
    FONT  = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"ship {conf:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), COLOR, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 3),
                    FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image to new_shape, maintaining aspect ratio."""
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==> Device: {device}")
    print(f"==> Loading weights: {args.weights}")

    # ── Try ultralytics hub (simpler API) ────────────────────────────────────
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom",
                               path=args.weights, verbose=True, force_reload=False)
        model.conf = args.conf
        model.iou  = args.iou
        model.to(device)
        use_hub = True
    except Exception:
        use_hub = False
        print("torch.hub failed — ensure YOLOv5 is cloned at ../yolov5")
        return

    # ── Collect images ────────────────────────────────────────────────────────
    src = Path(args.source)
    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([p for p in src.iterdir() if p.suffix.lower() in exts])
    else:
        images = [src]

    if args.n > 0:
        images = images[: args.n]

    print(f"==> Running inference on {len(images)} images...")

    # ── Inference loop ────────────────────────────────────────────────────────
    times = []
    total_dets = 0

    for i, img_path in enumerate(images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARN: could not read {img_path}")
            continue

        t0 = time.perf_counter()
        if use_hub:
            results = model(img_bgr[:, :, ::-1])  # BGR→RGB for hub
            t1 = time.perf_counter()

            # Parse results
            det = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        else:
            t1 = t0
            det = np.empty((0, 6))

        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)
        total_dets += len(det)

        # Draw and save
        vis = draw_detections(img_bgr.copy(), det, conf_thresh=args.conf)
        out_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(out_path), vis)

        print(f"  [{i+1:3d}/{len(images)}] {img_path.name:40s} "
              f"| {len(det):3d} dets | {elapsed_ms:6.1f} ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    if times:
        print("\n── Timing Summary ──────────────────────────────────────────")
        print(f"  Images processed   : {len(times)}")
        print(f"  Total detections   : {total_dets}")
        print(f"  Mean latency       : {np.mean(times):.1f} ms  "
              f"({1000/np.mean(times):.1f} FPS)")
        print(f"  Min / Max latency  : {np.min(times):.1f} / {np.max(times):.1f} ms")
        print(f"\n  NOTE: GPU inference latency above is the PC baseline.")
        print(f"  VCK-190 DPU target is typically 5-20 ms @ 640×640 after quant.")
        print(f"\n==> Saved {len(times)} annotated images to: {out_dir}/")


if __name__ == "__main__":
    main()
