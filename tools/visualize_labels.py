#!/usr/bin/env python3
"""
tools/visualize_labels.py
=========================
Draw ground-truth bounding boxes on a sample of images to verify the
VOC → YOLO conversion was correct before training.

Usage:
    python tools/visualize_labels.py \
        --split train \
        --n 12 \
        --output runs/label_check
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["ship"]
COLORS = [(0, 200, 255)]   # one per class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",   default="train", choices=["train", "val", "test"])
    p.add_argument("--n",       type=int, default=16, help="Images to visualize")
    p.add_argument("--output",  default="runs/label_check")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--img-dir", default=None, help="Override image directory")
    p.add_argument("--lbl-dir", default=None, help="Override label directory")
    return p.parse_args()


def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_yolo_labels(img_bgr, label_path):
    h, w = img_bgr.shape[:2]
    if not Path(label_path).exists():
        return img_bgr, 0

    count = 0
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_abs(cx, cy, bw, bh, w, h)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            cv2.putText(img_bgr, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            count += 1
    return img_bgr, count


def main():
    args = parse_args()
    img_dir = Path(args.img_dir or f"dataset/images/{args.split}")
    lbl_dir = Path(args.lbl_dir or f"dataset/labels/{args.split}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        print(f"No images found in {img_dir}")
        return

    random.seed(args.seed)
    sample = random.sample(images, min(args.n, len(images)))

    print(f"Visualizing {len(sample)} images from '{args.split}' split...")
    total_boxes = 0

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img, n = draw_yolo_labels(img, lbl_path)
        total_boxes += n

        # Overlay filename and box count
        cv2.putText(img, f"{img_path.name} | {n} ships",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)

    print(f"  Total boxes drawn : {total_boxes}")
    print(f"  Avg boxes/image   : {total_boxes / len(sample):.1f}")
    print(f"  Saved to          : {out_dir}/")
    print("\n  Open those images and verify:")
    print("  ✓ Boxes tightly wrap ships")
    print("  ✓ No wildly misplaced boxes")
    print("  ✓ No boxes on empty ocean/land")


if __name__ == "__main__":
    main()
