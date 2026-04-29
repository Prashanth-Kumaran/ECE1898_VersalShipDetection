#!/usr/bin/env python3
"""
tools/dataset_stats.py
======================
Analyze the converted YOLO dataset and print statistics useful for
training decisions and Vitis-AI quantization calibration set selection.

Reports:
  - Image count per split
  - Annotation count and distribution
  - Bounding box size distribution (small/medium/large ships)
  - Image resolution distribution
  - Recommended calibration set size for Vitis-AI PTQ

Usage:
    python tools/dataset_stats.py
"""

from pathlib import Path
import numpy as np


SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_labels(lbl_dir: Path):
    """Return flat list of (cx, cy, w, h) for all boxes in a label directory."""
    boxes = []
    n_images_with_ann = 0
    n_images_empty    = 0

    for txt in sorted(lbl_dir.glob("*.txt")):
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        if lines:
            n_images_with_ann += 1
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    boxes.append(list(map(float, parts[1:])))
        else:
            n_images_empty += 1

    return np.array(boxes, dtype=np.float32), n_images_with_ann, n_images_empty


def classify_box_size(w, h):
    """COCO-style size categories (normalized to [0,1])."""
    area = w * h
    if area < 0.01 ** 2:        # < 32×32 at 320px equivalent
        return "tiny"
    elif area < (0.05 ** 2):
        return "small"
    elif area < (0.2 ** 2):
        return "medium"
    else:
        return "large"


def main():
    print("=" * 58)
    print("  Ship Detection — Dataset Statistics")
    print("=" * 58)

    all_boxes = []
    total_images = 0

    for split in SPLITS:
        img_dir = Path(f"dataset/images/{split}")
        lbl_dir = Path(f"dataset/labels/{split}")

        if not lbl_dir.exists():
            print(f"\n[{split}] ← directory not found, skipping")
            continue

        n_imgs = len([p for p in img_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else 0
        boxes, n_with, n_empty = load_labels(lbl_dir)

        print(f"\n── {split.upper()} ─────────────────────────────────────")
        print(f"  Images total      : {n_imgs}")
        print(f"  Images w/ ships   : {n_with}")
        print(f"  Images empty      : {n_empty}")
        print(f"  Total boxes       : {len(boxes)}")

        if len(boxes):
            print(f"  Avg boxes/image   : {len(boxes)/max(n_with,1):.1f}")
            widths  = boxes[:, 2]
            heights = boxes[:, 3]
            areas   = widths * heights
            print(f"  Box w (norm) μ±σ  : {widths.mean():.3f} ± {widths.std():.3f}")
            print(f"  Box h (norm) μ±σ  : {heights.mean():.3f} ± {heights.std():.3f}")
            print(f"  Box area  μ±σ     : {areas.mean():.4f} ± {areas.std():.4f}")

            sizes = [classify_box_size(w, h) for w, h in zip(widths, heights)]
            for cat in ["tiny", "small", "medium", "large"]:
                n = sizes.count(cat)
                pct = 100 * n / len(sizes)
                bar = "█" * int(pct / 3)
                print(f"  {cat:8s}         : {n:5d} ({pct:5.1f}%)  {bar}")

        all_boxes.append(boxes)
        total_images += n_imgs

    # ── Global summary ────────────────────────────────────────────────────────
    if all_boxes:
        all_b = np.concatenate([b for b in all_boxes if len(b)])
        print("\n── GLOBAL ──────────────────────────────────────────────")
        print(f"  Total images      : {total_images}")
        print(f"  Total boxes       : {len(all_b)}")

    # ── Training recommendations ──────────────────────────────────────────────
    print("\n── Recommendations ─────────────────────────────────────")
    print("  YOLOv5s input size: 640×640 (keep fixed for Vitis-AI)")
    print("  Early stopping    : patience=50 epochs is safe")
    print("  Vitis-AI PTQ cal  : use 100–200 images from val set")
    if len(all_b) and (np.array([classify_box_size(w,h) for w,h in zip(all_b[:,2],all_b[:,3])])=="tiny").mean() > 0.3:
        print("  ⚠  >30% tiny boxes detected — consider:")
        print("     • img-size 1280 during training (then resize for quant)")
        print("     • P2 head variant (yolov5s6.pt) for better small object recall")
    print("=" * 58)


if __name__ == "__main__":
    main()
