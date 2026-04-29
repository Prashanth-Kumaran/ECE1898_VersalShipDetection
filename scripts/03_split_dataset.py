#!/usr/bin/env python3
"""
03_split_dataset.py
===================
Split converted images + labels into train / val / test sets.

Default split: 70% train | 20% val | 10% test  (stratified shuffle)

Usage:
    python scripts/03_split_dataset.py \
        --images  dataset/images/all \
        --labels  dataset/labels/all \
        --train 0.70 --val 0.20 --test 0.10 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images",  default="dataset/images/all")
    p.add_argument("--labels",  default="dataset/labels/all")
    p.add_argument("--train",   type=float, default=0.70)
    p.add_argument("--val",     type=float, default=0.20)
    p.add_argument("--test",    type=float, default=0.10)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out-images", default="dataset/images")
    p.add_argument("--out-labels", default="dataset/labels")
    return p.parse_args()


def copy_files(stems, img_dir, lbl_dir, out_img_dir, out_lbl_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    missing_labels = 0
    for stem in stems:
        # Image — try common extensions
        img_copied = False
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG"):
            src = img_dir / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, out_img_dir / src.name)
                img_copied = True
                break
        if not img_copied:
            print(f"  WARNING: image not found for stem '{stem}'")

        # Label
        lbl = lbl_dir / f"{stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_lbl_dir / lbl.name)
        else:
            missing_labels += 1
            # Write empty label file (YOLOv5 expects it)
            (out_lbl_dir / f"{stem}.txt").write_text("")

    return missing_labels


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train + val + test must sum to 1.0"

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_img = Path(args.out_images)
    out_lbl = Path(args.out_labels)

    # Collect stems from label files (ground truth of what got converted)
    all_stems = sorted([p.stem for p in lbl_dir.glob("*.txt")])
    if not all_stems:
        print(f"ERROR: no .txt files found in {lbl_dir}")
        return

    random.seed(args.seed)
    random.shuffle(all_stems)

    n = len(all_stems)
    n_train = int(n * args.train)
    n_val   = int(n * args.val)

    train_stems = all_stems[:n_train]
    val_stems   = all_stems[n_train:n_train + n_val]
    test_stems  = all_stems[n_train + n_val:]

    print(f"Dataset split (seed={args.seed}):")
    print(f"  Total  : {n}")
    print(f"  Train  : {len(train_stems)} ({len(train_stems)/n*100:.1f}%)")
    print(f"  Val    : {len(val_stems)}   ({len(val_stems)/n*100:.1f}%)")
    print(f"  Test   : {len(test_stems)}  ({len(test_stems)/n*100:.1f}%)")

    for split, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        missing = copy_files(
            stems,
            img_dir, lbl_dir,
            out_img / split,
            out_lbl / split,
        )
        if missing:
            print(f"  [{split}] {missing} images had no label → empty .txt written")

    print("\n==> Split complete.")
    print("==> Next step: python tools/visualize_labels.py  (sanity check)")
    print("==>            python tools/dataset_stats.py")
    print("==>            bash scripts/04_train.sh")


if __name__ == "__main__":
    main()
