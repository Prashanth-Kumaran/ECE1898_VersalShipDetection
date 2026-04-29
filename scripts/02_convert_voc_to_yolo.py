#!/usr/bin/env python3
"""
02_convert_voc_to_yolo.py
=========================
Convert Pascal VOC XML annotations (from the Kaggle ship detection dataset)
to YOLOv5 label format.

VOC format  : <xmin> <ymin> <xmax> <ymax>  (absolute pixel coords)
YOLO format : <class_id> <cx> <cy> <w> <h>  (normalized 0-1, center-based)

Usage:
    python scripts/02_convert_voc_to_yolo.py \
        --images  raw_dataset/images \
        --annotations raw_dataset/annotations \
        --output  dataset/labels/all

Output labels are written next to the images in dataset/labels/all/.
Run 03_split_dataset.py after this.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Class mapping ────────────────────────────────────────────────────────────
# The Kaggle dataset has a single class.  Extend here if you add more datasets.
CLASS_MAP = {
    "ship": 0,
    "Ship": 0,
}

# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="VOC XML → YOLOv5 label converter")
    p.add_argument("--images",      default="raw_dataset/images",
                   help="Directory containing source JPEG images")
    p.add_argument("--annotations", default="raw_dataset/annotations",
                   help="Directory containing Pascal VOC XML files")
    p.add_argument("--output",      default="dataset/labels/all",
                   help="Output directory for YOLO .txt label files")
    p.add_argument("--images-out",  default="dataset/images/all",
                   help="Output directory to copy images into")
    p.add_argument("--skip-empty",  action="store_true",
                   help="Skip images with no valid annotations")
    return p.parse_args()


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute VOC coords to normalized YOLO center format."""
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h


def convert_annotation(xml_path: Path, label_out: Path, skip_empty: bool) -> dict:
    """Parse one VOC XML file and write a YOLO .txt label file.

    Returns a stats dict.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    skipped_classes = set()

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            skipped_classes.add(name)
            continue

        class_id = CLASS_MAP[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Clamp to image bounds (some VOC annotations exceed image size slightly)
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box — skip

        cx, cy, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not lines and skip_empty:
        return {"boxes": 0, "skipped": True, "unknown_classes": skipped_classes}

    label_out.parent.mkdir(parents=True, exist_ok=True)
    with open(label_out, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    return {"boxes": len(lines), "skipped": False, "unknown_classes": skipped_classes}


def main():
    args = parse_args()
    ann_dir    = Path(args.annotations)
    img_dir    = Path(args.images)
    label_out  = Path(args.output)
    img_out    = Path(args.images_out)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        print(f"ERROR: No XML files found in {ann_dir}")
        return

    print(f"Found {len(xml_files)} XML annotation files.")

    total_boxes   = 0
    total_images  = 0
    skipped       = 0
    unknown_cls   = set()

    for xml_path in xml_files:
        stem = xml_path.stem
        label_path = label_out / f"{stem}.txt"

        stats = convert_annotation(xml_path, label_path, args.skip_empty)
        total_boxes  += stats["boxes"]
        unknown_cls  |= stats["unknown_classes"]

        if stats["skipped"]:
            skipped += 1
            continue

        # Copy image to output directory
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            src_img = img_dir / f"{stem}{ext}"
            if src_img.exists():
                img_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, img_out / src_img.name)
                break

        total_images += 1

    print("\n── Conversion Summary ──────────────────────────────────")
    print(f"  Images processed : {total_images}")
    print(f"  Images skipped   : {skipped}  (no annotations)")
    print(f"  Total boxes      : {total_boxes}")
    print(f"  Avg boxes/image  : {total_boxes / max(total_images, 1):.1f}")
    if unknown_cls:
        print(f"  Unknown classes  : {unknown_cls}  ← add to CLASS_MAP if needed")
    print(f"\n  Labels → {label_out}/")
    print(f"  Images → {img_out}/")
    print("\n==> Next step: python scripts/03_split_dataset.py")


if __name__ == "__main__":
    main()
