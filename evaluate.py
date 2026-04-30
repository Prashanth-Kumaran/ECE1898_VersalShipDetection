"""
Evaluation script for YOLOv5 Ship Detection.
Run: python evaluate.py

Metrics:
  - mAP@0.5
  - mAP@0.5:0.95  (COCO-style)
  - Precision, Recall, F1 at best threshold
  - Confusion matrix
  - Visual grid of predictions vs ground truth
"""

import os, glob, random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from yolov5_ship import (
    YOLOv5, parse_voc_xml, decode_predictions,
    ANCHORS, STRIDES, IMG_SIZE, NUM_CLS, DEVICE, CKPT,
    ANN_DIR, IMG_DIR
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def box_iou(b1, b2):
    """IoU between (N,4) and (M,4) boxes in xyxy format."""
    area1 = (b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1])
    area2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    ix1 = torch.max(b1[:,None,0], b2[None,:,0])
    iy1 = torch.max(b1[:,None,1], b2[None,:,1])
    ix2 = torch.min(b1[:,None,2], b2[None,:,2])
    iy2 = torch.min(b1[:,None,3], b2[None,:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    return inter / (area1[:,None] + area2[None,:] - inter + 1e-8)

def compute_ap(recalls, precisions):
    """11-point interpolated AP."""
    ap = 0.
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if p.size else 0.) / 11
    return ap

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path):
    """Load and normalise one image → (1,3,H,W) tensor."""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(img_path)
    orig_h, orig_w = img.shape[:2]
    inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.
    inp = ((inp - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(inp).unsqueeze(0), img, orig_h, orig_w

def preprocess_batch(img_paths):
    """Load and normalise a batch of images."""
    tensors, imgs, shapes = [], [], []
    for p in img_paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        shapes.append(img.shape[:2])
        imgs.append(img)
        inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.
        inp = ((inp - MEAN) / STD).transpose(2, 0, 1)
        tensors.append(torch.from_numpy(inp))
    return torch.stack(tensors), imgs, shapes

# ─────────────────────────────────────────────
# CORE EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, img_paths, iou_threshold=0.5, conf_threshold=0.01, batch_size=16):
    model.eval()
    all_detections = []
    total_gt = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(img_paths), batch_size),
                          desc=f"Evaluating IoU={iou_threshold:.2f}"):
            batch_paths = img_paths[start:start+batch_size]
            batch_tensors, _, shapes = preprocess_batch(batch_paths)
            preds = model(batch_tensors.to(DEVICE))
            batch_dets = decode_predictions(preds, ANCHORS, STRIDES,
                                            conf_thr=conf_threshold)

            for i, img_p in enumerate(batch_paths):
                xml_p    = os.path.join(ANN_DIR, os.path.splitext(
                               os.path.basename(img_p))[0] + ".xml")
                gt_boxes = parse_voc_xml(xml_p) if os.path.exists(xml_p) else []
                total_gt += len(gt_boxes)

                orig_h, orig_w = shapes[i]
                sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
                dets = batch_dets[i]
                pred_boxes = [[x1*sx,y1*sy,x2*sx,y2*sy] for (x1,y1,x2,y2,_) in dets]
                pred_confs = [conf for (*_,conf) in dets]

                if not gt_boxes:
                    for conf in pred_confs:
                        all_detections.append((conf, 0))
                    continue

                gt_t    = torch.tensor(gt_boxes, dtype=torch.float32)
                matched = set()
                for j in np.argsort(pred_confs)[::-1]:
                    pb   = torch.tensor(pred_boxes[j], dtype=torch.float32).unsqueeze(0)
                    ious = box_iou(pb, gt_t)[0]
                    best_iou, best_k = ious.max(0)
                    if best_iou >= iou_threshold and best_k.item() not in matched:
                        all_detections.append((pred_confs[j], 1))
                        matched.add(best_k.item())
                    else:
                        all_detections.append((pred_confs[j], 0))

    # PR curve
    all_detections.sort(key=lambda x: -x[0])
    confs = np.array([d[0] for d in all_detections])
    tps   = np.array([d[1] for d in all_detections])
    tp_cs = np.cumsum(tps)
    fp_cs = np.cumsum(1 - tps)
    rec   = tp_cs / (total_gt + 1e-8)
    prec  = tp_cs / (tp_cs + fp_cs + 1e-8)
    ap    = compute_ap(rec, prec)

    f1       = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = f1.argmax()

    return {
        "AP":        ap,
        "precision": prec[best_idx],
        "recall":    rec[best_idx],
        "f1":        f1[best_idx],
        "best_conf": confs[best_idx],
        "total_gt":  total_gt,
        "total_det": len(all_detections),
        "pr_curve":  (rec, prec),
    }

def evaluate_coco(model, img_paths):
    """mAP@0.5:0.95"""
    aps = []
    for iou_t in np.arange(0.5, 1.0, 0.05):
        r = evaluate(model, img_paths, iou_threshold=round(iou_t, 2))
        aps.append(r["AP"])
        print(f"  AP@{iou_t:.2f} = {r['AP']:.4f}")
    return float(np.mean(aps))

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_pr_curve(rec, prec, ap, save="pr_curve.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, "b-", linewidth=2, label=f"AP@0.5 = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Ship Detection")
    plt.legend(); plt.grid(True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save}")

def plot_prediction_grid(model, img_paths, n=6, conf_thr=0.4, save="predictions.png"):
    samples = random.sample(img_paths, min(n, len(img_paths)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    model.eval()

    for ax, img_p in zip(axes, samples):
        xml_p    = os.path.join(ANN_DIR, os.path.splitext(os.path.basename(img_p))[0] + ".xml")
        gt_boxes = parse_voc_xml(xml_p) if os.path.exists(xml_p) else []

        inp, img_rgb, orig_h, orig_w = preprocess(img_p)
        with torch.no_grad():
            preds = model(inp.to(DEVICE))
        dets = decode_predictions(preds, ANCHORS, STRIDES, conf_thr=conf_thr)[0]

        ax.imshow(img_rgb)
        for (x1,y1,x2,y2) in gt_boxes:
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                         linewidth=1.5, edgecolor="lime", facecolor="none"))

        sx, sy = orig_w/IMG_SIZE, orig_h/IMG_SIZE
        for (x1,y1,x2,y2,conf) in dets:
            ax.add_patch(patches.Rectangle((x1*sx,y1*sy),(x2-x1)*sx,(y2-y1)*sy,
                         linewidth=1.5, edgecolor="red", facecolor="none"))
            ax.text(x1*sx, y1*sy-4, f"{conf:.2f}", color="red", fontsize=7)

        ax.axis("off")
        ax.set_title(os.path.basename(img_p), fontsize=8)

    plt.suptitle("Green = GT  |  Red = Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save}")

def plot_confusion_matrix(model, img_paths, conf_thr=0.4, iou_thr=0.5, save="confusion.png"):
    TP = FP = FN = 0
    model.eval()

    for img_p in tqdm(img_paths, desc="Confusion matrix"):
        xml_p    = os.path.join(ANN_DIR, os.path.splitext(os.path.basename(img_p))[0] + ".xml")
        gt_boxes = parse_voc_xml(xml_p) if os.path.exists(xml_p) else []

        inp, _, orig_h, orig_w = preprocess(img_p)
        with torch.no_grad():
            preds = model(inp.to(DEVICE))
        dets = decode_predictions(preds, ANCHORS, STRIDES, conf_thr=conf_thr)[0]

        sx, sy = orig_w/IMG_SIZE, orig_h/IMG_SIZE
        pred_boxes = [[x1*sx,y1*sy,x2*sx,y2*sy] for (x1,y1,x2,y2,_) in dets]

        if not gt_boxes:  FP += len(pred_boxes); continue
        if not pred_boxes: FN += len(gt_boxes);  continue

        gt_t   = torch.tensor(gt_boxes,   dtype=torch.float32)
        pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
        ious   = box_iou(pred_t, gt_t)
        matched = set()
        for i in range(len(pred_boxes)):
            best_iou, best_j = ious[i].max(0)
            if best_iou >= iou_thr and best_j.item() not in matched:
                TP += 1; matched.add(best_j.item())
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched)

    cm = np.array([[TP, FN], [FP, 0]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: Ship", "Pred: BG"])
    ax.set_yticklabels(["GT: Ship",   "GT: BG"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"Confusion Matrix  (conf>{conf_thr}, IoU>{iou_thr})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"TP={TP}  FP={FP}  FN={FN}")
    print(f"Saved → {save}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf",       type=float, default=0.15)
    ap.add_argument("--nms",        type=float, default=0.30)
    ap.add_argument("--batch",      type=int,   default=16)
    ap.add_argument("--no-coco",    action="store_true",
                    help="Skip slow mAP@0.5:0.95 computation")
    args = ap.parse_args()

    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    random.seed(42); random.shuffle(all_imgs)
    val_imgs = all_imgs[int(0.85 * len(all_imgs)):]
    print(f"Evaluating on {len(val_imgs)} validation images...")
    print(f"  conf={args.conf}  nms={args.nms}  batch={args.batch}")

    model = YOLOv5(NUM_CLS).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE,
                                     weights_only=False))
    model.eval()

    # ── mAP@0.5 ───────────────────────────────
    results = evaluate(model, val_imgs,
                       iou_threshold=0.5,
                       conf_threshold=args.conf,
                       batch_size=args.batch)
    print("\n── Results ──────────────────────────────")
    print(f"  AP@0.5     : {results['AP']:.4f}")
    print(f"  Precision  : {results['precision']:.4f}")
    print(f"  Recall     : {results['recall']:.4f}")
    print(f"  F1 Score   : {results['f1']:.4f}")
    print(f"  Best conf  : {results['best_conf']:.4f}")
    print(f"  Total GT   : {results['total_gt']}")
    print(f"  Total dets : {results['total_det']}")

    # ── mAP@0.5:0.95 ──────────────────────────
    if not args.no_coco:
        print("\n── mAP@0.5:0.95 ─────────────────────────")
        map_coco = evaluate_coco(model, val_imgs)
        print(f"  mAP@0.5:0.95 = {map_coco:.4f}")

    # ── Plots ─────────────────────────────────
    plot_pr_curve(*results["pr_curve"], results["AP"])
    plot_prediction_grid(model, val_imgs, conf_thr=args.conf)
    plot_confusion_matrix(model, val_imgs,
                          conf_thr=args.conf, iou_thr=args.nms)