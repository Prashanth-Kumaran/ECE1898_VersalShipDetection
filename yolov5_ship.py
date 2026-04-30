"""
YOLOv5s Ship Detection — Vitis AI 3.0 / VCK190 Compatible
===========================================================
Dataset : https://www.kaggle.com/datasets/andrewmvd/ship-detection
           data/images/*.png   +   data/annotations/*.xml  (Pascal VOC)

Vitis AI 3.0 compatibility decisions:
  - LeakyReLU(0.1) throughout  (SiLU is NOT supported by the VCK190 DPU)
  - Conv → BN → Activation ordering (required for BN fusion in compiler)
  - No dynamic shapes, no custom ops, no slicing tricks
  - Upsample uses nearest-neighbour (bilinear is not DPU-friendly)
  - Detection head outputs raw tensors — sigmoid applied outside the graph
    so the DPU subgraph ends cleanly at the last Conv

Install:
  pip install torch torchvision opencv-python albumentations tqdm

Train:
  python yolov5_ship.py --mode train

Infer:
  python yolov5_ship.py --mode infer --image path/to/image.png --out result.png
"""

import os, glob, random, math, xml.etree.ElementTree as ET, argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
IMG_DIR  = "data/images"
ANN_DIR  = "data/annotations"
IMG_SIZE = 640          # YOLOv5 default; must be multiple of 32
NUM_CLS  = 1            # "ship" only
BATCH    = 8
EPOCHS   = 80
LR       = 1e-3
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT     = "yolov5_ship_best.pth"

# 9 k-means anchors tuned for aerial ship imagery (w×h, pixels at IMG_SIZE=640)
# Replace with output of anchor_kmeans() after first run for best results
ANCHORS = [
    [(8, 9), (38, 43), (46, 52)],    # P3/8   small
    [(80, 96), (88, 107), (98, 110)],   # P4/16  medium
    [(170, 180), (203, 210), (396, 448)],  # P5/32  large
]
STRIDES = [8, 16, 32]

# ──────────────────────────────────────────────────────────────
# ANCHOR K-MEANS  (run once, paste output back into ANCHORS)
# ──────────────────────────────────────────────────────────────
def anchor_kmeans(n_clusters=9):
    from sklearn.cluster import KMeans
    boxes = []
    for xml_p in glob.glob(os.path.join(ANN_DIR, "*.xml")):
        root = ET.parse(xml_p).getroot()
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            w = float(bb.find("xmax").text) - float(bb.find("xmin").text)
            h = float(bb.find("ymax").text) - float(bb.find("ymin").text)
            boxes.append([w, h])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(boxes)
    centers = np.sort(km.cluster_centers_, axis=0).astype(int).tolist()
    grouped = [centers[i*3:(i+1)*3] for i in range(3)]
    print("Paste these into ANCHORS:")
    for g in grouped:
        print(" ", [(int(a[0]), int(a[1])) for a in g])
    return grouped

# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
def parse_voc_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    return [[float(bb.find(t).text)
             for t in ("xmin","ymin","xmax","ymax")]
            for obj in root.findall("object")
            for bb in [obj.find("bndbox")]]

def get_transforms(train=True):
    ops = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.8,1.2), translate_percent=0.1,
                     rotate=(-15,15), p=0.4),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(20, 30, 20, p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.2),
            A.GaussNoise(p=0.2),
            A.CLAHE(p=0.2),
            A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,32),
                            hole_width_range=(8,32), p=0.2),
        ]
    ops += [
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ]
    return A.Compose(ops, bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"],
        min_visibility=0.3))

class ShipDataset(Dataset):
    def __init__(self, img_paths, transform=None, mosaic_prob=0.5):
        self.img_paths   = img_paths
        self.transform   = transform
        self.mosaic_prob = mosaic_prob

    def __len__(self): return len(self.img_paths)

    def _load(self, idx):
        p = self.img_paths[idx]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        xml_p = os.path.join(ANN_DIR, Path(p).stem + ".xml")
        boxes = parse_voc_xml(xml_p) if os.path.exists(xml_p) else []
        return img, boxes

    def _mosaic(self, idx):
        s  = IMG_SIZE
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        all_boxes = []
        indices = [idx] + random.choices(range(len(self)), k=3)
        quads = [(0,0,s//2,s//2),(s//2,0,s,s//2),(0,s//2,s//2,s),(s//2,s//2,s,s)]
        for i, (x1,y1,x2,y2) in zip(indices, quads):
            img, boxes = self._load(i)
            ph, pw = y2-y1, x2-x1
            h, w   = img.shape[:2]
            rsz    = cv2.resize(img, (pw, ph))
            canvas[y1:y2, x1:x2] = rsz
            sx, sy = pw/w, ph/h
            for (bx1,by1,bx2,by2) in boxes:
                all_boxes.append([bx1*sx+x1, by1*sy+y1,
                                   bx2*sx+x1, by2*sy+y1])
        return canvas, all_boxes

    def __getitem__(self, idx):
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            img, boxes = self._mosaic(idx)
        else:
            img, boxes = self._load(idx)

        if self.transform:
            res   = self.transform(image=img,
                                   bboxes=boxes if boxes else [],
                                   class_labels=[0]*len(boxes))
            img   = res["image"]
            boxes = list(res["bboxes"])

        targets = []
        for (xmin,ymin,xmax,ymax) in boxes:
            cx = (xmin+xmax)/(2*IMG_SIZE); cy = (ymin+ymax)/(2*IMG_SIZE)
            bw = (xmax-xmin)/IMG_SIZE;     bh = (ymax-ymin)/IMG_SIZE
            if bw > 0.005 and bh > 0.005:
                targets.append([0, cx, cy, bw, bh])
        targets = torch.tensor(targets, dtype=torch.float32) \
                  if targets else torch.zeros((0,5))
        return img, targets

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    labeled = []
    for i, t in enumerate(targets):
        if t.shape[0]:
            labeled.append(torch.cat([torch.full((len(t),1), i), t], 1))
    targets = torch.cat(labeled, 0) if labeled else torch.zeros((0,6))
    return imgs, targets

# ──────────────────────────────────────────────────────────────
# MODEL — YOLOv5s with Vitis AI 3.0 / VCK190 constraints
#
# Key design rules:
#   1. Conv → BN → LeakyReLU order (BN fusion requires this)
#   2. LeakyReLU(0.1) only — SiLU not supported on DPU
#   3. Nearest-neighbour upsample only
#   4. No in-place ops that confuse the quantizer's graph tracer
#   5. Detection head final Conv has NO activation (sigmoid applied
#      outside the model at inference so DPU subgraph is clean)
# ──────────────────────────────────────────────────────────────

def _lrelu(): return nn.LeakyReLU(0.1, inplace=False)   # inplace=False for quantizer

def ConvBnAct(cin, cout, k=1, s=1, p=None, groups=1):
    p = k//2 if p is None else p
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        _lrelu())

class Bottleneck(nn.Module):
    """YOLOv5-style bottleneck: shortcut only when cin==cout."""
    def __init__(self, c, shortcut=True, e=0.5):
        super().__init__()
        h = int(c * e)
        self.cv1 = ConvBnAct(c, h, 1)
        self.cv2 = ConvBnAct(h, c, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """Cross Stage Partial bottleneck (YOLOv5 C3 block)."""
    def __init__(self, cin, cout, n=1, shortcut=True):
        super().__init__()
        h = cout // 2
        self.cv1 = ConvBnAct(cin,  h, 1)
        self.cv2 = ConvBnAct(cin,  h, 1)
        self.cv3 = ConvBnAct(2*h, cout, 1)
        self.m   = nn.Sequential(*[Bottleneck(h, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling – Fast (replaces original SPP)."""
    def __init__(self, cin, cout, k=5):
        super().__init__()
        h = cin // 2
        self.cv1 = ConvBnAct(cin,  h, 1)
        self.cv2 = ConvBnAct(4*h, cout, 1)
        self.pool = nn.MaxPool2d(k, 1, k//2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], 1))

class YOLOv5(nn.Module):
    """
    YOLOv5s architecture, Vitis AI 3.0 compatible.
    Outputs a list of 3 raw feature maps (one per scale).
    Sigmoid / decode is intentionally kept OUTSIDE the model
    so the DPU subgraph ends at the final Conv layer.
    """
    def __init__(self, num_cls=1, depth=0.33, width=0.50):
        super().__init__()
        na = 3   # anchors per scale
        def c(x): return max(round(x*width), 1)
        def n(x): return max(round(x*depth), 1)

        # ── Backbone ─────────────────────────────────────────
        self.b0  = ConvBnAct(3,     c(64),  6, 2, 2)   # P1/2
        self.b1  = ConvBnAct(c(64), c(128), 3, 2)       # P2/4
        self.b2  = C3(c(128), c(128), n(3))
        self.b3  = ConvBnAct(c(128), c(256), 3, 2)      # P3/8
        self.b4  = C3(c(256), c(256), n(6))              # → route1
        self.b5  = ConvBnAct(c(256), c(512), 3, 2)      # P4/16
        self.b6  = C3(c(512), c(512), n(9))              # → route2
        self.b7  = ConvBnAct(c(512), c(1024), 3, 2)     # P5/32
        self.b8  = C3(c(1024), c(1024), n(3))
        self.b9  = SPPF(c(1024), c(1024))                # → route3

        # ── Neck (PANet) ──────────────────────────────────────
        # Top-down
        self.n0  = ConvBnAct(c(1024), c(512), 1)          # reduce P5
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n1  = C3(c(512)+c(512), c(512), n(3), shortcut=False)   # fuse P4

        self.n2  = ConvBnAct(c(512), c(256), 1)           # reduce fused-P4
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n3  = C3(c(256)+c(256), c(256), n(3), shortcut=False)   # P3 out → det1

        # Bottom-up
        self.n4  = ConvBnAct(c(256), c(256), 3, 2)        # downsample P3→P4
        self.n5  = C3(c(256)+c(512), c(512), n(3), shortcut=False)   # P4 out → det2

        self.n6  = ConvBnAct(c(512), c(512), 3, 2)        # downsample P4→P5
        self.n7  = C3(c(512)+c(1024), c(1024), n(3), shortcut=False) # P5 out → det3

        # ── Detection heads ───────────────────────────────────
        # Final Conv: NO activation — sigmoid applied outside model
        out = na * (5 + num_cls)
        self.det1 = nn.Conv2d(c(256),  out, 1)   # P3  small
        self.det2 = nn.Conv2d(c(512),  out, 1)   # P4  medium
        self.det3 = nn.Conv2d(c(1024), out, 1)   # P5  large

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # backbone
        x  = self.b0(x)
        x  = self.b1(x)
        x  = self.b2(x)
        x  = self.b3(x)
        p3 = self.b4(x)           # 80×80 @ 640
        x  = self.b5(p3)
        p4 = self.b6(x)           # 40×40
        x  = self.b7(p4)
        x  = self.b8(x)
        p5 = self.b9(x)           # 20×20

        # neck — top-down path
        td0 = self.n0(p5)                          # 1024→512, 20×20
        td0_up = self.up1(td0)                     # 512, 40×40
        td1 = self.n1(torch.cat([td0_up, p4], 1)) # 512+512→512, 40×40

        td1r = self.n2(td1)                        # 512→256, 40×40
        td1_up = self.up2(td1r)                    # 256, 80×80
        f_s = self.n3(torch.cat([td1_up, p3], 1)) # 256+256→256, 80×80  (small)

        # bottom-up path
        bu0 = self.n4(f_s)                         # 256, 40×40
        f_m = self.n5(torch.cat([bu0, td1], 1))   # 256+512→512, 40×40  (medium)

        bu1 = self.n6(f_m)                         # 512, 20×20
        f_l = self.n7(torch.cat([bu1, p5], 1))    # 512+1024→1024, 20×20 (large)

        # heads — raw logits, NO sigmoid here
        return [self.det1(f_s),
                self.det2(f_m),
                self.det3(f_l)]

# ──────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────
def iou_wh(wh1, wh2):
    inter = torch.min(wh1[:,None], wh2[None]).prod(-1)
    union = wh1.prod(-1)[:,None] + wh2.prod(-1)[None] - inter
    return inter / (union + 1e-8)

class FocalBCE(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha

    def forward(self, pred, tgt):
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        p   = torch.exp(-bce)
        return (self.alpha * (1-p)**self.gamma * bce).mean()

class YOLOLoss(nn.Module):
    def __init__(self, anchors, stride, img_size, num_cls):
        super().__init__()
        self.anchors  = torch.tensor(anchors, dtype=torch.float32)
        self.stride   = stride
        self.img_size = img_size
        self.num_cls  = num_cls
        self.na       = len(anchors)
        self.obj_loss = FocalBCE(gamma=1.5)
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.box_loss = nn.MSELoss()

    def forward(self, pred, targets):
        B, _, gs, _ = pred.shape[:4] if pred.dim()==4 else (pred.shape[0],None,pred.shape[2],None)
        gs   = pred.shape[-1]
        dev  = pred.device
        anc  = self.anchors.to(dev) / self.stride

        p = pred.view(B, self.na, 5+self.num_cls, gs, gs
                      ).permute(0,1,3,4,2).contiguous()

        tobj = torch.zeros(B, self.na, gs, gs, device=dev)
        txy  = torch.zeros(B, self.na, gs, gs, 2, device=dev)
        twh  = torch.zeros(B, self.na, gs, gs, 2, device=dev)
        tcls = torch.zeros(B, self.na, gs, gs, self.num_cls, device=dev)
        mask = torch.zeros(B, self.na, gs, gs, dtype=torch.bool, device=dev)

        if targets.shape[0]:
            t   = targets.to(dev)
            gxy = t[:,2:4] * gs
            gwh = t[:,4:6] * gs
            ba  = iou_wh(gwh, anc).argmax(1)
            bi  = t[:,0].long()
            gi  = gxy[:,0].long().clamp(0, gs-1)
            gj  = gxy[:,1].long().clamp(0, gs-1)
            mask[bi, ba, gj, gi] = True
            tobj[bi, ba, gj, gi] = 1.
            txy [bi, ba, gj, gi] = gxy - gxy.floor()
            twh [bi, ba, gj, gi] = torch.log(gwh / anc[ba] + 1e-8)
            tcls[bi, ba, gj, gi, t[:,1].long()] = 1.

        lo = self.obj_loss(p[...,4], tobj)
        if mask.any():
            lxy = self.box_loss(torch.sigmoid(p[...,0:2][mask]), txy[mask])
            lwh = self.box_loss(p[...,2:4][mask], twh[mask])
            lc  = self.cls_loss(p[...,5:][mask], tcls[mask])
        else:
            lxy = lwh = lc = torch.tensor(0., device=dev)
        return lo + lxy + lwh + lc

# ──────────────────────────────────────────────────────────────
# DECODE + NMS  (post-processing, runs on CPU/host, not DPU)
# Note: sigmoid applied HERE, not inside the model graph,
# so the DPU subgraph stays clean up to the last Conv layer.
# ──────────────────────────────────────────────────────────────
def decode_predictions(preds, anchors_list, strides, conf_thr=0.4, iou_thr=0.45):
    all_boxes = []
    for pred, ancs, stride in zip(preds, anchors_list, strides):
        B, _, gs, _ = pred.shape
        na  = len(ancs)
        anc = torch.tensor(ancs, dtype=torch.float32, device=pred.device)
        p   = pred.view(B, na, -1, gs, gs).permute(0,1,3,4,2).contiguous()

        g   = torch.arange(gs, device=pred.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        # sigmoid applied here (outside model for Vitis AI compatibility)
        xy  = (torch.sigmoid(p[...,0:2]) + torch.stack([gx,gy],-1)) * stride
        wh  = torch.exp(p[...,2:4]) * anc[:,None,None,:]
        obj = torch.sigmoid(p[...,4:5])
        cls = torch.sigmoid(p[...,5:])
        all_boxes.append(torch.cat([xy, wh, obj, cls], -1).view(B,-1,5+cls.shape[-1]))

    dets = torch.cat(all_boxes, 1)   # (B, N, 5+nc)
    results = []
    for b in range(dets.shape[0]):
        d = dets[b]
        mask = d[:,4] > conf_thr
        d = d[mask]
        if not d.shape[0]:
            results.append([]); continue
        # xyxy
        x1 = d[:,0] - d[:,2]/2; y1 = d[:,1] - d[:,3]/2
        x2 = d[:,0] + d[:,2]/2; y2 = d[:,1] + d[:,3]/2
        boxes  = torch.stack([x1,y1,x2,y2],-1)
        scores = d[:,4]
        # NMS
        keep = torchvision_nms(boxes, scores, iou_thr)
        results.append([(boxes[k].tolist() + [scores[k].item()]) for k in keep])
    return results

def torchvision_nms(boxes, scores, iou_thr):
    if not boxes.shape[0]: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel():
        i = order[0].item(); keep.append(i)
        if order.numel()==1: break
        rest = order[1:]
        ix1 = x1[rest].clamp(min=x1[i]); iy1 = y1[rest].clamp(min=y1[i])
        ix2 = x2[rest].clamp(max=x2[i]); iy2 = y2[rest].clamp(max=y2[i])
        inter = (ix2-ix1).clamp(0)*(iy2-iy1).clamp(0)
        iou   = inter/(areas[i]+areas[rest]-inter+1e-8)
        order = rest[iou<=iou_thr]
    return keep

# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────
def train():
    print(f"Training on: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: GPU not detected. Training will be slow.")

    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    assert len(all_imgs), f"No images found in {IMG_DIR}"
    random.seed(42); random.shuffle(all_imgs)
    split = int(0.85 * len(all_imgs))
    tr_imgs, va_imgs = all_imgs[:split], all_imgs[split:]
    print(f"Train: {len(tr_imgs)}  Val: {len(va_imgs)}")

    tr_ds = ShipDataset(tr_imgs, get_transforms(True),  mosaic_prob=0.5)
    va_ds = ShipDataset(va_imgs, get_transforms(False), mosaic_prob=0.0)
    tr_dl = DataLoader(tr_ds, BATCH, shuffle=True,  collate_fn=collate_fn,
                       num_workers=4, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                       num_workers=4, pin_memory=True)

    model = YOLOv5(NUM_CLS).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())/1e6
    print(f"YOLOv5s params: {total:.1f}M")

    # Separate backbone / head LRs
    backbone_ids = set(id(p) for p in list(model.b0.parameters()) +
                                       list(model.b1.parameters()) +
                                       list(model.b2.parameters()) +
                                       list(model.b3.parameters()) +
                                       list(model.b4.parameters()) +
                                       list(model.b5.parameters()) +
                                       list(model.b6.parameters()) +
                                       list(model.b7.parameters()) +
                                       list(model.b8.parameters()) +
                                       list(model.b9.parameters()))
    backbone_p = [p for p in model.parameters() if id(p) in backbone_ids]
    head_p     = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = optim.AdamW([{"params": backbone_p, "lr": LR*0.1},
                       {"params": head_p,     "lr": LR}],
                      weight_decay=1e-4)

    def lr_fn(ep):
        warmup = 3
        if ep < warmup: return (ep+1)/warmup
        t = (ep-warmup)/(EPOCHS-warmup)
        return 0.5*(1+math.cos(math.pi*t))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    criteria = [YOLOLoss(ANCHORS[i], STRIDES[i], IMG_SIZE, NUM_CLS)
                for i in range(3)]

    best = math.inf
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss = 0
        for imgs, tgts in tqdm(tr_dl, desc=f"Ep {ep:3d}/{EPOCHS} train"):
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            loss = sum(c(p, tgts) for c,p in zip(criteria, out))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()

        model.eval(); va_loss = 0
        with torch.no_grad():
            for imgs, tgts in tqdm(va_dl, desc=f"Ep {ep:3d}/{EPOCHS} val"):
                imgs = imgs.to(DEVICE)
                out  = model(imgs)
                va_loss += sum(c(p,tgts) for c,p in zip(criteria,out)).item()

        tr_loss /= len(tr_dl); va_loss /= len(va_dl)
        sched.step()
        lr_now = opt.param_groups[1]["lr"]
        print(f"Ep {ep:3d} | tr={tr_loss:.4f}  va={va_loss:.4f}  lr={lr_now:.6f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), CKPT)
            print(f"  ✓ checkpoint saved (val={best:.4f})")

# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────
def draw_boxes(img_path, model=None, save_path="result.png",
               conf_thr=0.4, iou_thr=0.45):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    if model is None:
        model = YOLOv5(NUM_CLS).to(DEVICE)
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read: {img_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    oh, ow = img.shape[:2]

    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    inp  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.
    inp  = ((inp - mean)/std).transpose(2,0,1)
    inp  = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(inp)
    results = decode_predictions(preds, ANCHORS, STRIDES, conf_thr, iou_thr)
    boxes = results[0]

    sx, sy = ow/IMG_SIZE, oh/IMG_SIZE
    for (x1,y1,x2,y2,conf) in boxes:
        x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,f"ship {conf:.2f}",(x1,max(y1-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.imwrite(save_path, img)
    print(f"Saved {save_path}  ({len(boxes)} detections)")
    return img

# ──────────────────────────────────────────────────────────────
# EXPORT — saves a clean float32 TorchScript for Vitis AI
# Run this BEFORE quantization with vai_q_pytorch
# ──────────────────────────────────────────────────────────────
def export_torchscript(out_path="yolov5_ship_float.pt"):
    model = YOLOv5(NUM_CLS).to("cpu").eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(model, dummy)
    traced.save(out_path)
    print(f"TorchScript saved → {out_path}")
    print("Next step: run the Vitis AI Inspector on this file inside the")
    print("  vitis-ai-pytorch Docker container:")
    print(f"  python -c \"import torch; from pytorch_nndct.apis import Inspector; "
          f"m=torch.jit.load('{out_path}'); Inspector('DPUCVDX8G_ISA3_C32B6').inspect(m, (dummy,))\"")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",   default="train",
                    choices=["train","infer","export","anchors"])
    ap.add_argument("--image",  default=None)
    ap.add_argument("--out",    default="result.png")
    ap.add_argument("--conf",   type=float, default=0.4)
    ap.add_argument("--nms",    type=float, default=0.45)
    args = ap.parse_args()

    if   args.mode == "train":   train()
    elif args.mode == "infer":
        assert args.image, "Provide --image"
        draw_boxes(args.image, save_path=args.out, conf_thr=args.conf, iou_thr=args.nms)
    elif args.mode == "export":  export_torchscript()
    elif args.mode == "anchors": anchor_kmeans()