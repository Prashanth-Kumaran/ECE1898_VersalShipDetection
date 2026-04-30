"""
float_model.py
==============
Self-contained float32 YOLOv5s model definition for Vitis AI 3.0 / VCK190.

This file is intentionally standalone — it does NOT import from yolov5_ship.py
so it can be dropped directly into the Vitis AI Docker container.

Vitis AI 3.0 / VCK190 (DPUCVDX8G) design constraints enforced here:
  - LeakyReLU(0.1) only          — SiLU/Mish not supported by DPU
  - Conv -> BN -> Activation      — required for BN fusion in compiler
  - Nearest-neighbour upsample    — bilinear not DPU-friendly
  - inplace=False on all ops      — required for quantizer graph tracing
  - No dynamic shapes             — fixed IMG_SIZE=640
  - Sigmoid applied OUTSIDE model — DPU subgraph ends cleanly at final Conv
  - No custom ops or torch.jit    — standard nn.Module only

Usage:
  # Load weights and export TorchScript for Vitis AI Inspector
  python float_model.py

  # Or import in your quantization script:
  from float_model import build_model, IMG_SIZE, NUM_CLS
"""

import math
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# CONFIG  (must match yolov5_ship.py)
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 640
NUM_CLS  = 1      # ship only
CKPT     = "yolov5_ship_best.pth"

# Anchors — paste your k-means anchors here (must match training)
ANCHORS = [
    [(10, 11),  (24, 28),   (40, 46)],    # P3/8   small
    [(60, 70),  (85, 100),  (120, 140)],  # P4/16  medium
    [(170, 180),(203, 210), (396, 448)],  # P5/32  large
]
STRIDES = [8, 16, 32]

# ──────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# All follow Conv -> BN -> LeakyReLU for DPU BN fusion.
# inplace=False required for Vitis AI graph tracer.
# ──────────────────────────────────────────────────────────────

def _lrelu():
    """LeakyReLU(0.1) — the only activation supported by DPUCVDX8G."""
    return nn.LeakyReLU(0.1, inplace=False)


def ConvBnAct(cin, cout, k=1, s=1, p=None, groups=1):
    """Conv -> BN -> LeakyReLU. Core building block."""
    p = k // 2 if p is None else p
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        _lrelu())


class Bottleneck(nn.Module):
    """
    YOLOv5-style CSP bottleneck.
    shortcut=True adds residual connection (only when cin==cout).
    """
    def __init__(self, c, shortcut=True, e=0.5):
        super().__init__()
        h = int(c * e)
        self.cv1 = ConvBnAct(c, h, 1)
        self.cv2 = ConvBnAct(h, c, 3)
        self.add = shortcut

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    """
    Cross Stage Partial (CSP) block — YOLOv5's main feature extraction unit.
    Splits input into two branches, processes one with n bottlenecks,
    then concatenates and projects back.
    """
    def __init__(self, cin, cout, n=1, shortcut=True):
        super().__init__()
        h = cout // 2
        self.cv1 = ConvBnAct(cin,  h, 1)
        self.cv2 = ConvBnAct(cin,  h, 1)
        self.cv3 = ConvBnAct(2*h, cout, 1)
        self.m   = nn.Sequential(
            *[Bottleneck(h, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast.
    Applies MaxPool at 3 scales and concatenates — gives multi-scale
    receptive field without extra conv layers.
    Uses MaxPool (DPU-supported) not AvgPool.
    """
    def __init__(self, cin, cout, k=5):
        super().__init__()
        h = cin // 2
        self.cv1  = ConvBnAct(cin,  h, 1)
        self.cv2  = ConvBnAct(4*h, cout, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], 1))


# ──────────────────────────────────────────────────────────────
# YOLOV5s FULL MODEL
# ──────────────────────────────────────────────────────────────

class YOLOv5(nn.Module):
    """
    YOLOv5s — float32 model for Vitis AI 3.0 quantization.

    Architecture: CSPDarknet53 backbone + PANet neck + detection heads.
    depth=0.33, width=0.50 gives the 's' (small) variant (~7M params).

    IMPORTANT — sigmoid is NOT applied inside the model.
    The DPU subgraph must end at the final Conv layer.
    Apply sigmoid in post-processing on the host ARM core.

    Output:
        List of 3 raw tensors:
          [0] (B, 3*(5+nc), 80, 80)  — small objects  (P3, stride 8)
          [1] (B, 3*(5+nc), 40, 40)  — medium objects (P4, stride 16)
          [2] (B, 3*(5+nc), 20, 20)  — large objects  (P5, stride 32)
        Each output channel layout per anchor:
          [tx, ty, tw, th, obj_logit, cls_logit_0, ...]
    """

    def __init__(self, num_cls=NUM_CLS, depth=0.33, width=0.50):
        super().__init__()

        def c(x): return max(round(x * width), 1)   # channel scaling
        def n(x): return max(round(x * depth), 1)   # depth scaling

        na  = 3                        # anchors per scale
        out = na * (5 + num_cls)       # output channels per head

        # ── Backbone (CSPDarknet) ────────────────────────────
        # Input: (B, 3, 640, 640)
        self.b0 = ConvBnAct(3,      c(64),  6, 2, 2)  # → (B, 32, 320, 320)  P1/2
        self.b1 = ConvBnAct(c(64),  c(128), 3, 2)     # → (B, 64, 160, 160)  P2/4
        self.b2 = C3(c(128), c(128), n(3))             # → (B, 64, 160, 160)
        self.b3 = ConvBnAct(c(128), c(256), 3, 2)     # → (B,128,  80,  80)  P3/8
        self.b4 = C3(c(256), c(256), n(6))             # → (B,128,  80,  80)  route1
        self.b5 = ConvBnAct(c(256), c(512), 3, 2)     # → (B,256,  40,  40)  P4/16
        self.b6 = C3(c(512), c(512), n(9))             # → (B,256,  40,  40)  route2
        self.b7 = ConvBnAct(c(512), c(1024), 3, 2)    # → (B,512,  20,  20)  P5/32
        self.b8 = C3(c(1024), c(1024), n(3))           # → (B,512,  20,  20)
        self.b9 = SPPF(c(1024), c(1024))               # → (B,512,  20,  20)  route3

        # ── Neck (PANet — top-down path) ─────────────────────
        self.n0  = ConvBnAct(c(1024), c(512), 1)       # reduce P5: 512→256
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")   # 20→40
        self.n1  = C3(c(512)+c(512), c(512), n(3), shortcut=False)  # fuse P4

        self.n2  = ConvBnAct(c(512), c(256), 1)        # reduce: 256→128
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")   # 40→80
        self.n3  = C3(c(256)+c(256), c(256), n(3), shortcut=False)  # fuse P3 → det1

        # ── Neck (PANet — bottom-up path) ────────────────────
        self.n4  = ConvBnAct(c(256), c(256), 3, 2)     # downsample 80→40
        self.n5  = C3(c(256)+c(512), c(512), n(3), shortcut=False)  # fuse → det2

        self.n6  = ConvBnAct(c(512), c(512), 3, 2)     # downsample 40→20
        self.n7  = C3(c(512)+c(1024), c(1024), n(3), shortcut=False) # fuse → det3

        # ── Detection heads ───────────────────────────────────
        # Final Conv: NO activation — sigmoid applied outside on host CPU/ARM.
        # This ensures the DPU subgraph ends cleanly at the last Conv layer,
        # which is required for Vitis AI 3.0 compilation.
        self.det1 = nn.Conv2d(c(256),  out, 1)   # P3 → small  objects
        self.det2 = nn.Conv2d(c(512),  out, 1)   # P4 → medium objects
        self.det3 = nn.Conv2d(c(1024), out, 1)   # P5 → large  objects

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.eps = 1e-3      # Vitis AI recommended BN epsilon
                m.momentum = 0.03

    def forward(self, x):
        # ── Backbone ──────────────────────────────────────────
        x  = self.b0(x)
        x  = self.b1(x)
        x  = self.b2(x)
        x  = self.b3(x)
        p3 = self.b4(x)    # small  feature map  80×80
        x  = self.b5(p3)
        p4 = self.b6(x)    # medium feature map  40×40
        x  = self.b7(p4)
        x  = self.b8(x)
        p5 = self.b9(x)    # large  feature map  20×20

        # ── Neck — top-down ───────────────────────────────────
        td0    = self.n0(p5)                           # 512→256, 20×20
        td0_up = self.up1(td0)                         # 256,     40×40
        td1    = self.n1(torch.cat([td0_up, p4], 1))  # 256+512→512, 40×40

        td1r   = self.n2(td1)                          # 512→256, 40×40
        td1_up = self.up2(td1r)                        # 256,     80×80
        f_s    = self.n3(torch.cat([td1_up, p3], 1))  # 256+256→256, 80×80

        # ── Neck — bottom-up ──────────────────────────────────
        bu0 = self.n4(f_s)                             # 256,     40×40
        f_m = self.n5(torch.cat([bu0, td1], 1))       # 256+512→512, 40×40

        bu1 = self.n6(f_m)                             # 512,     20×20
        f_l = self.n7(torch.cat([bu1, p5], 1))        # 512+1024→1024, 20×20

        # ── Detection heads (raw logits, no sigmoid) ──────────
        return [self.det1(f_s),   # (B, 3*(5+nc), 80, 80)
                self.det2(f_m),   # (B, 3*(5+nc), 40, 40)
                self.det3(f_l)]   # (B, 3*(5+nc), 20, 20)


# ──────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────

def build_model(ckpt=CKPT, device="cpu", eval_mode=True):
    """
    Build and return the float32 model with trained weights loaded.

    Args:
        ckpt     : path to .pth checkpoint from training
        device   : 'cpu' recommended for Vitis AI quantization
        eval_mode: True sets BN to eval mode (required before quantization)

    Returns:
        nn.Module in float32, ready for Vitis AI Inspector / quantizer
    """
    model = YOLOv5(NUM_CLS).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    print(f"Loaded weights from: {ckpt}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


# ──────────────────────────────────────────────────────────────
# VERIFY + EXPORT
# ──────────────────────────────────────────────────────────────

def verify_output_shapes(model, device="cpu"):
    """Sanity check — confirms output shapes are correct for all 3 heads."""
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        outs = model(dummy)
    na, nc = 3, NUM_CLS
    expected = [
        (1, na*(5+nc), IMG_SIZE//8,  IMG_SIZE//8),   # 80×80
        (1, na*(5+nc), IMG_SIZE//16, IMG_SIZE//16),  # 40×40
        (1, na*(5+nc), IMG_SIZE//32, IMG_SIZE//32),  # 20×20
    ]
    print("\nOutput shape verification:")
    all_ok = True
    for i, (out, exp) in enumerate(zip(outs, expected)):
        ok = tuple(out.shape) == exp
        all_ok = all_ok and ok
        status = "✓" if ok else "✗"
        print(f"  Head {i+1}: {tuple(out.shape)}  expected {exp}  {status}")
    print(f"  {'All shapes correct' if all_ok else 'SHAPE MISMATCH — check model'}")
    return all_ok


def export_torchscript(model, out_path="yolov5_ship_float.pt", device="cpu"):
    """
    Export to TorchScript for Vitis AI Inspector.

    Run inside the Vitis AI Docker container:
      python -c "
        from pytorch_nndct.apis import Inspector
        import torch
        model = torch.jit.load('yolov5_ship_float.pt')
        dummy = torch.zeros(1, 3, 640, 640)
        Inspector('DPUCVDX8G_ISA3_C32B6').inspect(model, (dummy,), image_format='float')
      "
    """
    model.eval()
    dummy  = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    traced = torch.jit.trace(model, dummy)
    traced.save(out_path)
    print(f"\nTorchScript saved → {out_path}")
    print("\nNext steps for Vitis AI 3.0:")
    print("  1. Copy yolov5_ship_float.pt into the Vitis AI Docker container")
    print("  2. Run Inspector to check DPU compatibility:")
    print("       from pytorch_nndct.apis import Inspector")
    print("       Inspector('DPUCVDX8G_ISA3_C32B6').inspect(model, (dummy,))")
    print("  3. Run quantizer:")
    print("       from pytorch_nndct.apis import torch_quantizer")
    print("       quantizer = torch_quantizer('calib', model, (dummy,))")
    print("       # feed ~100-1000 calibration images through quantizer.quant_model")
    print("       quantizer.export_quant_config()")
    print("  4. Compile with vai_c_xir for VCK190 deployment")
    return traced


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    device = "cpu"   # always use CPU for Vitis AI quantization prep
    print("=" * 55)
    print("  YOLOv5s Float Model — Vitis AI 3.0 / VCK190")
    print("=" * 55)

    if not os.path.exists(CKPT):
        print(f"\nNo checkpoint found at '{CKPT}'.")
        print("Building model with random weights for shape verification only.")
        model = YOLOv5(NUM_CLS).to(device).eval()
    else:
        model = build_model(CKPT, device=device)

    # verify shapes
    verify_output_shapes(model, device)

    # export TorchScript
    export_torchscript(model, "yolov5_ship_float.pt", device)