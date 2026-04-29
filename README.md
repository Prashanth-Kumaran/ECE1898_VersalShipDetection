# Ship Detection — YOLOv5 Training Pipeline
### Target Deployment: Xilinx VCK-190 (Vitis-AI 3.0)

This repository covers **Stage 1**: PC-based training and validation of a YOLOv5s ship detection model using PyTorch. Quantization (PTQ/QAT via Vitis-AI) is handled in Stage 2.

---

## Directory Structure

```
ship_detection/
├── README.md
├── requirements.txt
├── configs/
│   └── ship.yaml              # Dataset config for YOLOv5
├── scripts/
│   ├── 01_download_dataset.sh # Kaggle download instructions
│   ├── 02_convert_voc_to_yolo.py  # Pascal VOC XML → YOLOv5 label format
│   ├── 03_split_dataset.py    # Train/val/test split
│   ├── 04_train.sh            # Launch training
│   ├── 05_validate.sh         # Run validation on val set
│   ├── 06_test_inference.py   # Run inference + save visual results
│   └── 07_export_torchscript.sh  # Export for pre-quantization inspection
├── tools/
│   ├── visualize_labels.py    # Sanity-check: draw boxes on images
│   └── dataset_stats.py       # Class distribution, image size stats
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── runs/                      # YOLOv5 training outputs land here
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
cd ..
```

### 3. Download Dataset
Follow `scripts/01_download_dataset.sh` — you need a Kaggle API token.

### 4. Convert & Split
```bash
python scripts/02_convert_voc_to_yolo.py
python scripts/03_split_dataset.py
```

### 5. Sanity Check Labels
```bash
python tools/visualize_labels.py
python tools/dataset_stats.py
```

### 6. Train
```bash
bash scripts/04_train.sh
```

### 7. Validate & Test
```bash
bash scripts/05_validate.sh
python scripts/06_test_inference.py
```

### 8. Export (pre-quantization)
```bash
bash scripts/07_export_torchscript.sh
```

---

## Model Choice: YOLOv5s

| Model    | Params | GFLOPs | Why |
|----------|--------|--------|-----|
| YOLOv5n  | 1.9M   | 4.5    | Too small — lower mAP for small ships |
| **YOLOv5s** | **7.2M** | **16.5** | **Best tradeoff for VCK-190 DPU** |
| YOLOv5m  | 21.2M  | 49.0   | Larger, may not fit DPU budget |

YOLOv5s is the recommended starting point. Vitis-AI's DPUCVDX8G (VCK-190) handles it well, and the model size keeps quantization error manageable.

---

## Vitis-AI Notes (Stage 2 Prep)

- Train with **input size 640×640** — this is the size you'll freeze for quantization
- Keep batch norm layers unfused during training (default YOLOv5 behavior ✓)
- Export to **TorchScript** before handing off to `vai_q_pytorch`
- Target: `DPUCVDX8G_ISA3_C32B6` for VCK-190

---

## Dataset: Kaggle Ship Detection

- **Source**: https://www.kaggle.com/datasets/andrewmvd/ship-detection
- **Format**: Pascal VOC XML annotations
- **Classes**: 1 (`ship`)
- **Image type**: Aerial/satellite imagery
- **~4000 images** with ship annotations

### Recommended Additional Datasets (if mAP plateaus)
- **HRSC2016** — high-res ship detection, rotated bounding boxes (use horizontal boxes only)
- **SAR-Ship** — SAR radar imagery ships (domain diversity)
- **DOTA v1.5** — large aerial dataset, filter `ship` + `harbor` classes
