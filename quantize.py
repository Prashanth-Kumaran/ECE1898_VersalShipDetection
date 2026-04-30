"""
quantize.py
===========
Calibration and quantization script for Vitis AI 3.0 / VCK190.

Requirements (run inside Vitis AI 3.0 Docker container):
  - float_model.py     in the same directory
  - yolov5_ship_best.pth in the same directory
  - calibration images in data/images/

Run:
  python quantize.py --subset 500
"""

import os, glob, random, argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from float_model import build_model, IMG_SIZE, NUM_CLS, CKPT

# ──────────────────────────────────────────────────────────────
# CALIBRATION DATASET
# ──────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class CalibDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = img_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1))  # CHW float32


def get_calib_loader(img_dir="data/images", subset=500, batch_size=8):
    all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert len(all_imgs), f"No images found in {img_dir}"
    random.seed(42)
    random.shuffle(all_imgs)
    calib_imgs = all_imgs[:min(subset, len(all_imgs))]
    print(f"Calibration images: {len(calib_imgs)}")
    ds = CalibDataset(calib_imgs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ──────────────────────────────────────────────────────────────
# QUANTIZE
# ──────────────────────────────────────────────────────────────
def quantize(img_dir="data/images", subset=500, output_dir="quantized"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")   # Vitis AI quantizer runs on CPU

    # 1. Load float model
    model = build_model(CKPT, device=str(device), eval_mode=True)

    # 2. Dummy input — defines input shape for the quantizer
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    # 3. Create quantizer
    from pytorch_nndct.apis import torch_quantizer
    quantizer = torch_quantizer(
        quant_mode  = "calib",           # calibration pass
        module      = model,
        input_args  = (dummy,),
        output_dir  = output_dir,
        device      = device,
        quant_config_file = None,
    )
    quant_model = quantizer.quant_model

    # 4. Run calibration — feed images through quantized model
    print("Running calibration...")
    calib_loader = get_calib_loader(img_dir, subset)
    quant_model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(calib_loader):
            quant_model(imgs.to(device))
            if (i+1) % 10 == 0:
                print(f"  Calibrated {(i+1)*imgs.shape[0]}/{len(calib_loader.dataset)} images")

    # 5. Export quantization config
    quantizer.export_quant_config()
    print(f"\nCalibration complete. Quant config saved to: {output_dir}/")

    # 6. Test pass — evaluate quantized model on a few images
    print("\nRunning test pass...")
    quantizer_test = torch_quantizer(
        quant_mode  = "test",
        module      = model,
        input_args  = (dummy,),
        output_dir  = output_dir,
        device      = device,
    )
    quant_model_test = quantizer_test.quant_model
    quant_model_test.eval()
    with torch.no_grad():
        test_out = quant_model_test(dummy)
    print("Test pass output shapes:",
          [tuple(o.shape) for o in test_out])

    # 7. Export quantized model for compilation
    quantizer_test.export_quant_config()
    try:
        quantizer_test.export_xmodel(output_dir=output_dir, deploy_check=False)
        print(f"\nExported xmodel to: {output_dir}/")
    except Exception as e:
        print(f"\nexport_xmodel failed: {e}")
        print("Try exporting manually with:")
        print("  quantizer_test.export_quant_config()")
        print("  # then run vai_c_xir on the generated quant config")
    print("\nNext step — compile for VCK190:")
    print(f"  vai_c_xir -x {output_dir}/YOLOv5_int.xmodel \\")
    print(f"            -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \\")
    print(f"            -o compiled/ -n yolov5_ship")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--subset",  type=int, default=500,
                    help="Number of calibration images (100-1000 recommended)")
    ap.add_argument("--output",  default="quantized")
    ap.add_argument("--batch",   type=int, default=8)
    args = ap.parse_args()

    quantize(args.img_dir, args.subset, args.output)