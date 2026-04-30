"""
app_mt.py
=========
Ship detection application for Vitis AI 3.0 targets (VCK190, ZCU102, etc.)
Follows Vitis AI app_mt.py naming and structure conventions.

Run on board:
  python3 app_mt.py -m yolov5_ship.xmodel -i images/ -t 4
"""

import os, glob, math, argparse, threading, queue, time
import numpy as np
import cv2
import vart
import xir

# ──────────────────────────────────────────────────────────────
# CONFIG — must match training and quantization
# ──────────────────────────────────────────────────────────────
IMG_SIZE = 640
CONF_THR = 0.35
NMS_THR  = 0.30
NUM_CLS  = 1

ANCHORS = [
    [(10, 11),  (24, 28),   (40, 46)],    # P3/8   small
    [(60, 70),  (85, 100),  (120, 140)],  # P4/16  medium
    [(170, 180),(203, 210), (396, 448)],  # P5/32  large
]
STRIDES = [8, 16, 32]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ──────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────
def preprocess(img_path):
    """Load image and convert to int8 NHWC tensor for DPU."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    orig_h, orig_w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resz = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.
    norm = (resz - MEAN) / STD
    inp  = (norm * 127).clip(-128, 127).astype(np.int8)
    return inp, img, orig_h, orig_w   # inp is HWC

# ──────────────────────────────────────────────────────────────
# POST-PROCESSING
# ──────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))

def decode_output(raw, anchors, stride, scale):
    """
    Decode one DPU output tensor.
    raw   : (1, H, W, na*(5+nc)) int8 from VART
    scale : output tensor scale factor from xmodel
    """
    gs = raw.shape[1]
    na = len(anchors)
    nc = NUM_CLS
    out = raw[0].reshape(gs, gs, na, 5 + nc).astype(np.float32) * scale
    boxes = []
    for gy in range(gs):
        for gx in range(gs):
            for a, (aw, ah) in enumerate(anchors):
                tx, ty, tw, th = out[gy, gx, a, 0:4]
                obj_logit       = out[gy, gx, a, 4]
                cls_logit       = out[gy, gx, a, 5:]
                obj_conf  = sigmoid(np.array([obj_logit]))[0]
                cls_conf  = sigmoid(cls_logit).max()
                conf      = obj_conf * cls_conf
                if conf < CONF_THR:
                    continue
                cx = (sigmoid(np.array([tx]))[0] + gx) * stride
                cy = (sigmoid(np.array([ty]))[0] + gy) * stride
                w  = math.exp(float(tw)) * aw
                h  = math.exp(float(th)) * ah
                boxes.append([cx, cy, w, h, conf])
    return boxes

def nms(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: -x[4])
    keep  = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        def iou(a, b):
            ax1 = a[0]-a[2]/2; ay1 = a[1]-a[3]/2
            ax2 = a[0]+a[2]/2; ay2 = a[1]+a[3]/2
            bx1 = b[0]-b[2]/2; by1 = b[1]-b[3]/2
            bx2 = b[0]+b[2]/2; by2 = b[1]+b[3]/2
            ix  = max(0, min(ax2,bx2) - max(ax1,bx1))
            iy  = max(0, min(ay2,by2) - max(ay1,by1))
            inter = ix * iy
            union = a[2]*a[3] + b[2]*b[3] - inter + 1e-8
            return inter / union
        boxes = [b for b in boxes if iou(best, b) < NMS_THR]
    return keep

def draw_boxes(img, boxes, orig_h, orig_w):
    sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
    for (cx, cy, w, h, conf) in boxes:
        x1 = int((cx - w/2) * sx); y1 = int((cy - h/2) * sy)
        x2 = int((cx + w/2) * sx); y2 = int((cy + h/2) * sy)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"ship {conf:.2f}", (x1, max(y1-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

# ──────────────────────────────────────────────────────────────
# VART RUNNER SETUP
# ──────────────────────────────────────────────────────────────
def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    child_subgraphs = root.toposort_child_subgraph()
    return [cs for cs in child_subgraphs
            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

def get_output_scale(tensor):
    """Get fixed-point scale from xmodel tensor metadata."""
    try:
        fix_pos = tensor.get_attr("fix_point")
        return 2 ** (-fix_pos)
    except Exception:
        return 1.0 / 64.0   # safe default

# ──────────────────────────────────────────────────────────────
# WORKER THREAD  (multi-threaded inference)
# ──────────────────────────────────────────────────────────────
def run_dpu_worker(runner, img_paths, out_dir, result_list, thread_id):
    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    batch       = in_tensors[0].dims[0]

    # get output scales
    out_scales = [get_output_scale(t) for t in out_tensors]

    in_buf  = [np.zeros(t.dims, dtype=np.int8) for t in in_tensors]
    out_buf = [np.zeros(t.dims, dtype=np.int8) for t in out_tensors]

    i = 0
    while i < len(img_paths):
        chunk = img_paths[i:i+batch]
        orig_imgs, orig_shapes = [], []

        for b, path in enumerate(chunk):
            inp, orig_img, orig_h, orig_w = preprocess(path)
            in_buf[0][b] = inp
            orig_imgs.append(orig_img)
            orig_shapes.append((orig_h, orig_w))

        job_id = runner.execute_async(in_buf, out_buf)
        runner.wait(job_id)

        for b in range(len(chunk)):
            all_boxes = []
            # out_tensors are ordered by size: large→small or small→large
            # match by spatial size to anchor scale
            sorted_outs = sorted(zip(out_buf, out_scales),
                                 key=lambda x: x[0].shape[1], reverse=True)
            for (raw, scale), (anchors, stride) in zip(sorted_outs,
                                                        zip(ANCHORS, STRIDES)):
                single = raw[b:b+1]
                all_boxes += decode_output(single, anchors, stride, scale)

            detections = nms(all_boxes)
            orig_h, orig_w = orig_shapes[b]
            out_img = draw_boxes(orig_imgs[b], detections, orig_h, orig_w)

            fname   = os.path.basename(chunk[b])
            outpath = os.path.join(out_dir, f"result_{fname}")
            cv2.imwrite(outpath, out_img)
            result_list.append((chunk[b], len(detections)))
            print(f"[Thread {thread_id}] {fname}: {len(detections)} ships → {outpath}")

        i += batch

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",   required=True,
                    help="Path to compiled .xmodel")
    ap.add_argument("-i", "--image",   required=True,
                    help="Path to image file or folder of images")
    ap.add_argument("-t", "--threads", type=int, default=2,
                    help="Number of DPU runner threads")
    ap.add_argument("-o", "--output",  default="results",
                    help="Output folder for annotated images")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # collect images
    if os.path.isdir(args.image):
        img_paths = sorted(glob.glob(os.path.join(args.image, "*.png")) +
                           glob.glob(os.path.join(args.image, "*.jpg")))
    else:
        img_paths = [args.image]
    assert img_paths, f"No images found at {args.image}"
    print(f"Found {len(img_paths)} images")

    # load xmodel
    graph      = xir.Graph.deserialize(args.model)
    dpu_graphs = get_child_subgraph_dpu(graph)
    assert dpu_graphs, "No DPU subgraph in xmodel"

    # split images across threads
    chunk_size = math.ceil(len(img_paths) / args.threads)
    chunks     = [img_paths[i:i+chunk_size]
                  for i in range(0, len(img_paths), chunk_size)]

    results  = []
    runners  = []
    threads  = []
    t_start  = time.time()

    for tid, chunk in enumerate(chunks):
        runner = vart.Runner.create_runner(dpu_graphs[0], "run")
        runners.append(runner)
        t = threading.Thread(target=run_dpu_worker,
                             args=(runner, chunk, args.output,
                                   results, tid))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - t_start
    total   = len(img_paths)
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Images processed : {total}")
    print(f"  Total time       : {elapsed:.2f}s")
    print(f"  Throughput       : {total/elapsed:.1f} FPS")
    print(f"  Results saved to : {args.output}/")

if __name__ == "__main__":
    main()