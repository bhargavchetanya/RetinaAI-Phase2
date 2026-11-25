"""
Step 4: Export the trained models for the website (run after training, before deploying).

Run:  pip install onnx onnxscript onnxconverter-common      (once)
      python3 export_web.py
Needs : the trained .pt files (cnn_*.pt and <model>_<task>.pt) and outputs/*/
Output: web_models/<file>.onnx   every trained model as ONNX with float16 weights (runs with
                                 onnxruntime - no PyTorch on the server - and all 10 models need
                                 only ~280 MB of RAM, so it fits in Render's free plan)
        web_models/manifest.json  test metrics, training history (per epoch), layer table,
                                 normalisation values and data split of every model
"""
import json
import os
import re
import shutil
import warnings

import onnx
import torch
from onnxconverter_common import float16

from cnn import load_model
from preprocess import IMG_SIZE
from pretrained import ARCHS, load_pretrained

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "web_models")

EPOCH_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)%\s+(\d+)s(\s+\*)?\s*$")
LAYER_RE = re.compile(r"^(.+?)\s{2,}(\(.*?\))\s+([\d,]+)\s*$")
SPLIT_RE = re.compile(r"^\s+(train|val|test)\s+(.*)$")


def parse_log(path):
    """Read the epoch table, the layer table and the data split from a training_log.txt."""
    out = {"history": [], "layers": [], "split": {}, "optimizer": None, "duplicates": None, "seconds": 0}
    if not os.path.exists(path):
        return out
    in_layers = False
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = EPOCH_RE.match(line)
        if m:
            e, tl, ta, vl, va, sec, star = m.groups()
            out["history"].append({"epoch": int(e), "train_loss": float(tl), "train_acc": round(float(ta) / 100, 4),
                                   "val_loss": float(vl), "val_acc": round(float(va) / 100, 4), "best": bool(star)})
            out["seconds"] += int(sec)
            continue
        if line.startswith("Layer") and "Output shape" in line:
            in_layers = True
            continue
        if in_layers:
            if line.startswith("Total trainable"):
                in_layers = False
                continue
            m = LAYER_RE.match(line)
            if m:
                out["layers"].append({"name": m.group(1).strip(), "shape": m.group(2),
                                      "params": int(m.group(3).replace(",", ""))})
            continue
        m = SPLIT_RE.match(line)
        if m:
            counts = re.findall(r"([A-Za-z][A-Za-z ]*?):\s*(\d+)", m.group(2))
            out["split"][m.group(1)] = {k.strip(): int(v) for k, v in counts}
        elif line.startswith("Optimizer:"):
            out["optimizer"] = line[len("Optimizer:"):].strip()
        elif line.startswith("Duplicate photos:"):
            out["duplicates"] = line[len("Duplicate photos:"):].strip()
    return out


def export(model, path):
    """Export to ONNX with a variable batch size (the server sends the photo + 3 mirrored copies),
    then store the weights as float16: half the size and memory, predictions unchanged
    (input and output stay float32)."""
    model.eval()
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    kwargs = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kwargs)
    except TypeError:                                    # older PyTorch without the dynamo argument
        torch.onnx.export(model, x, path, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # "number truncated" notes for tiny weights
        onnx.save(float16.convert_float_to_float16(onnx.load(path), keep_io_types=True), path)


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    # (checkpoint, outputs folder, loader, key, display name, pre-trained?)
    jobs = []
    for task in ("dr", "grade"):
        jobs.append((f"cnn_{task}.pt", task, load_model, "cnn", "5-layer CNN", False))
        for key, cfg in ARCHS.items():
            jobs.append((f"{key}_{task}.pt", f"{key}_{task}", load_pretrained, key, cfg["name"], True))

    models = []
    for ckpt_file, out, loader, key, name, pretrained in jobs:
        ckpt_path = os.path.join(BASE_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"  skip   {ckpt_file:<28} (not trained)")
            continue
        model, info = loader(ckpt_path)
        onnx_name = ckpt_file.replace(".pt", ".onnx")
        export(model, os.path.join(OUT_DIR, onnx_name))
        metrics_path = os.path.join(BASE_DIR, "outputs", out, "metrics.json")
        metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        metrics = {k: None if isinstance(v, float) and v != v else v for k, v in metrics.items()}   # NaN -> null
        log = parse_log(os.path.join(BASE_DIR, "outputs", out, "training_log.txt"))
        size_mb = os.path.getsize(os.path.join(OUT_DIR, onnx_name)) / 1e6
        models.append({
            "id": f"{key}_{info['task']}", "key": key, "name": name, "pretrained": pretrained,
            "task": info["task"], "classes": info["classes"], "file": onnx_name,
            "mean": info["mean"], "std": info["std"], "img_size": IMG_SIZE,
            "parameters": sum(p.numel() for p in model.parameters()),
            "onnx_mb": round(size_mb, 1), "metrics": metrics, **log,
        })
        print(f"  export {ckpt_file:<28} -> web_models/{onnx_name}  ({size_mb:.1f} MB, "
              f"{100 * info['test_accuracy']:.1f}% test accuracy)")

    if not models:
        print("No trained model found - train first (python3 train_cnn.py / train_pretrained.py).")
        return
    manifest = {"img_size": IMG_SIZE, "models": models}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(m["onnx_mb"] for m in models)
    print(f"\nExported {len(models)} models ({total:.0f} MB) + web_models/manifest.json")


if __name__ == "__main__":
    main()
