"""
RetinaAI website: upload a retina photo and see what every trained model says,
next to the test results and training graphs of each model.

Run:  python3 app.py        then open http://127.0.0.1:8000   (Ctrl+C to stop)
Uses whichever models are trained:
  5-layer CNN (from scratch)  : cnn_dr.pt, cnn_grade.pt
  4 pre-trained models        : <model>_dr.pt, <model>_grade.pt for
                                resnet18, efficientnet_b0, densenet121, mobilenet_v3_large
"""
import base64
import json
import os
import tempfile

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cnn import load_model, predict_proba, to_tensor
from preprocess import preprocess_image
from pretrained import ARCHS, load_pretrained

# files are found next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# (model file, outputs folder, loader, name, pre-trained?) - the CNN first, then the 4 pre-trained models
MODELS = []
for task in ("dr", "grade"):
    MODELS.append((f"cnn_{task}.pt", task, load_model, "5-layer CNN", False))
    for key, cfg in ARCHS.items():
        MODELS.append((f"{key}_{task}.pt", f"{key}_{task}", load_pretrained, cfg["name"], True))


def load_all():
    """Load every trained model once, with the test metrics saved by its training script."""
    loaded = []
    for model_file, out, loader, name, pretrained in MODELS:
        path = os.path.join(BASE_DIR, model_file)
        if not os.path.exists(path):
            continue
        model, info = loader(path)
        metrics_path = os.path.join(OUTPUTS_DIR, out, "metrics.json")
        metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        # NaN (e.g. precision of a model that never says "DR") is not valid JSON for the page
        metrics = {k: None if isinstance(v, float) and v != v else v for k, v in metrics.items()}
        loaded.append({"file": model_file, "outputs": out, "name": name, "model": model, "info": info,
                       "metrics": metrics, "pretrained": pretrained,
                       "parameters": sum(p.numel() for p in model.parameters())})
        print(f"  loaded {model_file:<26} {name:<18} {info['task']:<6} "
              f"{100 * info['test_accuracy']:.1f}% test accuracy")
    return loaded


print("Loading models ...")
LOADED = load_all()
app = FastAPI(title="RetinaAI")


def predict(path):
    img = preprocess_image(path)                       # same pre-processing as training
    if img is None:
        raise HTTPException(400, "Could not read this file as an image.")
    results = []
    for m in LOADED:
        x = to_tensor(img[None], m["info"]["mean"], m["info"]["std"], "cpu")
        probs = predict_proba(m["model"], x)[0].tolist()
        results.append({"name": m["name"], "pretrained": m["pretrained"],
                        "task": m["info"]["task"], "classes": m["info"]["classes"],
                        "probs": probs, "pred": int(np.argmax(probs)),
                        "test_accuracy": m["info"]["test_accuracy"]})
    # what the models see: the 48 x 48 pre-processed photo, enlarged for the screen
    seen = cv2.resize((img * 255).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST)
    _, png = cv2.imencode(".png", seen)
    return {"results": results, "model_view": "data:image/png;base64," + base64.b64encode(png).decode()}


@app.get("/api/models")
def models():
    return [{"file": m["file"], "name": m["name"], "task": m["info"]["task"], "outputs": m["outputs"],
             "metrics": m["metrics"], "pretrained": m["pretrained"], "parameters": m["parameters"],
             "has_plots": os.path.exists(os.path.join(OUTPUTS_DIR, m["outputs"], "training_curves.png"))}
            for m in LOADED]


@app.get("/api/missing")
def missing():
    """Pre-trained models that are not trained yet, with the command that trains them."""
    have = {m["file"] for m in LOADED}
    return [{"name": name, "task": out.rsplit("_", 1)[1],
             "command": f"python3 train_pretrained.py --model {out.rsplit('_', 1)[0]} --task {out.rsplit('_', 1)[1]}"}
            for model_file, out, _, name, pretrained in MODELS if pretrained and model_file not in have]


@app.get("/api/samples")
def samples():
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))


@app.get("/api/predict-sample/{name}")
def predict_sample(name: str):
    if name not in samples():
        raise HTTPException(404, "No such sample")
    return predict(os.path.join(SAMPLES_DIR, name))


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
    try:
        return predict(tmp.name)
    finally:
        os.remove(tmp.name)


@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "web", "index.html"))


app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
if os.path.isdir(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


if __name__ == "__main__":
    if not LOADED:
        print("No trained model found - train one first:  python3 train_cnn.py")
    print("\nRetinaAI website: open  http://127.0.0.1:8000   (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
