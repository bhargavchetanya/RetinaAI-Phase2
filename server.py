"""
RetinaAI website server (the one deployed on Render).

Runs the exported ONNX models with onnxruntime - no PyTorch needed, so it fits in
512 MB of RAM (all 10 models together use ~280 MB) - and serves the Next.js website (frontend/out) on the same address.

Local run:
    python3 export_web.py                 (once, after training: .pt -> web_models/*.onnx)
    cd frontend && npm install && npm run build && cd ..
    python3 server.py                     then open http://127.0.0.1:8000
On Render the Dockerfile does the same; the port comes from $PORT.
"""
import base64
import json
import os
import tempfile
import time

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from preprocess import GRADES, preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "web_models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
SITE_DIR = os.path.join(BASE_DIR, "frontend", "out")
MAX_UPLOAD = 15 * 1024 * 1024              # 15 MB


def load_models():
    path = os.path.join(MODELS_DIR, "manifest.json")
    if not os.path.exists(path):
        print("web_models/manifest.json not found - run  python3 export_web.py  first")
        return {"models": []}, []
    manifest = json.load(open(path))
    opts = ort.SessionOptions()
    # settings for a small server: all 10 models together need ~280 MB of RAM
    opts.intra_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC   # no extra weight copies
    loaded = []
    for m in manifest["models"]:
        session = ort.InferenceSession(os.path.join(MODELS_DIR, m["file"]), opts,
                                       providers=["CPUExecutionProvider"])
        loaded.append((m, session))
        print(f"  loaded {m['file']:<30} {m['name']:<18} {m['task']:<6} "
              f"{100 * (m['metrics'].get('test_accuracy') or 0):.1f}% test accuracy")
    return manifest, loaded


print("Loading models ...")
MANIFEST, LOADED = load_models()

app = FastAPI(title="RetinaAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def run(model, session, img):
    """Class probabilities, averaged over the photo and 3 mirrored copies (same as training)."""
    x = (img - np.array(model["mean"], np.float32)) / np.array(model["std"], np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)                       # 1 x 3 x 48 x 48
    batch = np.concatenate([x, x[..., ::-1], x[..., ::-1, :], x[..., ::-1, ::-1]]).copy()
    logits = session.run(None, {"image": batch})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).mean(axis=0)


def consensus(results, task):
    rs = [r for r in results if r["task"] == task]
    if not rs:
        return None
    classes = rs[0]["classes"]
    votes = np.bincount([r["pred"] for r in rs], minlength=len(classes))
    mean = np.mean([r["probs"] for r in rs], axis=0)
    top = int(np.argmax(mean))
    return {"classes": classes, "votes": votes.tolist(), "mean_probs": mean.round(4).tolist(),
            "pred": top, "agree": int(votes[top]), "n": len(rs)}


def predict(path):
    if not LOADED:
        raise HTTPException(503, "No models on the server - run python3 export_web.py and redeploy.")
    img = preprocess_image(path)                       # same pre-processing as training
    if img is None:
        raise HTTPException(400, "Could not read this file as an image.")
    start = time.perf_counter()
    results = []
    for m, session in LOADED:
        probs = run(m, session, img)
        results.append({"id": m["id"], "key": m["key"], "name": m["name"], "pretrained": m["pretrained"],
                        "task": m["task"], "classes": m["classes"], "probs": probs.round(4).tolist(),
                        "pred": int(np.argmax(probs)), "test_accuracy": m["metrics"].get("test_accuracy")})
    ms = round(1000 * (time.perf_counter() - start))
    # what the models see: the 48 x 48 pre-processed photo, enlarged for the screen (BGR -> PNG)
    seen = cv2.resize((img * 255).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST)
    _, png = cv2.imencode(".png", seen)
    return {"results": results, "consensus": {t: consensus(results, t) for t in ("dr", "grade")},
            "model_view": "data:image/png;base64," + base64.b64encode(png).decode(), "inference_ms": ms}


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": len(LOADED)}


@app.get("/api/models")
def models():
    return MANIFEST


def sample_names():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))


@app.get("/api/samples")
def samples():
    out = []
    for name in sample_names():
        grade = next((int(c) for c in name.split("_")[0] if c.isdigit()), None) if name.startswith("grade") else None
        out.append({"file": name, "grade": grade, "grade_name": GRADES[grade] if grade is not None else None})
    return out


@app.get("/api/predict-sample/{name}")
def predict_sample(name: str):
    if name not in sample_names():
        raise HTTPException(404, "No such sample")
    return predict(os.path.join(SAMPLES_DIR, name))


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 15 MB).")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return predict(tmp.name)
    finally:
        os.remove(tmp.name)


if os.path.isdir(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
if os.path.isdir(SITE_DIR):                            # the built website; must be mounted last
    app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
else:
    print("frontend/out not found - build the website:  cd frontend && npm install && npm run build")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    print(f"\nRetinaAI website: open  http://127.0.0.1:{port}   (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
