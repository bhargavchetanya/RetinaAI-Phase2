# RetinaAI website - one container: Node builds the Next.js site, Python serves site + API.
# Render builds this automatically (see render.yaml). Local test:
#   docker build -t retinaai . && docker run -p 8000:8000 -e PORT=8000 retinaai

# ---- 1. build the website (static HTML/JS in /site/out) ----
FROM node:22-alpine AS site
WORKDIR /site
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. the server: FastAPI + onnxruntime (no PyTorch, ~280 MB RAM) ----
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt
COPY server.py preprocess.py ./
COPY web_models/ web_models/
COPY samples/ samples/
COPY --from=site /site/out frontend/out
ENV PORT=8000
EXPOSE 8000
CMD ["python", "server.py"]
