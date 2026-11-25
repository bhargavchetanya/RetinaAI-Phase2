// Built site: same address as the website (server.py serves both).
// `npm run dev` (port 3000): talk to python3 server.py on port 8000.
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "");

export type Task = "dr" | "grade";

export interface EpochRow {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  best: boolean;
}

export interface Metrics {
  test_accuracy?: number | null;
  val_accuracy?: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
  precision?: number | null;
  roc_auc?: number | null;
  qwk?: number | null;
  dr_yes_no_accuracy?: number | null;
  epochs?: number;
  best_epoch?: number;
  test_images?: number;
  train_images?: number;
  val_images?: number;
  confusion_matrix?: number[][];
}

export interface ModelInfo {
  id: string;
  key: string;
  name: string;
  pretrained: boolean;
  task: Task;
  classes: string[];
  parameters: number;
  onnx_mb: number;
  metrics: Metrics;
  history: EpochRow[];
  layers: { name: string; shape: string; params: number }[];
  split: Record<string, Record<string, number>>;
  optimizer: string | null;
  duplicates: string | null;
  seconds: number;
}

export interface Manifest {
  img_size: number;
  models: ModelInfo[];
}

export interface ModelResult {
  id: string;
  key: string;
  name: string;
  pretrained: boolean;
  task: Task;
  classes: string[];
  probs: number[];
  pred: number;
  test_accuracy: number | null;
}

export interface Consensus {
  classes: string[];
  votes: number[];
  mean_probs: number[];
  pred: number;
  agree: number;
  n: number;
}

export interface Prediction {
  results: ModelResult[];
  consensus: Record<Task, Consensus | null>;
  model_view: string;
  inference_ms: number;
}

export interface Sample {
  file: string;
  grade: number | null;
  grade_name: string | null;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { ...init, cache: "no-store" });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") msg = body.detail;
    } catch {}
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  models: () => call<Manifest>("/api/models"),
  samples: () => call<Sample[]>("/api/samples"),
  predictSample: (name: string) => call<Prediction>(`/api/predict-sample/${encodeURIComponent(name)}`),
  predict: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<Prediction>("/api/predict", { method: "POST", body: fd });
  },
};

export const sampleUrl = (file: string) => `${API_URL}/samples/${encodeURIComponent(file)}`;

/** Fixed colour per model (identity follows the model, never its rank). Validated palette. */
export const MODEL_COLOR: Record<string, string> = {
  resnet18: "var(--series-1)",
  efficientnet_b0: "var(--series-2)",
  densenet121: "var(--series-3)",
  mobilenet_v3_large: "var(--series-4)",
  cnn: "var(--series-5)",
};
export const MODEL_ORDER = ["resnet18", "efficientnet_b0", "densenet121", "mobilenet_v3_large", "cnn"];
export const TASK_LABEL: Record<Task, string> = { dr: "DR detection (No DR / DR)", grade: "DR grade (5 classes)" };

export const pct = (v: number | null | undefined, d = 1) =>
  v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d)}%`;
export const num = (v: number | null | undefined, d = 3) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));
export const millions = (n: number) => `${(n / 1e6).toFixed(1)} M`;

export function sortModels<T extends { key: string }>(list: T[]): T[] {
  return [...list].sort((a, b) => MODEL_ORDER.indexOf(a.key) - MODEL_ORDER.indexOf(b.key));
}
