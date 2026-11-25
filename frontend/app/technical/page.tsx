"use client";

import { ArrowRight } from "lucide-react";
import { useManifest } from "@/components/useManifest";
import { millions, MODEL_COLOR, sortModels, type ModelInfo } from "@/lib/api";

const ARCH: Record<string, { year: string; paper: string; idea: string; blocks: string }> = {
  resnet18: {
    year: "2015",
    paper: "He et al., Deep Residual Learning for Image Recognition",
    idea: "Residual blocks: two 3×3 convolutions plus a skip connection that adds the block's input to its output. The gradient can flow through the shortcut, so deep networks train without vanishing gradients.",
    blocks: "stem (7×7 conv, max-pool) → 4 stages × 2 residual blocks (64, 128, 256, 512 filters) → global average pooling",
  },
  efficientnet_b0: {
    year: "2019",
    paper: "Tan & Le, EfficientNet: Rethinking Model Scaling for CNNs",
    idea: "MBConv blocks: depthwise-separable convolutions with squeeze-and-excitation (channel attention) and SiLU activation. Depth, width and input resolution are scaled together (compound scaling) for the best accuracy per computation.",
    blocks: "stem conv → 7 stages of MBConv blocks (16 → 320 channels) → 1×1 conv to 1280 → global average pooling",
  },
  densenet121: {
    year: "2017",
    paper: "Huang et al., Densely Connected Convolutional Networks",
    idea: "Inside a dense block every layer receives the feature maps of all earlier layers (concatenation, not addition). Features are re-used, so the network needs fewer weights and gradients reach early layers easily.",
    blocks: "stem → 4 dense blocks (6, 12, 24, 16 layers) joined by transition layers (1×1 conv + average pooling) → 1024 channels",
  },
  mobilenet_v3_large: {
    year: "2019",
    paper: "Howard et al., Searching for MobileNetV3",
    idea: "Inverted-residual blocks (expand → depthwise conv → project) with squeeze-and-excitation and the cheap hard-swish activation; the block layout was found by neural architecture search for fast inference on phones.",
    blocks: "stem → 15 inverted-residual blocks (16 → 160 channels) → 1×1 conv to 960 → global average pooling",
  },
  cnn: {
    year: "ours",
    paper: "Baseline designed for this project",
    idea: "Five 3×3 convolution layers (32, 64, 128, 256, 512 filters), each followed by batch normalisation and ReLU, with 2×2 max-pooling after layers 2–5. Trained from random weights.",
    blocks: "48×48 input → 5 conv layers → 3×3×512 → global average pooling → dropout → dense",
  },
};

const PREPROCESS = [
  ["Read", "OpenCV reads the photo (any size; APTOS photos are ~2048 × 1536)."],
  ["Crop", "Remove the black area around the round retina (pixels darker than 7)."],
  ["Resize", "Scale to 256 × 256 so every photo has the same size."],
  ["Enhance", "Ben Graham filter: 4 · image − 4 · Gaussian blur (σ = 10) + 128. Removes uneven lighting and makes lesions and vessels stand out."],
  ["Shrink", "Area-interpolate to 48 × 48 × 3 and scale pixel values to 0–1."],
  ["Standardise", "Subtract the training-set mean and divide by its standard deviation (per colour channel)."],
];

const TRAINING = [
  ["Split", "70 % train / 15 % validation / 15 % test, stratified by grade (rare grades appear in every split)."],
  ["No leakage", "Duplicate photos of the same eye (vessel-pattern correlation ≥ 0.90) are grouped and kept in one split, so no model is tested on a photo it trained on."],
  ["Augmentation", "Every epoch each training photo is randomly rotated by 90°, mirrored, shifted up to 3 px and changed in brightness / contrast – on the GPU."],
  ["Optimiser", "AdamW with a one-cycle learning-rate schedule (warm-up, then cosine decay). Weight decay 0.05."],
  ["Regularisation", "Dropout 0.4 before the output layer, label smoothing 0.05, weight decay, data augmentation."],
  ["Model selection", "After every epoch the model is scored on the validation set; the epoch with the best validation accuracy is kept."],
  ["Test-time augmentation", "A prediction averages the photo and its 3 mirrored copies (horizontal, vertical, both)."],
  ["Hardware", "Apple Silicon GPU through PyTorch MPS."],
];

const TRANSFER = [
  "Load the network with its ImageNet weights (torchvision).",
  "Remove the old 1000-class output layer; add global average pooling → dropout → a new dense layer with 2 or 5 outputs.",
  "Enlarge the 48 × 48 photo to 128 × 128 (bilinear) and turn OpenCV's BGR into RGB inside the model, because the pre-trained filters were learned on larger RGB photos.",
  "Fine-tune ALL layers for 10 epochs with a small learning rate (0.0005 – 0.001): the pre-trained features only need adjusting, not learning from zero.",
];

const STACK = [
  ["Training", "Python 3, PyTorch, torchvision (pre-trained weights), scikit-learn (split, metrics), NumPy, OpenCV, Matplotlib"],
  ["Model export", "ONNX (opset 17) with float16 weights – half the size, same predictions"],
  ["Inference server", "FastAPI + Uvicorn, onnxruntime on CPU – no PyTorch needed; all 10 models use ≈ 280 MB of RAM"],
  ["Website", "Next.js 15 (static export), React 19, TypeScript, Tailwind CSS 4, Recharts, lucide icons"],
  ["Deployment", "One Docker image (Node builds the site, Python serves site + API) on Render"],
];

const METRICS = [
  ["Accuracy", "Share of test photos classified correctly."],
  ["Sensitivity (recall)", "Of the eyes that really have DR, the share the model found. The most important number for screening – a missed patient can go blind."],
  ["Specificity", "Of the healthy eyes, the share the model called healthy (fewer needless referrals)."],
  ["Precision", "Of the photos the model called DR, the share that really had DR."],
  ["ROC-AUC", "Probability that a random DR photo gets a higher DR score than a random healthy photo (1 = perfect, 0.5 = guessing)."],
  ["QWK", "Quadratic weighted kappa: agreement with the doctor's grade, where mistakes cost the square of the grade distance (grade 4 called 0 is 16× worse than 4 called 3). Official metric of the APTOS competition."],
  ["F1", "Harmonic mean of precision and recall for one class."],
];

export default function TechnicalPage() {
  const { data, error } = useManifest();
  const models = sortModels((data?.models ?? []).filter((m) => m.task === "grade"));
  const drModels = sortModels((data?.models ?? []).filter((m) => m.task === "dr"));
  const ref = models[0] ?? drModels[0];
  const split = ref?.split ?? {};
  const classes = ref && ref.task === "grade" ? ref.classes : ["No DR", "Mild", "Moderate", "Severe", "Proliferative"];

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <h1 className="text-3xl font-bold">Technical details</h1>
      <p className="mt-2 max-w-3xl text-slate-400">How the data is prepared, how the five networks are built and trained, and how the website runs them.</p>
      {error && <p className="mt-5 rounded-lg bg-red-500/10 p-3 text-sm text-red-300">Server unreachable – model details are missing: {error}</p>}

      {/* system flow */}
      <Section title="How a prediction is made">
        <div className="flex flex-wrap items-stretch gap-2">
          {["Photo upload", "Pre-processing (OpenCV)", "48 × 48 × 3 image", "10 ONNX models (5 per task)", "Average of 4 mirrored copies", "Votes + average probability"].map((s, i, a) => (
            <div key={s} className="flex items-center gap-2">
              <div className="rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm">{s}</div>
              {i < a.length - 1 && <ArrowRight className="h-4 w-4 shrink-0 text-slate-600" />}
            </div>
          ))}
        </div>
      </Section>

      {/* data */}
      <Section title="Dataset: APTOS 2019 Blindness Detection">
        <div className="grid gap-6 lg:grid-cols-[1fr_1.3fr]">
          <div className="space-y-3 text-sm leading-6 text-slate-300">
            <p>
              3,662 colour retina (fundus) photos from Aravind Eye Hospital, India, published on Kaggle. Each photo was graded
              by doctors on the 5-step ICDR scale (0 No DR → 4 Proliferative DR).
            </p>
            <p>
              The classes are imbalanced: about half of the photos are healthy, and Severe has only ~190 photos. That is why
              sensitivity, specificity and QWK are reported next to accuracy.
            </p>
            {ref?.duplicates && (
              <p className="text-slate-400">Duplicates found: {ref.duplicates}.</p>
            )}
          </div>
          {Object.keys(split).length > 0 && (
            <div className="overflow-x-auto">
              <table className="tabular w-full text-sm">
                <thead className="text-left text-xs text-slate-400">
                  <tr>
                    <th className="py-2">Split</th>
                    {classes.map((c) => <th key={c} className="py-2 text-right">{c}</th>)}
                    <th className="py-2 text-right">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(split).map(([s, counts]) => (
                    <tr key={s} className="border-t border-slate-800">
                      <td className="py-2 capitalize">{s === "val" ? "validation" : s}</td>
                      {classes.map((c) => <td key={c} className="py-2 text-right">{counts[c] ?? "–"}</td>)}
                      <td className="py-2 text-right font-semibold">{Object.values(counts).reduce((a, b) => a + b, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Section>

      <Section title="Pre-processing (same for training and the website)">
        <Steps rows={PREPROCESS} />
      </Section>

      <Section title="Training setup (identical for all models)">
        <Steps rows={TRAINING} />
        {drModels.length > 0 && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-400">
                <tr><th className="py-2 pr-4">Model</th><th className="py-2">Settings (from the training log)</th></tr>
              </thead>
              <tbody>
                {drModels.map((m) => (
                  <tr key={m.key} className="border-t border-slate-800 align-top">
                    <td className="whitespace-nowrap py-2 pr-4"><Dot k={m.key} /> {m.name}</td>
                    <td className="py-2 text-slate-400">{m.optimizer ?? "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="Transfer learning – how the 4 pre-trained models are adapted">
        <ol className="list-decimal space-y-2 pl-5 text-sm leading-6 text-slate-300">
          {TRANSFER.map((t) => <li key={t}>{t}</li>)}
        </ol>
      </Section>

      {/* architectures */}
      <h2 className="mt-10 text-xl font-semibold">The 5 architectures</h2>
      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        {(models.length ? models : drModels).map((m) => <ArchCard key={m.key} m={m} />)}
      </div>

      <Section title="Metrics used">
        <dl className="grid gap-4 text-sm sm:grid-cols-2">
          {METRICS.map(([k, v]) => (
            <div key={k}>
              <dt className="font-semibold text-slate-200">{k}</dt>
              <dd className="mt-1 leading-6 text-slate-400">{v}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section title="Tech stack & deployment">
        <Steps rows={STACK} />
      </Section>

      <Section title="Limitations">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-6 text-slate-300">
          <li>Photos are shrunk to 48 × 48 pixels, so tiny lesions (micro-aneurysms) are mostly lost – this limits grading accuracy more than the choice of network. Larger inputs (e.g. 320 px) reach higher accuracy but train many times slower.</li>
          <li>All models confuse neighbouring grades most, especially Severe (fewest photos) with Moderate.</li>
          <li>One dataset from one hospital: results on photos from other cameras or populations may be lower.</li>
          <li>A student project for learning – not a medical device and not a substitute for an eye doctor.</li>
        </ul>
      </Section>
    </main>
  );
}

function Dot({ k }: { k: string }) {
  return <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full align-middle" style={{ background: MODEL_COLOR[k] }} />;
}

function ArchCard({ m }: { m: ModelInfo }) {
  const a = ARCH[m.key];
  return (
    <div className="card">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-lg font-semibold"><Dot k={m.key} />{m.name}</div>
          <div className="text-xs text-slate-500">{a?.year} · {a?.paper}</div>
        </div>
        <span className={`shrink-0 rounded-full border px-2 py-0.5 text-xs ${m.pretrained ? "border-cyan-500/40 text-cyan-400" : "border-slate-600 text-slate-400"}`}>
          {m.pretrained ? "pre-trained" : "from scratch"}
        </span>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-300">{a?.idea}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{a?.blocks}</p>
      <div className="tabular mt-4 grid grid-cols-3 gap-3 text-sm">
        <Fact k="Weights" v={millions(m.parameters)} />
        <Fact k="Stages" v={String(Math.max(0, m.layers.length - 1))} />
        <Fact k="ONNX file" v={`${m.onnx_mb} MB`} />
      </div>
      {m.layers.length > 0 && (
        <details className="mt-4 rounded-lg border border-slate-800 p-3 text-sm">
          <summary className="cursor-pointer text-slate-300">Layer table (output shape, weights)</summary>
          <table className="tabular mt-3 w-full text-xs">
            <tbody>
              {m.layers.map((l, i) => (
                <tr key={i} className="border-t border-slate-800">
                  <td className="py-1.5 pr-2 text-slate-300">{l.name}</td>
                  <td className="py-1.5 pr-2 text-slate-400">{l.shape}</td>
                  <td className="py-1.5 text-right text-slate-400">{l.params.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{k}</div>
      <div className="font-semibold">{v}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card mt-6">
      <h2 className="mb-4 text-lg font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Steps({ rows }: { rows: string[][] }) {
  return (
    <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-[180px_1fr]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="font-semibold text-slate-200">{k}</dt>
          <dd className="leading-6 text-slate-400">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
