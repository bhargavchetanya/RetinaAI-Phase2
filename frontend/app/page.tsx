import Link from "next/link";
import HomeStats from "@/components/HomeStats";

const PIPELINE = [
  { step: "01", title: "Data", text: "APTOS 2019 (Aravind Eye Hospital, India, via Kaggle) – 3,662 retina photos, each graded 0–4 by doctors." },
  { step: "02", title: "Pre-processing", text: "Crop the black border, resize to 256 px, Ben Graham contrast enhancement, shrink to 48 × 48 × 3." },
  { step: "03", title: "Leak-free split", text: "Stratified 70 / 15 / 15 split. Duplicate photos of the same eye are found by correlation and kept in one split." },
  { step: "04", title: "Baseline CNN", text: "A 5-layer convolutional network trained from scratch (1.6 M weights) – the reference point." },
  { step: "05", title: "4 pre-trained models", text: "ResNet-18, EfficientNet-B0, DenseNet-121 and MobileNetV3-Large, pre-trained on ImageNet and fine-tuned (transfer learning)." },
  { step: "06", title: "Two tasks", text: "Every model is trained twice: DR detection (No DR / DR) and DR grading (5 classes)." },
  { step: "07", title: "Same test photos", text: "All models are scored on the same 548 unseen test photos: accuracy, sensitivity, specificity, ROC-AUC, QWK." },
  { step: "08", title: "Ensemble view", text: "The website runs all models on your photo and shows how many of them agree." },
  { step: "09", title: "Deployment", text: "Models exported to ONNX (float16), served by FastAPI + onnxruntime with a Next.js front end on Render." },
];

const MODELS = [
  { name: "ResNet-18", year: "2015", idea: "Residual blocks: a skip connection adds each block's input to its output, so deep networks still train.", color: "var(--series-1)" },
  { name: "EfficientNet-B0", year: "2019", idea: "MBConv blocks with squeeze-and-excitation; depth, width and resolution scaled together.", color: "var(--series-2)" },
  { name: "DenseNet-121", year: "2017", idea: "Dense blocks: every layer receives the feature maps of all earlier layers, so features are re-used.", color: "var(--series-3)" },
  { name: "MobileNetV3-Large", year: "2019", idea: "Light inverted-residual blocks with hard-swish, designed to run fast on phones.", color: "var(--series-4)" },
  { name: "5-layer CNN", year: "baseline", idea: "Our own network trained from scratch: 5 conv layers (32 → 512 filters) with BatchNorm, ReLU, max-pooling.", color: "var(--series-5)" },
];

const GRADES = [
  { g: 0, name: "No DR", text: "Healthy retina" },
  { g: 1, name: "Mild", text: "Micro-aneurysms only" },
  { g: 2, name: "Moderate", text: "More than micro-aneurysms" },
  { g: 3, name: "Severe", text: "Many haemorrhages, venous beading" },
  { g: 4, name: "Proliferative", text: "New fragile vessels – risk of blindness" },
];

export default function Home() {
  return (
    <main className="text-white">
      <section className="mx-auto flex max-w-7xl flex-col px-4 py-16 sm:px-6 sm:py-20">
        <div className="max-w-4xl">
          <div className="mb-6 inline-block rounded-full border border-cyan-500/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-400">
            Computer Vision • Transfer Learning • Model Comparison
          </div>

          <h1 className="text-4xl font-bold leading-tight sm:text-5xl md:text-7xl">
            <span className="text-cyan-400">RetinaAI</span>
            <br />
            Diabetic Retinopathy Detection
          </h1>

          <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-400">
            Diabetic retinopathy damages the blood vessels of the retina and is a leading cause of preventable blindness.
            Upload a retina photo and five neural networks – a CNN trained from scratch and four networks pre-trained on
            ImageNet – tell you whether it shows DR and how severe it is.
          </p>

          <div className="mt-10 flex flex-col gap-4 sm:flex-row">
            <Link href="/screening/" className="rounded-xl bg-cyan-500 px-6 py-3 text-center font-semibold text-slate-950 transition hover:bg-cyan-400">
              Try a screening
            </Link>
            <Link href="/results/" className="rounded-xl border border-slate-700 px-6 py-3 text-center font-semibold text-slate-100 transition hover:border-slate-500 hover:bg-slate-900">
              Training results
            </Link>
            <Link href="/technical/" className="rounded-xl border border-slate-700 px-6 py-3 text-center font-semibold text-slate-100 transition hover:border-slate-500 hover:bg-slate-900">
              How it works
            </Link>
          </div>
        </div>

        <HomeStats />

        <h2 className="mt-16 text-2xl font-semibold">The 5 models</h2>
        <p className="mt-2 max-w-3xl text-slate-400">
          Four well-known architectures pre-trained on 1.2 million ImageNet photos (transfer learning), compared with a CNN
          trained from scratch. All five see exactly the same photos.
        </p>
        <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {MODELS.map((m) => (
            <div key={m.name} className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: m.color }} />
                <span className="text-xs text-slate-500">{m.year}</span>
              </div>
              <h3 className="mt-2 font-semibold">{m.name}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">{m.idea}</p>
            </div>
          ))}
        </div>

        <h2 className="mt-16 text-2xl font-semibold">Pipeline</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {PIPELINE.map((p) => (
            <div key={p.step} className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <div className="text-sm font-semibold text-cyan-400">{p.step}</div>
              <h3 className="mt-1 font-semibold">{p.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">{p.text}</p>
            </div>
          ))}
        </div>

        <h2 className="mt-16 text-2xl font-semibold">The 5 DR grades (ICDR scale)</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {GRADES.map((g) => (
            <div key={g.g} className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
              <div className="text-3xl font-bold text-cyan-400">{g.g}</div>
              <h3 className="mt-2 font-semibold">{g.name}</h3>
              <p className="mt-1 text-sm text-slate-500">{g.text}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
