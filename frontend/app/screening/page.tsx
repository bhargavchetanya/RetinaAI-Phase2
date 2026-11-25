"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, ScanEye, ShieldAlert, XCircle } from "lucide-react";
import ImageUploader from "@/components/ImageUploader";
import ProbBars from "@/components/ProbBars";
import {
  api,
  MODEL_COLOR,
  pct,
  sampleUrl,
  sortModels,
  TASK_LABEL,
  type Consensus,
  type ModelResult,
  type Prediction,
  type Sample,
  type Task,
} from "@/lib/api";

export default function ScreeningPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  useEffect(() => {
    api.samples().then(setSamples).catch((e) => setError(`Server not reachable: ${(e as Error).message}`));
  }, []);

  async function run(p: Promise<Prediction>) {
    setLoading(true);
    setError(null);
    try {
      setResult(await p);
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function chooseFile(f: File | null) {
    setFile(f);
    setSample(null);
    setResult(null);
  }

  function chooseSample(s: Sample) {
    setFile(null);
    setSample(s);
    run(api.predictSample(s.file));
  }

  const truth = sample?.grade ?? null;

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <h1 className="text-3xl font-bold">Screening</h1>
      <p className="mt-2 max-w-3xl text-slate-400">
        Upload a retina (fundus) photo or pick one of the sample photos. All trained models look at it – the 5-layer CNN
        and the 4 pre-trained networks – for both tasks: <b className="text-slate-200">is there DR?</b> and{" "}
        <b className="text-slate-200">which grade?</b>
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,380px)_1fr]">
        <div className="card self-start lg:sticky lg:top-20">
          <ImageUploader file={file} onFile={chooseFile} previewUrl={sample ? sampleUrl(sample.file) : null} />
          <button
            disabled={!file || loading}
            onClick={() => file && run(api.predict(file))}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanEye className="h-4 w-4" />}
            {loading ? "Analysing…" : "Analyse photo"}
          </button>

          {samples.length > 0 && (
            <>
              <div className="label mb-2 mt-6">Or try a sample (doctor&apos;s grade shown)</div>
              <div className="grid grid-cols-5 gap-2">
                {samples.map((s) => (
                  <button
                    key={s.file}
                    onClick={() => chooseSample(s)}
                    disabled={loading}
                    title={s.grade_name ? `Grade ${s.grade}: ${s.grade_name}` : s.file}
                    className={`overflow-hidden rounded-lg border text-center transition ${
                      sample?.file === s.file ? "border-cyan-400" : "border-slate-800 hover:border-slate-600"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={sampleUrl(s.file)} alt="" className="aspect-square w-full object-cover" />
                    <div className="py-1 text-[11px] text-slate-400">{s.grade !== null ? `grade ${s.grade}` : "sample"}</div>
                  </button>
                ))}
              </div>
            </>
          )}
          {error && (
            <p className="mt-4 flex items-start gap-2 rounded-lg bg-red-500/10 p-3 text-sm text-red-300">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {error}
            </p>
          )}
        </div>

        <div className="min-w-0">
          {loading && !result && (
            <div className="card flex h-full min-h-64 items-center justify-center text-slate-400">
              <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Running all models…
            </div>
          )}
          {!loading && !result && (
            <div className="card flex h-full min-h-64 flex-col items-center justify-center text-center text-slate-400">
              <ScanEye className="h-10 w-10 text-slate-600" />
              <p className="mt-3">Upload a photo or pick a sample to see what the models say.</p>
            </div>
          )}
          {result && <Result r={result} truth={truth} busy={loading} />}
        </div>
      </div>
    </main>
  );
}

function Result({ r, truth, busy }: { r: Prediction; truth: number | null; busy: boolean }) {
  return (
    <div className={`flex flex-col gap-6 transition ${busy ? "opacity-50" : ""}`}>
      <div className="grid gap-4 md:grid-cols-2">
        {(["dr", "grade"] as const).map((t) =>
          r.consensus[t] ? <ConsensusCard key={t} task={t} c={r.consensus[t]!} truth={truth} /> : null,
        )}
      </div>

      <div className="card">
        <div className="label mb-3">What the models see</div>
        <div className="flex flex-wrap items-center gap-5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={r.model_view} alt="pre-processed 48 × 48 photo" className="h-36 w-36 rounded-lg" style={{ imageRendering: "pixelated" }} />
          <p className="max-w-xl text-sm leading-6 text-slate-400">
            The photo after pre-processing: black border cropped, Ben Graham contrast enhancement (lesions and vessels stand
            out, uneven lighting removed) and shrunk to <b className="text-slate-200">48 × 48 pixels</b>. The pre-trained
            networks enlarge it to 128 × 128 inside the model. Each prediction averages the photo and 3 mirrored copies.
            <span className="mt-2 block text-xs text-slate-500">All {r.results.length} models ran in {r.inference_ms} ms on the server.</span>
          </p>
        </div>
      </div>

      {(["dr", "grade"] as const).map((t) => {
        const rs = sortModels(r.results.filter((x) => x.task === t));
        if (!rs.length) return null;
        return (
          <div key={t}>
            <h2 className="mb-3 text-lg font-semibold">{TASK_LABEL[t]} – every model</h2>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {rs.map((m) => (
                <ModelCard key={m.id} m={m} truth={truth} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function isRight(task: Task, pred: number, truth: number) {
  return task === "dr" ? (pred === 1) === truth > 0 : pred === truth;
}

function ConsensusCard({ task, c, truth }: { task: Task; c: Consensus; truth: number | null }) {
  const answer = c.classes[c.pred];
  const alarm = task === "dr" ? c.pred === 1 : c.pred >= 2;
  const right = truth !== null ? isRight(task, c.pred, truth) : null;
  return (
    <div className={`rounded-2xl border p-5 ${alarm ? "border-red-500/40 bg-red-500/10" : "border-green-500/40 bg-green-500/10"}`}>
      <div className="label">{task === "dr" ? "DR detection" : "DR grade"} · all models</div>
      <div className="mt-2 flex items-center gap-3">
        {alarm ? (
          <ShieldAlert className="h-8 w-8 shrink-0" style={{ color: "var(--critical)" }} />
        ) : (
          <CheckCircle2 className="h-8 w-8 shrink-0" style={{ color: "var(--good)" }} />
        )}
        <div>
          <div className="text-2xl font-bold">{answer}</div>
          <div className="text-sm text-slate-300">
            {c.agree} of {c.n} models agree · average {pct(c.mean_probs[c.pred], 0)}
          </div>
        </div>
      </div>
      <div className="mt-4">
        <ProbBars classes={c.classes} probs={c.mean_probs} pred={c.pred} color="var(--text-secondary)" />
      </div>
      {task === "grade" && alarm && (
        <p className="mt-3 text-xs text-slate-300">Grade 2 or worse is usually referred to an eye specialist.</p>
      )}
      {right !== null && (
        <p className="mt-3 flex items-center gap-1.5 text-sm">
          {right ? <CheckCircle2 className="h-4 w-4" style={{ color: "var(--good)" }} /> : <XCircle className="h-4 w-4" style={{ color: "var(--critical)" }} />}
          Doctor&apos;s grade: {truth} – the models are {right ? "right" : "wrong"}
        </p>
      )}
    </div>
  );
}

function ModelCard({ m, truth }: { m: ModelResult; truth: number | null }) {
  const right = truth !== null ? isRight(m.task, m.pred, truth) : null;
  return (
    <div className="card">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 font-semibold">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: MODEL_COLOR[m.key] }} />
            {m.name}
          </div>
          <div className="mt-0.5 text-xs text-slate-500">
            {m.pretrained ? "pre-trained (ImageNet)" : "trained from scratch"} · {pct(m.test_accuracy)} test accuracy
          </div>
        </div>
        {right !== null &&
          (right ? (
            <span className="flex items-center gap-1 text-xs" style={{ color: "var(--good)" }}>
              <CheckCircle2 className="h-3.5 w-3.5" /> right
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs" style={{ color: "var(--critical)" }}>
              <XCircle className="h-3.5 w-3.5" /> wrong
            </span>
          ))}
      </div>
      <div className="mb-3 mt-3 text-xl font-bold">{m.classes[m.pred]}</div>
      <ProbBars classes={m.classes} probs={m.probs} pred={m.pred} color={MODEL_COLOR[m.key]} />
    </div>
  );
}
