"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { axisProps, gridProps, Legend, StatTile, tooltipProps } from "@/components/charts";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import TaskToggle from "@/components/TaskToggle";
import { useManifest } from "@/components/useManifest";
import { millions, MODEL_COLOR, num, pct, sortModels, TASK_LABEL, type ModelInfo, type Task } from "@/lib/api";

const mins = (s: number) => (s >= 60 ? `${Math.floor(s / 60)} min ${s % 60}s` : `${s}s`);

export default function ResultsPage() {
  const { data, error } = useManifest();
  const [task, setTask] = useState<Task>("dr");
  const [pick, setPick] = useState<string | null>(null);

  const models = useMemo(() => sortModels((data?.models ?? []).filter((m) => m.task === task)), [data, task]);
  const best = models.reduce<ModelInfo | null>((b, m) => ((m.metrics.test_accuracy ?? 0) > (b?.metrics.test_accuracy ?? -1) ? m : b), null);
  const selected = models.find((m) => m.key === pick) ?? best ?? models[0];

  if (error) return <Shell><p className="mt-5 rounded-lg bg-red-500/10 p-3 text-sm text-red-300">Server unreachable: {error}</p></Shell>;
  if (!data) return <Shell><p className="mt-6 text-slate-400">Loading…</p></Shell>;
  if (!models.length)
    return (
      <Shell>
        <div className="mt-6 flex"><TaskToggle task={task} onChange={setTask} /></div>
        <p className="card mt-6 text-slate-300">No trained model for this task. Train, then run <code>python3 export_web.py</code>.</p>
      </Shell>
    );

  const second = task === "dr"
    ? { key: "roc_auc" as const, label: "ROC-AUC", fmt: (v: number) => v.toFixed(3) }
    : { key: "qwk" as const, label: "Quadratic weighted kappa (QWK)", fmt: (v: number) => v.toFixed(3) };
  const bars = models.map((m) => ({ key: m.key, name: m.name, acc: m.metrics.test_accuracy ?? 0, sec: m.metrics[second.key] ?? 0 }));

  // one row per epoch, one column per model (CNN trains 40 epochs, the pre-trained ones 10)
  const maxEp = Math.max(...models.map((m) => m.history.length));
  const valRows = Array.from({ length: maxEp }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    models.forEach((m) => {
      const h = m.history[i];
      if (h) row[m.key] = h.val_acc;
    });
    return row;
  });

  return (
    <Shell>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-slate-500">
          {TASK_LABEL[task]} · test set n = {models[0].metrics.test_images ?? "–"} unseen photos, the same for every model
        </p>
        <TaskToggle task={task} onChange={(t) => { setTask(t); setPick(null); }} />
      </div>

      {/* headline numbers */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Best model" value={best?.name ?? "–"} sub={best?.pretrained ? "pre-trained on ImageNet" : "trained from scratch"} />
        <StatTile label="Best test accuracy" value={pct(best?.metrics.test_accuracy)} sub={`validation ${pct(best?.metrics.val_accuracy)}`} />
        {task === "dr" ? (
          <StatTile label="Its sensitivity / specificity" value={`${pct(best?.metrics.sensitivity, 0)} / ${pct(best?.metrics.specificity, 0)}`} sub="DR eyes found / healthy eyes cleared" />
        ) : (
          <StatTile label="Its QWK" value={num(best?.metrics.qwk)} sub={`DR yes/no accuracy ${pct(best?.metrics.dr_yes_no_accuracy)}`} />
        )}
        <StatTile
          label="Spread across 5 models"
          value={`${(100 * (Math.max(...bars.map((b) => b.acc)) - Math.min(...bars.map((b) => b.acc)))).toFixed(1)} pts`}
          sub="best minus worst test accuracy"
        />
      </div>

      {/* comparison: two charts, never a dual axis */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <BarCard title="Test accuracy" data={bars} field="acc" fmt={(v) => pct(v)} />
        <BarCard title={second.label} data={bars} field="sec" fmt={second.fmt} />
      </div>

      <div className="mt-6">
      <ChartCard
        title="Validation accuracy per epoch"
        note="For each model the epoch with the best validation accuracy is kept (see the table). The CNN trains 40 epochs from scratch and jumps around early on; the pre-trained models start high and need only 10."
        legend={models.map((m) => ({ label: m.name, color: MODEL_COLOR[m.key] }))}
        height="h-72"
      >
        <LineChart data={valRows}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="epoch" {...axisProps} />
          <YAxis {...axisProps} width={44} domain={["auto", "auto"]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip {...tooltipProps} formatter={(v, n) => [pct(Number(v)), models.find((m) => m.key === n)?.name ?? n]} labelFormatter={(v) => `epoch ${v}`} />
          {models.map((m) => (
            <Line key={m.key} dataKey={m.key} stroke={MODEL_COLOR[m.key]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} connectNulls={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ChartCard>
      </div>

      {/* full table */}
      <div className="card mt-6 overflow-x-auto">
        <div className="mb-3 font-semibold">All metrics</div>
        <table className="tabular w-full whitespace-nowrap text-sm">
          <thead className="text-left text-xs text-slate-400">
            <tr>
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Type</th>
              <th className="py-2 pr-4 text-right">Weights</th>
              <th className="py-2 pr-4 text-right">Epochs (best)</th>
              <th className="py-2 pr-4 text-right">Train time</th>
              <th className="py-2 pr-4 text-right">Val acc</th>
              <th className="py-2 pr-4 text-right">Test acc</th>
              {task === "dr" ? (
                <>
                  <th className="py-2 pr-4 text-right">Sensitivity</th>
                  <th className="py-2 pr-4 text-right">Specificity</th>
                  <th className="py-2 pr-4 text-right">Precision</th>
                  <th className="py-2 text-right">ROC-AUC</th>
                </>
              ) : (
                <>
                  <th className="py-2 pr-4 text-right">QWK</th>
                  <th className="py-2 text-right">DR yes/no acc</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const mt = m.metrics;
              const isBest = m === best;
              return (
                <tr key={m.key} className="border-t border-slate-800">
                  <td className="py-2 pr-4">
                    <span className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR[m.key] }} />
                      {m.name}
                      {isBest && <span className="rounded-full border border-cyan-500/40 px-2 text-xs text-cyan-400">best</span>}
                    </span>
                  </td>
                  <td className="py-2 pr-4 text-slate-400">{m.pretrained ? "pre-trained" : "from scratch"}</td>
                  <td className="py-2 pr-4 text-right">{millions(m.parameters)}</td>
                  <td className="py-2 pr-4 text-right">{mt.epochs ?? "–"} ({mt.best_epoch ?? "–"})</td>
                  <td className="py-2 pr-4 text-right text-slate-400">{m.seconds ? mins(m.seconds) : "–"}</td>
                  <td className="py-2 pr-4 text-right">{pct(mt.val_accuracy)}</td>
                  <td className={`py-2 pr-4 text-right font-semibold ${isBest ? "text-cyan-300" : ""}`}>{pct(mt.test_accuracy)}</td>
                  {task === "dr" ? (
                    <>
                      <td className="py-2 pr-4 text-right">{pct(mt.sensitivity)}</td>
                      <td className="py-2 pr-4 text-right">{pct(mt.specificity)}</td>
                      <td className="py-2 pr-4 text-right">{pct(mt.precision)}</td>
                      <td className="py-2 text-right">{num(mt.roc_auc)}</td>
                    </>
                  ) : (
                    <>
                      <td className="py-2 pr-4 text-right">{num(mt.qwk)}</td>
                      <td className="py-2 text-right">{pct(mt.dr_yes_no_accuracy)}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-slate-500">
          Sensitivity = share of eyes with DR that were found · specificity = share of healthy eyes called healthy · QWK =
          agreement with the doctor that punishes far-off grades more (1 = perfect, the official APTOS metric). Train time
          measured on an Apple M-series GPU (MPS).
        </p>
      </div>

      {/* one model in detail */}
      <h2 className="mt-10 text-xl font-semibold">One model in detail</h2>
      <div className="mt-3 flex flex-wrap gap-2">
        {models.map((m) => (
          <button
            key={m.key}
            onClick={() => setPick(m.key)}
            className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${
              selected?.key === m.key ? "border-slate-500 bg-slate-800 text-white" : "border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLOR[m.key] }} />
            {m.name}
          </button>
        ))}
      </div>
      {selected && <Detail m={selected} />}
    </Shell>
  );
}

function Detail({ m }: { m: ModelInfo }) {
  const be = m.metrics.best_epoch;
  const cm = m.metrics.confusion_matrix;
  const perClass = cm
    ? m.classes.map((c, i) => {
        const tp = cm[i][i];
        const actual = cm[i].reduce((a, b) => a + b, 0);
        const predicted = cm.reduce((a, row) => a + row[i], 0);
        const recall = actual ? tp / actual : null;
        const precision = predicted ? tp / predicted : null;
        const f1 = recall != null && precision != null && recall + precision > 0 ? (2 * recall * precision) / (recall + precision) : null;
        return { c, n: actual, recall, precision, f1 };
      })
    : [];
  return (
    <>
      {m.history.length > 0 && (
        <div className="mt-4 grid gap-6 lg:grid-cols-2">
          <ChartCard title="Loss per epoch" legend={[{ label: "Training", color: "var(--series-1)" }, { label: "Validation", color: "var(--series-2)" }]}>
            <LineChart data={m.history}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epoch" {...axisProps} />
              <YAxis {...axisProps} width={44} domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
              <Tooltip {...tooltipProps} formatter={(v) => Number(v).toFixed(4)} labelFormatter={(v) => `epoch ${v}`} />
              {be && <ReferenceLine x={be} stroke="var(--axis)" strokeDasharray="4 4" label={{ value: "kept", fill: "var(--text-muted)", fontSize: 10, position: "insideTopRight" }} />}
              <Line dataKey="train_loss" name="Training" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              <Line dataKey="val_loss" name="Validation" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ChartCard>
          <ChartCard title="Accuracy per epoch" legend={[{ label: "Training", color: "var(--series-1)" }, { label: "Validation", color: "var(--series-2)" }]}>
            <LineChart data={m.history}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="epoch" {...axisProps} />
              <YAxis {...axisProps} width={44} domain={["auto", "auto"]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip {...tooltipProps} formatter={(v) => pct(Number(v))} labelFormatter={(v) => `epoch ${v}`} />
              {be && <ReferenceLine x={be} stroke="var(--axis)" strokeDasharray="4 4" />}
              <Line dataKey="train_acc" name="Training" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              <Line dataKey="val_acc" name="Validation" stroke="var(--series-2)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ChartCard>
        </div>
      )}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="card">
          <div className="mb-1 font-semibold">Confusion matrix (test)</div>
          <p className="mb-3 text-xs text-slate-500">Colour = share of the true class. The diagonal is correct.</p>
          {cm ? <ConfusionMatrix cm={cm} classes={m.classes} /> : <p className="text-slate-500">Not available.</p>}
        </div>
        <div className="card overflow-x-auto">
          <div className="mb-3 font-semibold">Per-class results (test)</div>
          <table className="tabular w-full text-sm">
            <thead className="text-left text-xs text-slate-400">
              <tr>
                <th className="py-2">Class</th>
                <th className="py-2 text-right">Photos</th>
                <th className="py-2 text-right">Recall</th>
                <th className="py-2 text-right">Precision</th>
                <th className="py-2 text-right">F1</th>
              </tr>
            </thead>
            <tbody>
              {perClass.map((r) => (
                <tr key={r.c} className="border-t border-slate-800">
                  <td className="py-2">{r.c}</td>
                  <td className="py-2 text-right text-slate-400">{r.n}</td>
                  <td className="py-2 text-right">{pct(r.recall)}</td>
                  <td className="py-2 text-right">{pct(r.precision)}</td>
                  <td className="py-2 text-right">{num(r.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-slate-500">
            {m.name}: {m.optimizer ?? "–"}. Kept epoch {be ?? "–"} of {m.metrics.epochs ?? "–"}.
          </p>
        </div>
      </div>
    </>
  );
}

function BarCard({ title, data, field, fmt }: {
  title: string;
  data: { key: string; name: string; acc: number; sec: number }[];
  field: "acc" | "sec";
  fmt: (v: number) => string;
}) {
  return (
    <div className="card">
      <div className="font-semibold">{title}</div>
      <p className="mt-1 text-xs text-slate-500">Scale starts at 0 – the models are close.</p>
      <div className="mt-3" style={{ height: 44 * data.length + 30 }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 56 }} barCategoryGap={6}>
            <CartesianGrid {...gridProps} horizontal={false} vertical />
            <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axisProps} tickFormatter={(v) => (field === "acc" ? `${Math.round(v * 100)}%` : Number(v).toFixed(2))} />
            <YAxis type="category" dataKey="name" {...axisProps} width={128} />
            <Tooltip {...tooltipProps} formatter={(v) => fmt(Number(v))} />
            <Bar dataKey={field} name={title} radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.key} fill={MODEL_COLOR[d.key]} />
              ))}
              <LabelList dataKey={field} position="right" fill="var(--text-secondary)" fontSize={11} formatter={(v: unknown) => fmt(Number(v))} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <h1 className="text-3xl font-bold">Training results</h1>
      {children}
    </main>
  );
}

function ChartCard({ title, note, legend, children, height = "h-64" }: {
  title: string;
  note?: string;
  legend?: { label: string; color: string; dashed?: boolean }[];
  children: React.ReactElement;
  height?: string;
}) {
  return (
    <div className="card">
      <div className="mb-1 font-semibold">{title}</div>
      {legend && <Legend items={legend} />}
      <div className={`mt-3 ${height}`}>
        <ResponsiveContainer>{children}</ResponsiveContainer>
      </div>
      {note && <p className="mt-2 text-xs text-slate-500">{note}</p>}
    </div>
  );
}
