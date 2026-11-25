"use client";

import { useManifest } from "@/components/useManifest";
import { pct, num, type ModelInfo } from "@/lib/api";

function best(models: ModelInfo[], task: "dr" | "grade") {
  return models
    .filter((m) => m.task === task)
    .reduce<ModelInfo | null>((b, m) => ((m.metrics.test_accuracy ?? 0) > (b?.metrics.test_accuracy ?? -1) ? m : b), null);
}

/** Headline numbers, read live from the trained models' test results. */
export default function HomeStats() {
  const { data } = useManifest();
  const models = data?.models ?? [];
  const dr = best(models, "dr");
  const grade = best(models, "grade");
  const testN = models[0]?.metrics.test_images;
  const cards = [
    { value: String(new Set(models.map((m) => m.key)).size || 5), title: "Models compared", text: "4 pre-trained on ImageNet + 1 CNN from scratch" },
    { value: dr ? pct(dr.metrics.test_accuracy) : "–", title: "Best DR detection", text: dr ? `${dr.name} · ROC-AUC ${num(dr.metrics.roc_auc)}` : "No DR / DR" },
    { value: grade ? pct(grade.metrics.test_accuracy) : "–", title: "Best 5-grade accuracy", text: grade ? `${grade.name} · QWK ${num(grade.metrics.qwk)}` : "5 DR grades" },
    { value: testN ? String(testN) : "548", title: "Unseen test photos", text: "Same photos for every model – never used in training" },
  ];
  return (
    <div className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((c) => (
        <div key={c.title} className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
          <div className="tabular text-3xl font-bold text-cyan-400">{c.value}</div>
          <h3 className="mt-3 font-semibold">{c.title}</h3>
          <p className="mt-2 text-sm text-slate-500">{c.text}</p>
        </div>
      ))}
    </div>
  );
}
