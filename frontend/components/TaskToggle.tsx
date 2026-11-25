"use client";

import type { Task } from "@/lib/api";

export default function TaskToggle({ task, onChange }: { task: Task; onChange: (t: Task) => void }) {
  return (
    <div className="flex gap-1 rounded-lg border border-slate-800 p-1 text-sm">
      {(["dr", "grade"] as const).map((t) => (
        <button
          key={t}
          onClick={() => onChange(t)}
          className={`rounded-md px-3 py-1.5 ${task === t ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"}`}
        >
          {t === "dr" ? "DR detection (2 classes)" : "DR grade (5 classes)"}
        </button>
      ))}
    </div>
  );
}
