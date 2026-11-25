"use client";

/** Horizontal bar per class. The predicted class is drawn in the accent colour, the rest muted;
 *  every bar is labelled with its value. */
export default function ProbBars({ classes, probs, pred, color = "var(--series-1)" }: {
  classes: string[];
  probs: number[];
  pred: number;
  color?: string;
}) {
  return (
    <div className="space-y-2" role="table" aria-label="class probabilities">
      {classes.map((c, i) => {
        const pct = probs[i] * 100;
        const active = i === pred;
        return (
          <div key={c} role="row" className="group" title={`${c}: ${pct.toFixed(1)}%`}>
            <div className="mb-1 flex justify-between text-xs">
              <span role="cell" className={active ? "font-semibold text-slate-100" : "text-slate-400"}>
                {c}
              </span>
              <span role="cell" className="tabular text-slate-300">
                {pct.toFixed(1)}%
              </span>
            </div>
            <div className="h-2 w-full rounded bg-slate-800">
              <div
                className="h-2 rounded transition-all group-hover:brightness-125"
                style={{ width: `${Math.max(pct, 0.5)}%`, background: active ? color : "#475569" }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
