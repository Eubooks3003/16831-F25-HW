#!/usr/bin/env python3
# plot_q4_b10000_lr0.02_all.py
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAGS = ["Eval_AverageReturn", "Evaluation_AverageReturn", "AverageReturn", "eval/avg_return"]

RUN_NAMES = [
    "q4_search_b10000_lr0.02",                 # Plain
    "q4_search_b10000_lr0.02_rtg",             # RTG
    "q4_search_b10000_lr0.02_nnbaseline",      # Baseline
    "q4_search_b10000_lr0.02_rtg_nnbaseline",  # RTG + Baseline
]

LABELS = {
    "q4_search_b10000_lr0.02": "Plain",
    "q4_search_b10000_lr0.02_rtg": "RTG",
    "q4_search_b10000_lr0.02_nnbaseline": "Baseline",
    "q4_search_b10000_lr0.02_rtg_nnbaseline": "RTG + Baseline",
}

def latest_tfevents(run_dir: str):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return files[-1] if files else None

def read_eval_curve(evt_path: str):
    ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    ytag = next((t for t in EVAL_TAGS if t in tags), None)
    if ytag is None:
        return None, None, None
    scalars = ea.Scalars(ytag)
    y = np.array([s.value for s in scalars], dtype=float)
    x = np.arange(len(y), dtype=float)  # iteration index
    return x, y, ytag

def ema(y, w=0.8):
    if len(y) == 0: return y
    s = np.empty_like(y, dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = w * s[i-1] + (1 - w) * y[i]
    return s

def main(data_dir="data", out_dir="plots", smooth=0.8, target=200.0):
    os.makedirs(out_dir, exist_ok=True)

    curves = []
    for name in RUN_NAMES:
        # pick the newest directory whose basename starts exactly with this run name
        dirs = sorted([d for d in glob.glob(os.path.join(data_dir, name + "*")) if os.path.isdir(d)])
        if not dirs:
            print(f"[WARN] missing run folder for {name}")
            continue
        run_dir = dirs[-1]
        evt = latest_tfevents(run_dir)
        if not evt:
            print(f"[WARN] no tfevents in {run_dir}")
            continue
        x, y, ytag = read_eval_curve(evt)
        if x is None or len(x) < 5:
            print(f"[WARN] not enough points for {name}")
            continue
        y_plot = ema(y, smooth) if 0.0 <= smooth < 1.0 else y
        curves.append((LABELS.get(name, name), x, y_plot, ytag))
        print(f"[OK] {name}: {len(x)} iters from {os.path.basename(run_dir)}")

    if not curves:
        print("No usable runs found.")
        return

    plt.figure(figsize=(10, 6))
    for label, x, y, ytag in curves:
        plt.plot(x, y, linewidth=1.8, label=label)
    if target is not None:
        plt.axhline(target, ls="--", lw=1.0, color="gray", label=f"target {target:g}")

    plt.xlabel("Iteration")
    plt.ylabel(curves[0][3] or "Eval_AverageReturn")
    plt.title("HalfCheetah-v4 (ep_len=150): b=10000, lr=0.02 — All variants")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=9)

    out_path = os.path.join(out_dir, "q4_b10000_lr0.02_all.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()
