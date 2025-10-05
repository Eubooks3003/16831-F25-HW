#!/usr/bin/env python3
"""
Plot Q3 (LunarLanderContinuous-v2) learning curve(s) from ./data/q3_*.

Default: pick the most recent run with >= --min_iters scalars and save a single curve.
Use --mode all to overlay all q3 runs (useful for multiple seeds).

Example:
  python scripts/plot_q3.py
  python scripts/plot_q3.py --mode all
"""

import os, glob, argparse, time
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG_CANDIDATES = [
    "Eval_AverageReturn",
    "Evaluation_AverageReturn",
    "AverageReturn",
    "eval/avg_return",
    "Train_AverageReturn",
]

def latest_event(run_dir: str):
    cand = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not cand:
        cand = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return cand[-1] if cand else None

def load_curve(evt_path: str):
    try:
        ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        tag = next((t for t in TAG_CANDIDATES if t in tags), None)
        if not tag:
            return None, None, None
        s = ea.Scalars(tag)
        steps = [x.step for x in s]
        vals  = [x.value for x in s]
        return steps, vals, tag
    except Exception:
        return None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="plots/q3_curve.png")
    ap.add_argument("--mode", choices=["single","all"], default="single")
    ap.add_argument("--min_iters", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    run_dirs = [d for d in glob.glob(os.path.join(args.data_dir, "q3_*")) if os.path.isdir(d)]
    if not run_dirs:
        print(f"No q3_* directories in {args.data_dir}")
        return

    # collect curves
    curves = []
    for d in run_dirs:
        evt = latest_event(d)
        if not evt: continue
        steps, vals, tag = load_curve(evt)
        if not steps or len(steps) < args.min_iters:  # require enough iterations
            continue
        mtime = os.path.getmtime(evt)
        curves.append({"dir": d, "evt": evt, "steps": steps, "vals": vals, "tag": tag, "mtime": mtime})

    if not curves:
        print("Found q3_* runs, but none with enough iterations.")
        return

    plt.figure(figsize=(8,5))

    if args.mode == "all":
        for c in sorted(curves, key=lambda x: x["mtime"]):
            label = os.path.basename(c["dir"])
            plt.plot(c["steps"], c["vals"], label=label, linewidth=1.8)
        title = "LunarLanderContinuous-v2 – Q3 (all q3_* runs)"
    else:
        # pick most recent completed run
        best = max(curves, key=lambda x: x["mtime"])
        label = os.path.basename(best["dir"])
        plt.plot(best["steps"], best["vals"], label=label, linewidth=2.2)
        title = "LunarLanderContinuous-v2 – Q3 (most recent run)"

    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[Saved] {args.out}")

if __name__ == "__main__":
    main()
