#!/usr/bin/env python3
import os, glob, re, argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "Eval_AverageReturn", "Evaluation_AverageReturn",
    "AverageReturn", "eval/avg_return", "Train_AverageReturn",
]

LABEL_ALIASES = [
    (re.compile(r"ms_pg_baseline", re.I),       "Single step (1x, full batch)"),
    (re.compile(r"ms_pg_fullbatch5", re.I),     "Multi-step (5x, full batch)"),
    (re.compile(r"ms_pg_minibatch5_2048", re.I),"Multi-step (5x, minibatch=2048)"),
]

def latest_event(run_dir: str):
    c = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not c:
        c = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return c[-1] if c else None

def read_curve(evt_path):
    ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in TAGS if t in tags), None)
    if not tag:
        return None, None, None
    s = ea.Scalars(tag)
    steps = np.array([x.step for x in s], dtype=np.int32)
    vals  = np.array([x.value for x in s], dtype=np.float32)
    return steps, vals, tag

def smooth(y, kw=5):
    if kw <= 1: return y
    kw = min(kw, len(y))
    k = np.ones(kw, dtype=np.float32) / kw
    return np.convolve(y, k, mode="same")

def make_label(run_dir_name: str):
    for pat, label in LABEL_ALIASES:
        if pat.search(run_dir_name):
            return label
    return run_dir_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--env", default="HalfCheetah-v4")
    ap.add_argument("--out", default="plots/ms_pg_compare.png")
    ap.add_argument("--smooth", type=int, default=7, help="moving-average window (iterations)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # find relevant runs
    patt = os.path.join(args.data_dir, f"ms_pg_*_{args.env}_*")
    
    print(f"Looking for runs matching {patt}")
    run_dirs = [d for d in glob.glob(patt) if os.path.isdir(d)]
    if not run_dirs:
        print(f"No runs found matching {patt}")
        return

    # sort for stable legend order
    run_dirs.sort()

    plt.figure(figsize=(10, 6))
    found = 0
    legend_entries = []
    for d in run_dirs:
        evt = latest_event(d)
        if not evt:
            continue
        steps, vals, tag = read_curve(evt)
        if steps is None:
            continue
        # smooth & plot
        y = smooth(vals, args.smooth) if args.smooth > 1 else vals
        label = make_label(os.path.basename(d))
        plt.plot(steps, y, linewidth=2, label=label)
        found += 1

    if not found:
        print("No scalar curves found in the matched runs.")
        return

    # cosmetics
    plt.xlabel("Iteration")
    plt.ylabel("Eval_AverageReturn")
    plt.title(f"{args.env}: Multiple-Gradient-Steps Comparison")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[Saved] {args.out}")

if __name__ == "__main__":
    main()
