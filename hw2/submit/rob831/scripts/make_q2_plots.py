#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "Eval_AverageReturn", "Evaluation_AverageReturn",
    "AverageReturn", "eval/avg_return", "Train_AverageReturn",
]

def read_curve(evt_path):
    ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = next((t for t in TAGS if t in tags), None)
    if not tag:
        return None, None, None
    s = ea.Scalars(tag)
    return [x.step for x in s], [x.value for x in s], tag

def latest_event(run_dir):
    c = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not c:
        c = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return c[-1] if c else None

def parse_b_lr(name):
    m = re.search(r"q2_b(\d+)_r([0-9.eE+-]+)", name)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))

def first_hit(x, y, thresh):
    for i, v in enumerate(y):
        if v >= thresh:
            return i, x[i], v
    return None, None, None

def plot_single(run, out_dir, thresh, use_step):
    x = run["steps"] if use_step else list(range(len(run["vals"])))
    y = run["vals"]
    k, t, v = first_hit(x, y, thresh)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(x, y, linewidth=1.8, label=run["label"])
    if k is not None:
        plt.scatter([t], [v], s=28, zorder=3)
    plt.axhline(thresh, linestyle="--", linewidth=1.0, color="gray", label=f"threshold {thresh:g}")
    plt.xlabel("Step" if use_step else "Iteration")
    plt.ylabel(run["tag"] or "Average Return")
    title = f"InvertedPendulum-v4 — {run['label']}"
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{run['label']}.png")
    out = out.replace("/", "_")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[Saved] {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir",  default="plots")
    ap.add_argument("--thresh", type=float, default=950.0)
    ap.add_argument("--min_len", type=int, default=100, help="min #points to include a run")
    ap.add_argument("--use_step", action="store_true", help="use TB scalar step on x-axis instead of iteration index")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run_dirs = [d for d in glob.glob(os.path.join(args.data_dir, "q2_*")) if os.path.isdir(d)]
    if not run_dirs:
        print("No q2_* directories found under", args.data_dir)
        return

    runs = []
    for d in sorted(run_dirs):
        evt = latest_event(d)
        if not evt:
            continue
        steps, vals, tag = read_curve(evt)
        if not steps or len(steps) < args.min_len:
            continue
        b, lr = parse_b_lr(os.path.basename(d))
        label = os.path.basename(d)
        runs.append({
            "dir": d, "label": label, "b": b, "lr": lr, "tag": tag,
            "steps": steps, "vals": vals
        })

    if not runs:
        print("No usable q2_* runs with >= min_len points.")
        return

    # ---- Combined plot (all runs) ----
    plt.figure(figsize=(9.5, 5.5))
    for r in runs:
        x = r["steps"] if args.use_step else list(range(len(r["vals"])))
        y = r["vals"]
        k, t, v = first_hit(x, y, args.thresh)
        plt.plot(x, y, linewidth=1.5, label=r["label"])
        if k is not None:
            plt.scatter([t], [v], s=24)
    plt.axhline(args.thresh, linestyle="--", linewidth=1.0, color="gray", label=f"threshold {args.thresh:g}")
    plt.xlabel("Step" if args.use_step else "Iteration")
    plt.ylabel(runs[0]["tag"] or "Average Return")
    plt.title("InvertedPendulum-v4 – All Q2 runs")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    if len(runs) <= 12:
        plt.legend(loc="best", fontsize=8)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False)
        plt.tight_layout(rect=[0,0,0.78,1])
    out = os.path.join(args.out_dir, "q2_all.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[Saved] {out}")

    # ---- Individual plots ----
    indiv_dir = os.path.join(args.out_dir, "q2_individual")
    os.makedirs(indiv_dir, exist_ok=True)
    for r in runs:
        plot_single(r, indiv_dir, args.thresh, args.use_step)

if __name__ == "__main__":
    main()
