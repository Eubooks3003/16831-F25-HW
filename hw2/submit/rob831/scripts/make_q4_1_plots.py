#!/usr/bin/env python3
# plot_q4_best_br.py  (fixed variant-matching)
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAGS = ["Eval_AverageReturn", "Evaluation_AverageReturn", "AverageReturn", "eval/avg_return"]

def latest_tfevents(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    return files[-1] if files else None

def read_eval_curve(evt_path):
    ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    ytag = next((t for t in EVAL_TAGS if t in tags), None)
    if ytag is None:
        return None, None, None
    s = ea.Scalars(ytag)
    y = np.array([p.value for p in s], dtype=float)
    x = np.arange(len(y), dtype=float)  # iteration index
    return x, y, ytag

def ema(y, w):
    if w is None or not (0.0 <= w < 1.0) or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = w * out[i-1] + (1 - w) * y[i]
    return out

def pick_run_dir(data_dir, base, need_rtg=False, need_nn=False):
    """
    Choose the newest folder that matches *exactly* the requested flags.
    - Plain            -> no '_rtg' and no '_nnbaseline'
    - RTG              ->     '_rtg' and no '_nnbaseline'
    - Baseline         -> no '_rtg' and     '_nnbaseline'
    - RTG + Baseline   ->     '_rtg' and     '_nnbaseline'
    """
    cands = sorted(glob.glob(os.path.join(data_dir, base + "*")))
    good = []
    for d in cands:
        name = os.path.basename(d)
        has_rtg = "_rtg" in name
        has_nn  = "_nnbaseline" in name
        if has_rtg == need_rtg and has_nn == need_nn:
            good.append(d)
    return good[-1] if good else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--b", type=int, required=True, help="batch size, e.g., 15000")
    ap.add_argument("--lr", type=float, required=True, help="learning rate, e.g., 0.02")
    ap.add_argument("--smooth", type=float, default=0.8)
    ap.add_argument("--min_len", type=int, default=15)
    ap.add_argument("--target", type=float, default=200.0, help="reference score line")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base = f"q4_search_b{args.b}_lr{args.lr}"

    variants = [
        ("Plain",            pick_run_dir(args.data_dir, base, need_rtg=False, need_nn=False)),
        ("RTG",              pick_run_dir(args.data_dir, base, need_rtg=True,  need_nn=False)),
        ("Baseline",         pick_run_dir(args.data_dir, base, need_rtg=False, need_nn=True)),
        ("RTG + Baseline",   pick_run_dir(args.data_dir, base, need_rtg=True,  need_nn=True)),
    ]

    curves = []
    for label, run_dir in variants:
        if not run_dir:
            print(f"[WARN] Missing run for: {label}")
            continue
        evt = latest_tfevents(run_dir)
        if not evt:
            print(f"[WARN] No tfevents in {run_dir}")
            continue
        x, y, ytag = read_eval_curve(evt)
        if x is None or len(x) < args.min_len:
            print(f"[WARN] Too few points for: {label} ({run_dir})")
            continue
        y_plot = ema(y, args.smooth)
        curves.append((label, x, y_plot, ytag))
        print(f"[OK] {label} <- {os.path.basename(run_dir)} ({len(x)} iters)")

    if not curves:
        print("No usable runs found for requested b/lr.")
        return

    # Plot
    plt.figure(figsize=(9.5, 5.5))
    for label, x, y, ytag in curves:
        plt.plot(x, y, linewidth=1.8, label=label)
    if args.target is not None:
        plt.axhline(args.target, ls="--", lw=1.0, color="gray", label=f"target {args.target:g}")

    plt.xlabel("Iteration")
    plt.ylabel(curves[0][3] or "Eval_AverageReturn")
    plt.title(f"HalfCheetah-v4 (ep_len=150): Best b={args.b}, lr={args.lr}")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=9)
    out_png = os.path.join(args.out_dir, f"q4_best_b{args.b}_lr{args.lr}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    print(f"[Saved] {out_png}")

if __name__ == "__main__":
    main()
