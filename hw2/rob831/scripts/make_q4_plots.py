#!/usr/bin/env python3
# plot_q4_halfcheetah.py
import os, re, glob, argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Common tag fallbacks for the HW logger
EVAL_TAGS = ["Eval_AverageReturn", "Evaluation_AverageReturn", "AverageReturn", "eval/avg_return"]

# Match run folders like: q4_search_b15000_lr0.01_rtg_nnbaseline_...
RUN_GLOB = "q4_search_b*_lr*_rtg_nnbaseline*"

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
    x = np.arange(len(y), dtype=float)  # iterations on x-axis
    return x, y, ytag

def parse_b_lr(run_name: str):
    # e.g., q4_search_b15000_lr0.01_rtg_nnbaseline
    m = re.search(r"b(\d+).*?lr([0-9.]+(?:e-?\d+)?)", run_name)
    b = int(m.group(1)) if m else None
    lr = float(m.group(2)) if m else None
    return b, lr

def ema(y, w=0.8):
    if len(y) == 0: return y
    s = np.empty_like(y, dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = w * s[i-1] + (1 - w) * y[i]
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="where q4_* run folders live")
    ap.add_argument("--out_dir", default="plots", help="where to save plot/csv")
    ap.add_argument("--smooth", type=float, default=0.8, help="EMA smoothing weight (0..1)")
    ap.add_argument("--min_len", type=int, default=20, help="minimum points to include a run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # discover runs
    candidates = sorted(glob.glob(os.path.join(args.data_dir, RUN_GLOB)))
    runs = []
    for d in candidates:
        if not os.path.isdir(d): 
            continue
        evt = latest_tfevents(d)
        if not evt: 
            continue
        x, y, ytag = read_eval_curve(evt)
        if x is None or len(x) < args.min_len: 
            continue
        name = os.path.basename(d)
        b, lr = parse_b_lr(name)
        ys = ema(y, args.smooth) if 0.0 <= args.smooth < 1.0 else y
        runs.append({"dir": d, "name": name, "b": b, "lr": lr, "x": x, "y": y, "ys": ys, "ytag": ytag})

    if not runs:
        print("No usable HalfCheetah Q4 runs found.")
        return

    # sort for tidy legend: by batch size, then lr
    runs.sort(key=lambda r: (r["b"] if r["b"] is not None else 1e18, r["lr"] if r["lr"] is not None else 1e18))

    # plot
    plt.figure(figsize=(10, 6))
    for r in runs:
        label = f"b={r['b']}, lr={r['lr']}" if r["b"] is not None else r["name"]
        plt.plot(r["x"], r["ys"], linewidth=1.8, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(runs[0]["ytag"] or "Eval_AverageReturn")
    plt.title("HalfCheetah-v4 (ep_len=150): Learning Curves (Q4 sweep)")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=8, ncol=1)
    out_png = os.path.join(args.out_dir, "q4_halfcheetah_all.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Saved] {out_png}")

    # also dump curves to CSV for report
    out_csv = os.path.join(args.out_dir, "q4_halfcheetah_curves.csv")
    max_len = max(len(r["x"]) for r in runs)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["iter"]
        for r in runs:
            header.append(f"{r['b']}_{r['lr']}")
        w.writerow(header)
        for i in range(max_len):
            row = [i]
            for r in runs:
                row.append(r["ys"][i] if i < len(r["ys"]) else "")
            w.writerow(row)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
