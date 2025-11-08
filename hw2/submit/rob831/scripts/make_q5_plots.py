#!/usr/bin/env python3
# plot_q5_hopper_lambda.py
import os, re, glob, argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAGS = ["Eval_AverageReturn", "Evaluation_AverageReturn", "AverageReturn", "eval/avg_return"]
RUN_GLOB = "q5_*lambda*"

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
    x = np.arange(len(y), dtype=float)  # iterations on x-axis
    return x, y, ytag

def parse_lambda(run_name):
    # matches ..._lambda0, _lambda0.95, _lambda1, etc.
    m = re.search(r"lambda([0-9.]+)", run_name)
    return float(m.group(1)) if m else None

def ema(y, w=0.8):
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = w * out[i-1] + (1 - w) * y[i]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="where q5_* folders live")
    ap.add_argument("--out_dir", default="plots", help="output dir")
    ap.add_argument("--smooth", type=float, default=0.8, help="EMA smoothing weight (0..1)")
    ap.add_argument("--min_len", type=int, default=20, help="min points to include a run")
    ap.add_argument("--threshold", type=float, default=400.0, help="reference line for target score")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    candidates = sorted(glob.glob(os.path.join(args.data_dir, RUN_GLOB)))
    curves = []
    for d in candidates:
        if not os.path.isdir(d): 
            continue
        evt = latest_tfevents(d)
        if not evt:
            continue
        x, y, ytag = read_eval_curve(evt)
        if x is None or len(x) < args.min_len:
            continue
        lam = parse_lambda(os.path.basename(d))
        ys = ema(y, args.smooth) if 0.0 <= args.smooth < 1.0 else y
        curves.append({"name": os.path.basename(d), "lambda": lam, "x": x, "y": y, "ys": ys, "ytag": ytag})

    if not curves:
        print("No usable Hopper Q5 runs found.")
        return

    # sort legend by lambda
    curves.sort(key=lambda r: (r["lambda"] if r["lambda"] is not None else 1e9))

    # plot all curves
    plt.figure(figsize=(10, 6))
    for c in curves:
        label = f"λ={c['lambda']}" if c["lambda"] is not None else c["name"]
        plt.plot(c["x"], c["ys"], linewidth=1.8, label=label)
    if args.threshold is not None:
        plt.axhline(args.threshold, ls="--", lw=1.0, color="gray", label=f"target {args.threshold:g}")
    plt.xlabel("Iteration")
    plt.ylabel(curves[0]["ytag"] or "Eval_AverageReturn")
    plt.title("Hopper-v4: Effect of GAE λ on Learning (Q5)")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(loc="best", fontsize=9)
    out_png = os.path.join(args.out_dir, "q5_hopper_lambda_all.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[Saved] {out_png}")

    # CSV export (smoothed values)
    out_csv = os.path.join(args.out_dir, "q5_hopper_lambda_curves.csv")
    max_len = max(len(c["ys"]) for c in curves)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["iter"] + [f"lambda_{c['lambda']}" for c in curves]
        w.writerow(header)
        for i in range(max_len):
            row = [i]
            for c in curves:
                row.append(c["ys"][i] if i < len(c["ys"]) else "")
            w.writerow(row)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
