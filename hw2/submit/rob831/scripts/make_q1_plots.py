#!/usr/bin/env python3
# plot_q1_split_iter.py
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAGS = ["Eval_AverageReturn", "Evaluation_AverageReturn", "AverageReturn", "eval/avg_return"]

RUN_GROUPS = {
    "sb": ["q1_sb_no_rtg_dsa", "q1_sb_rtg_dsa", "q1_sb_rtg_na"],
    "lb": ["q1_lb_no_rtg_dsa", "q1_lb_rtg_dsa", "q1_lb_rtg_na"],
}

LABELS = {
    "q1_sb_no_rtg_dsa": "SB: no RTG, no std-adv",
    "q1_sb_rtg_dsa":    "SB: RTG, no std-adv",
    "q1_sb_rtg_na":     "SB: RTG, std-adv",
    "q1_lb_no_rtg_dsa": "LB: no RTG, no std-adv",
    "q1_lb_rtg_dsa":    "LB: RTG, no std-adv",
    "q1_lb_rtg_na":     "LB: RTG, std-adv",
}

def find_event_file(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    return files[0] if files else None

def load_eval_curve(evt_path):
    ea = EventAccumulator(evt_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    ytag = next((t for t in EVAL_TAGS if t in tags), None)
    if ytag is None:
        raise ValueError(f"No eval-return tag found in {evt_path}. Have: {sorted(tags)}")
    ys = ea.Scalars(ytag)
    y = np.array([s.value for s in ys], dtype=float)
    iters = np.arange(len(y), dtype=float)  # <-- iteration index on x-axis
    return iters, y, ytag

def ema(y, w=0.8):
    if len(y) == 0: return y
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = w * out[i-1] + (1 - w) * y[i]
    return out

def latest_run_dir(data_dir, prefix):
    cands = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_CartPole-v0_*")))
    return cands[-1] if cands else None

def plot_group(group_name, prefixes, data_dir, out_dir, smooth):
    curves = {}
    used_ytag = None

    for pref in prefixes:
        rdir = latest_run_dir(data_dir, pref)
        if not rdir:
            print(f"[WARN] No folder found for {pref}")
            continue
        evt = find_event_file(rdir)
        if not evt:
            print(f"[WARN] No tfevents in {rdir}")
            continue
        try:
            x, y, ytag = load_eval_curve(evt)
            y_s = ema(y, smooth) if smooth is not None and 0 <= smooth < 1 else y
            curves[pref] = (x, y_s)
            used_ytag = used_ytag or ytag
            print(f"[OK] {pref}: {len(x)} iters (eval='{ytag}')")
        except Exception as e:
            print(f"[WARN] Skip {pref}: {e}")

    if not curves:
        print(f"[WARN] No curves for group {group_name}")
        return

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,5))
    for pref, (x, y) in curves.items():
        plt.plot(x, y, label=LABELS.get(pref, pref))
    plt.xlabel("Iteration")                               # <-- iteration on x-axis
    plt.ylabel(used_ytag or "Eval_AverageReturn")
    plt.title(f"Q1 ({group_name.upper()}): CartPole-v0 — Eval Average Return vs Iteration")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f"q1_{group_name}_eval_returns_iter.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--smooth", type=float, default=0.8)
    args = ap.parse_args()

    plot_group("sb", RUN_GROUPS["sb"], args.data_dir, args.out_dir, args.smooth)
    plot_group("lb", RUN_GROUPS["lb"], args.data_dir, args.out_dir, args.smooth)

if __name__ == "__main__":
    main()
