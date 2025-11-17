#!/usr/bin/env python3
import os, re, sys, argparse, shutil, glob, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# tensorboard event reader
from tensorboard.backend.event_processing import event_accumulator as ea

# ---------- helpers ----------
def newest_dir(paths):
    """Return path with latest mtime; paths is a list of Path."""
    if not paths: return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def find_event_file(run_dir: Path):
    ev = sorted(run_dir.glob("events.out.tfevents.*"))
    return ev[-1] if ev else None

def load_scalars(event_file: Path):
    acc = ea.EventAccumulator(str(event_file), size_guidance={
        ea.SCALARS: 0,
        ea.HISTOGRAMS: 0,
        ea.IMAGES: 0,
        ea.AUDIO: 0,
        ea.COMPRESSED_HISTOGRAMS: 0,
        ea.TENSORS: 0,
    })
    acc.Reload()
    scalars = {}
    for tag in acc.Tags().get('scalars', []):
        sv = acc.Scalars(tag)
        steps = np.array([s.step for s in sv], dtype=int)
        vals = np.array([s.value for s in sv], dtype=float)
        scalars[tag] = (steps, vals)
    return scalars

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_latest_run(data_dir: Path, exp_key: str):
    """Return latest run dir for experiment key (prefix of folder names)."""
    # Folders look like: hw4_<exp_key>_<env>_<timestamp>...
    # Match those starting with 'hw4_' + exp_key + '_'
    patt = f"hw4_{exp_key}_*"
    candidates = [d for d in data_dir.glob(patt) if d.is_dir()]
    return newest_dir(candidates)

def plot_lines(x, series, labels, title, out_png, xlabel="Iteration", ylabel="AverageReturn"):
    plt.figure()
    for y, lbl in zip(series, labels):
        if y is None: continue
        plt.plot(x, y, label=lbl)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    ensure(out_png.parent)
    plt.savefig(out_png, bbox_inches='tight', dpi=160)
    plt.close()

def bar_plot(labels, values, title, out_png, ylabel="Final Eval AverageReturn"):
    plt.figure()
    idx = np.arange(len(labels))
    plt.bar(idx, values)
    plt.xticks(idx, labels, rotation=20)
    plt.ylabel(ylabel); plt.title(title); plt.grid(True, axis='y', alpha=0.3)
    ensure(out_png.parent)
    plt.savefig(out_png, bbox_inches='tight', dpi=160)
    plt.close()

# Tags we’ll look for (logger names sometimes vary slightly)
CANDIDATE_EVAL_TAGS  = ["Eval_AverageReturn", "Eval AverageReturn", "Eval/AverageReturn"]
CANDIDATE_TRAIN_TAGS = ["Train_AverageReturn", "Train AverageReturn", "Train/AverageReturn"]

def get_best_tag(scalars, candidates):
    for t in candidates:
        if t in scalars:
            return t
    # try fuzzy contains
    for t in scalars.keys():
        for c in candidates:
            if c.replace(" ", "_") in t.replace(" ", "_"):
                return t
    return None

# ---------- main workflow ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path to hw4 run logs (e.g., ./rob831/data)")
    ap.add_argument("--out_dir",  type=str, default="./figs", help="Where to save compiled figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    ensure(out_dir)

    # -------------- Q1: copy qualitative prediction + loss plots --------------
    q1_exps = [
        "q1_cheetah_n500_arch1x16",
        "q1_cheetah_n10_arch2x200",
        "q1_cheetah_n500_arch2x200",
    ]
    for key in q1_exps:
        run = pick_latest_run(data_dir, key)
        if not run:
            print(f"[Q1] Missing run for {key}")
            continue
        for fname in ["itr_0_predictions.png", "itr_0_losses.png"]:
            src = run / fname
            if src.exists():
                dst = out_dir / "Q1" / f"{key}_{fname}"
                ensure(dst.parent)
                shutil.copy2(src, dst)
                print(f"[Q1] Copied {src.name} -> {dst}")
            else:
                print(f"[Q1] Not found: {src}")

    # -------------- Q2: single-iteration train vs eval (scatter/points) --------------
    key = "q2_obstacles_singleiteration"
    run = pick_latest_run(data_dir, key)
    if run:
        ev = find_event_file(run)
        if ev:
            sc = load_scalars(ev)
            eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
            train_tag = get_best_tag(sc, CANDIDATE_TRAIN_TAGS)
            if eval_tag and train_tag:
                # Take last value of each (single-iter anyway)
                tr_last = sc[train_tag][1][-1]
                ev_last = sc[eval_tag][1][-1]
                # tiny plot
                plt.figure()
                plt.scatter([0],[tr_last], label="Train_AverageReturn")
                plt.scatter([1],[ev_last], label="Eval_AverageReturn")
                plt.xticks([0,1], ["Train","Eval"])
                plt.ylabel("AverageReturn"); plt.title("Q2: Train vs Eval (single iter)")
                plt.legend(); plt.grid(True, alpha=0.3)
                out = out_dir / "Q2" / "q2_train_vs_eval.png"
                ensure(out.parent); plt.savefig(out, bbox_inches='tight', dpi=160); plt.close()
                print(f"[Q2] Wrote {out}")
            else:
                print(f"[Q2] Couldn’t find average-return tags. Available: {list(sc.keys())}")
        else:
            print("[Q2] No event file found")
    else:
        print("[Q2] Missing run")

    # -------------- Q3: learning curves per env --------------
    q3_keys = [
        ("q3_obstacles", "Obstacles"),
        ("q3_reacher",   "Reacher"),
        ("q3_cheetah",   "Cheetah"),
    ]
    for key, label in q3_keys:
        run = pick_latest_run(data_dir, key)
        if not run:
            print(f"[Q3] Missing run for {key}")
            continue
        ev = find_event_file(run)
        if not ev:
            print(f"[Q3] No events for {key}")
            continue
        sc = load_scalars(ev)
        eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
        train_tag = get_best_tag(sc, CANDIDATE_TRAIN_TAGS)
        if not eval_tag:
            print(f"[Q3] No eval tag for {key}. Have: {list(sc.keys())}")
            continue
        steps_e, vals_e = sc[eval_tag]
        series = [vals_e]
        labels = ["Eval"]
        if train_tag:
            steps_t, vals_t = sc[train_tag]
            # align by step union; we’ll just plot as-is with step on x
            # to keep simple, plot vs index
            series.append(vals_t); labels.append("Train")
            x = np.arange(max(len(vals_e), len(vals_t)))
        else:
            x = np.arange(len(vals_e))
        out = out_dir / "Q3" / f"{key}_returns.png"
        plot_lines(x, series, labels, f"Q3 {label}: AverageReturn vs Iter", out)
        print(f"[Q3] Wrote {out}")

    # -------------- Q4: comparisons on reacher --------------
    # Horizon: 5,15,30
    horizon_keys = [("q4_reacher_horizon5","H=5"),
                    ("q4_reacher_horizon15","H=15"),
                    ("q4_reacher_horizon30","H=30")]
    labels, finals = [], []
    for key, lab in horizon_keys:
        run = pick_latest_run(data_dir, key)
        if not run: continue
        ev = find_event_file(run)
        if not ev: continue
        sc = load_scalars(ev)
        eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
        if not eval_tag: continue
        finals.append(sc[eval_tag][1][-1]); labels.append(lab)
    if finals:
        out = out_dir / "Q4" / "reacher_horizon_comparison.png"
        bar_plot(labels, finals, "Q4 Reacher: Effect of Planning Horizon", out)

    # Num sequences: 100, 1000
    numseq_keys = [("q4_reacher_numseq100","N=100"),
                   ("q4_reacher_numseq1000","N=1000")]
    labels, finals = [], []
    for key, lab in numseq_keys:
        run = pick_latest_run(data_dir, key)
        if not run: continue
        ev = find_event_file(run)
        if not ev: continue
        sc = load_scalars(ev)
        eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
        if not eval_tag: continue
        finals.append(sc[eval_tag][1][-1]); labels.append(lab)
    if finals:
        out = out_dir / "Q4" / "reacher_numseq_comparison.png"
        bar_plot(labels, finals, "Q4 Reacher: Effect of # Candidate Sequences", out)

    # Ensemble: 1,3,5
    ens_keys = [("q4_reacher_ensemble1","E=1"),
                ("q4_reacher_ensemble3","E=3"),
                ("q4_reacher_ensemble5","E=5")]
    labels, finals = [], []
    for key, lab in ens_keys:
        run = pick_latest_run(data_dir, key)
        if not run: continue
        ev = find_event_file(run); 
        if not ev: continue
        sc = load_scalars(ev)
        eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
        if not eval_tag: continue
        finals.append(sc[eval_tag][1][-1]); labels.append(lab)
    if finals:
        out = out_dir / "Q4" / "reacher_ensemble_comparison.png"
        bar_plot(labels, finals, "Q4 Reacher: Effect of Ensemble Size", out)

    # -------------- Q5: random vs CEM 2 vs CEM 4 on cheetah --------------
    q5_keys = [("q5_cheetah_random", "Random-shooting"),
               ("q5_cheetah_cem_2",  "CEM (2 iters)"),
               ("q5_cheetah_cem_4",  "CEM (4 iters)")]
    series, labels = [], []
    xlen = 0
    for key, lab in q5_keys:
        run = pick_latest_run(data_dir, key)
        if not run:
            print(f"[Q5] Missing run for {key}")
            continue
        ev = find_event_file(run)
        if not ev:
            print(f"[Q5] No events for {key}")
            continue
        sc = load_scalars(ev)
        eval_tag = get_best_tag(sc, CANDIDATE_EVAL_TAGS)
        if not eval_tag:
            print(f"[Q5] No eval tag for {key}. Available: {list(sc.keys())}")
            continue
        vals = sc[eval_tag][1]
        series.append(vals); labels.append(lab)
        xlen = max(xlen, len(vals))
    if series:
        x = np.arange(xlen)
        out = out_dir / "Q5" / "cheetah_random_vs_cem.png"
        plot_lines(x, series, labels, "Q5 Cheetah: Random vs CEM", out)

    print(f"\nDone. Figures saved under: {out_dir}")

if __name__ == "__main__":
    main()
