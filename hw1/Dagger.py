#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "Eval_AverageReturn"

def rglob_dirs(base: Path, pattern: str):
    # return run directories whose names match pattern anywhere under base
    return sorted([p for p in base.rglob("*") if p.is_dir() and p.match(pattern)])

def find_event_files(run_dir: Path):
    return sorted([p for p in run_dir.rglob("*") if "tfevents" in p.name])

def load_scalars_from_events(event_files, tag: str):
    """Read all event files; return dict step->list(values) (duplicates merged)."""
    vals = {}
    for ef in event_files:
        try:
            ea = EventAccumulator(str(ef), size_guidance={"scalars": 0})
            ea.Reload()
            if tag not in ea.Tags().get("scalars", []):
                continue
            for s in ea.Scalars(tag):
                vals.setdefault(s.step, []).append(s.value)
        except Exception:
            pass
    # Collapse multiples per step within a run by averaging
    step_to_val = {k: float(np.mean(vs)) for k, vs in vals.items()}
    return step_to_val

def load_curve_from_run(run_dir: Path, tag: str):
    files = find_event_files(run_dir)
    if not files:
        return [], []
    step_to_val = load_scalars_from_events(files, tag)
    if not step_to_val:
        return [], []
    steps = sorted(step_to_val.keys())
    vals = [step_to_val[s] for s in steps]
    return steps, vals

def aggregate_runs(run_dirs, tag: str):
    """Align by step index (union); for each step, average across runs that have it."""
    if not run_dirs:
        return [], [], []
    per = []
    all_steps = set()
    for rd in run_dirs:
        s, v = load_curve_from_run(rd, tag)
        if s:
            per.append((s, v))
            all_steps.update(s)
    if not per:
        return [], [], []
    steps_sorted = sorted(all_steps)
    means, stds = [], []
    for s in steps_sorted:
        vals_here = [v[s_list.index(s)] for (s_list, v) in per if s in s_list]
        means.append(float(np.mean(vals_here)))
        stds.append(float(np.std(vals_here)))
    return steps_sorted, means, stds

def expert_mean(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    returns = [float(np.sum(path["reward"])) for path in data]
    return float(np.mean(returns))

def bc_baseline(data_dir: Path, glob_pat: str):
    runs = rglob_dirs(data_dir, glob_pat)
    finals = []
    for rd in runs:
        _, vals = load_curve_from_run(rd, TAG)
        if vals:
            finals.append(vals[-1])
    if not finals:
        return None, None
    return float(np.mean(finals)), float(np.std(finals))

def plot_env(ax, *, title, data_dir, dagger_glob, expert_pkl, bc_glob):
    dagger_runs = rglob_dirs(data_dir, dagger_glob)
    steps, mean, std = aggregate_runs(dagger_runs, TAG)
    if steps:
        ax.plot(steps, mean, marker="o", lw=2, label="DAgger (mean)")
        ax.fill_between(steps, np.array(mean)-np.array(std), np.array(mean)+np.array(std),
                        alpha=0.2, label="DAgger (std)")
    else:
        ax.text(0.5, 0.5, f"No DAgger runs matched:\n{dagger_glob}",
                ha="center", va="center", transform=ax.transAxes)

    # Expert baseline
    try:
        e_mean = expert_mean(expert_pkl)
        ax.axhline(e_mean, ls="--", lw=2, color="tab:green", label="Expert")
    except Exception as e:
        ax.text(0.01, 0.97, f"Expert not loaded: {e}", transform=ax.transAxes,
                va="top", fontsize=8)

    # BC baseline (optional)
    bc_mean, bc_std = bc_baseline(data_dir, bc_glob)
    if bc_mean is not None:
        ax.axhline(bc_mean, ls=":", lw=2, color="tab:red",
                   label=f"BC (mean ± {bc_std:.1f})")

    ax.set_title(title)
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True,
                    help="Folder containing your run subdirs (e.g., hw1/data).")
    ap.add_argument("--expert_ant", type=Path, required=True,
                    help="Path to expert_data_Ant-v2.pkl")
    ap.add_argument("--expert_hum", type=Path, required=True,
                    help="Path to expert_data_Humanoid-v2.pkl")
    ap.add_argument("--ant_glob", default="q2_dagger_ant_*",
                    help="Glob (directory name) for Ant DAgger runs")
    ap.add_argument("--hum_glob", default="q2_dagger_humanoid_*",
                    help="Glob for Humanoid DAgger runs")
    ap.add_argument("--bc_ant_glob", default="*bc*Ant-v2*",
                    help="Glob for Ant BC runs (optional)")
    ap.add_argument("--bc_hum_glob", default="*bc*Humanoid-v2*",
                    help="Glob for Humanoid BC runs (optional)")
    ap.add_argument("--out", type=Path, default=Path("q2_dagger_curves.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    plot_env(axes[0],
             title="Ant-v2",
             data_dir=args.data_dir,
             dagger_glob=args.ant_glob,
             expert_pkl=args.expert_ant,
             bc_glob=args.bc_ant_glob)
    plot_env(axes[1],
             title="Humanoid-v2",
             data_dir=args.data_dir,
             dagger_glob=args.hum_glob,
             expert_pkl=args.expert_hum,
             bc_glob=args.bc_hum_glob)

    fig.suptitle("DAgger learning curves (mean ± std across seeds)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
