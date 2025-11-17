#!/usr/bin/env python3
import argparse, glob, os, time, math
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ---------- TensorBoard reading ----------
def read_event_file(event_path: str, max_points: int = None) -> Tuple[List[float], List[float]]:
    """Return (steps, returns) from a single events file."""
    X, Y = [], []
    for e in tf.train.summary_iterator(event_path):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if max_points is not None and len(X) >= max_points:
            break
    return X, Y

def get_newest_events_file(run_dir: str) -> str:
    """Pick newest events* file in a run directory."""
    paths = glob.glob(os.path.join(run_dir, "events*"))
    if not paths:
        raise FileNotFoundError(f"No TensorBoard event files in {run_dir}")
    return max(paths, key=os.path.getmtime)

# ---------- Run selection (pick most recent matching folder) ----------
def find_latest_run_dir(data_dir: str, exp_prefix: str) -> str:
    """
    Among directories under data_dir that start with exp_prefix + '_',
    return the one with the most recent mtime.
    """
    candidates = []
    prefix = exp_prefix + "_"
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full) and name.startswith(prefix):
            candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f"No run dirs found for prefix '{exp_prefix}' in {data_dir}")
    return max(candidates, key=os.path.getmtime)

# ---------- Curve utilities ----------
def truncate_to_min_length(curves: List[Tuple[List[float], List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Given list of (X, Y) curves, truncate all to the shortest length and return stacked arrays."""
    min_len = min(len(x) for x, _ in curves)
    Xs = np.array([np.array(x[:min_len]) for x, _ in curves], dtype=np.float64)
    Ys = np.array([np.array(y[:min_len]) for _, y in curves], dtype=np.float64)
    return Xs, Ys

def align_xy(X: list, Y: list):
    """Trim a single (X,Y) curve to the same length using the per-run min."""
    m = min(len(X), len(Y))
    return np.asarray(X[:m], dtype=np.float64), np.asarray(Y[:m], dtype=np.float64)

def mean_std_across_seeds(curves):
    """
    curves: list of (X, Y) lists from multiple seeds.
    1) Pairwise align each (X,Y) by its own min length.
    2) Then truncate all runs to the global min across seeds.
    3) Return (X_mean, Y_mean, Y_std).
    """
    aligned = [align_xy(X, Y) for (X, Y) in curves]
    # global min length across seeds AFTER pairwise alignment
    gmin = min(x.shape[0] for (x, y) in aligned)
    Xs = np.stack([x[:gmin] for (x, _) in aligned], axis=0)  # [S, T]
    Ys = np.stack([y[:gmin] for (_, y) in aligned], axis=0)  # [S, T]

    X_mean = Xs.mean(axis=0)
    Y_mean = Ys.mean(axis=0)
    Y_std  = Ys.std(axis=0)
    return X_mean, Y_mean, Y_std


def plot_single(ax, X, Y, label):
    ax.plot(X, Y, label=label)

def plot_mean_std(ax, X, mean, std, label):
    ax.plot(X, mean, label=label)
    ax.fill_between(X, mean - std, mean + std, alpha=0.2)

# ---------- High-level tasks ----------
def load_curve_for_prefix(data_dir: str, exp_prefix: str, max_points: int = None) -> Tuple[List[float], List[float], str]:
    run_dir = find_latest_run_dir(data_dir, exp_prefix)
    ev = get_newest_events_file(run_dir)
    X, Y = read_event_file(ev, max_points=max_points)
    return X, Y, run_dir

def q1_make_plot(data_dir: str, out_path: str):
    """DQN vs DoubleDQN (3 seeds each), averaged with std error bands."""
    dqn_prefixes   = ["q1_dqn_1", "q1_dqn_2", "q1_dqn_3"]
    ddqn_prefixes  = ["q1_doubledqn_1", "q1_doubledqn_2", "q1_doubledqn_3"]

    dqn_curves = []
    for p in dqn_prefixes:
        X, Y, rd = load_curve_for_prefix(data_dir, p)
        dqn_curves.append((X, Y))
        print(f"[Q1] DQN seed '{p}' from {rd}: final return {Y[-1]:.1f}")

    ddqn_curves = []
    for p in ddqn_prefixes:
        X, Y, rd = load_curve_for_prefix(data_dir, p)
        ddqn_curves.append((X, Y))
        print(f"[Q1] DDQN seed '{p}' from {rd}: final return {Y[-1]:.1f}")

    X_dqn, Y_dqn, S_dqn   = mean_std_across_seeds(dqn_curves)
    X_ddqn, Y_ddqn, S_ddqn = mean_std_across_seeds(ddqn_curves)

    # Print summary
    print(f"[Q1] DQN final mean±std: {Y_dqn[-1]:.1f} ± {S_dqn[-1]:.1f}")
    print(f"[Q1] DDQN final mean±std: {Y_ddqn[-1]:.1f} ± {S_ddqn[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_mean_std(ax, X_dqn,  Y_dqn,  S_dqn,  "DQN (mean ± std)")
    plot_mean_std(ax, X_ddqn, Y_ddqn, S_ddqn, "Double DQN (mean ± std)")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Average Return per eval")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Q1] Saved: {out_path}")

def q2_make_plot(data_dir: str, out_path: str, exp_prefix: str = "q2_10_10"):
    X, Y, rd = load_curve_for_prefix(data_dir, exp_prefix)
    print(f"[Q2] Using run dir: {rd}. Final return: {Y[-1]:.1f}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_single(ax, X, Y, "CartPole-v0")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Average Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Q2] Saved: {out_path}")

def q3_make_plot(data_dir: str, out_path: str, exp_prefix: str = "q3_10_10"):
    X, Y, rd = load_curve_for_prefix(data_dir, exp_prefix)
    print(f"[Q3] Using run dir: {rd}. Final return: {Y[-1]:.1f}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_single(ax, X, Y, "InvertedPendulum-v4")
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Average Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Q3] Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="rob831/data",
                        help="Directory containing run subfolders (q1_..., q2_..., q3_...)")
    parser.add_argument("--out_dir", type=str, default="rob831/data/figs",
                        help="Where to save the figures")
    parser.add_argument("--skip_q1", action="store_true")
    parser.add_argument("--skip_q2", action="store_true")
    parser.add_argument("--skip_q3", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_q1:
        q1_make_plot(args.data_dir, os.path.join(args.out_dir, "q1_dqn_vs_ddqn.png"))
    if not args.skip_q2:
        q2_make_plot(args.data_dir, os.path.join(args.out_dir, "q2_cartpole.png"))
    if not args.skip_q3:
        q3_make_plot(args.data_dir, os.path.join(args.out_dir, "q3_invertedpendulum.png"))

if __name__ == "__main__":
    main()
