# save as: p4_ant_steps_sweep.py
import argparse, subprocess, re, os, csv, sys, textwrap
from datetime import datetime
import matplotlib.pyplot as plt

def run_once(steps, args):
    exp_name = f"p4_ant_steps_{steps}"
    cmd = [
        sys.executable, "rob831/scripts/run_hw1.py",
        "--expert_policy_file", "rob831/policies/experts/Ant.pkl",
        "--expert_data",       "rob831/expert_data/expert_data_Ant-v2.pkl",
        "--env_name",          "Ant-v2",
        "--exp_name",          exp_name,
        "--n_iter",            "1",
        "--learning_rate",     str(args.lr),
        "--n_layers",          str(args.n_layers),
        "--size",              str(args.size),
        "--train_batch_size",  str(args.train_batch_size),
        "--num_agent_train_steps_per_iter", str(steps),
        "--eval_batch_size",   str(args.eval_batch_size),
        "--video_log_freq",    "-1",
        "--scalar_log_freq",   "1",
        "--seed",              str(args.seed),
    ]
    if args.no_gpu:
        cmd.append("--no_gpu")

    # run and capture logs
    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"{exp_name}.log")
    print(f"\n[RUN] steps={steps}\n  -> logging to {log_path}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_path, "w") as f:
        f.write(proc.stdout)

    # parse numbers
    txt = proc.stdout
    m_mean = re.search(r"Eval_AverageReturn\s*:\s*([-\d\.]+)", txt)
    m_std  = re.search(r"Eval_StdReturn\s*:\s*([-\d\.]+)", txt)
    if not (m_mean and m_std):
        print("  ! Could not parse metrics from logs; check the run output.")
        return None

    mean = float(m_mean.group(1))
    std  = float(m_std.group(1))
    print(f"  Eval mean={mean:.2f}, std={std:.2f}")
    return {"steps": steps, "mean": mean, "std": std, "log": log_path}

def main():
    parser = argparse.ArgumentParser(
        description="Part 4: Ant-v2 BC sweep over training steps and plot mean±std returns."
    )
    parser.add_argument("--steps", type=str,
                        default="1000,5000,10000,20000,50000,100000",
                        help="Comma-separated list of training steps to sweep.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=10000,  # ~10 rollouts for Ant (ep_len=1000)
                        help="Total eval timesteps; ensure >= 5000 for ≥5 rollouts.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--outdir", type=str, default="p4_outputs")
    parser.add_argument("--logdir", type=str, default="logs_p4_ant")
    args = parser.parse_args()

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for s in steps_list:
        r = run_once(s, args)
        if r:
            results.append(r)

    if not results:
        print("No results parsed; aborting plot.")
        return

    # sort by steps
    results.sort(key=lambda x: x["steps"])

    # write CSV
    csv_path = os.path.join(args.outdir, "p4_ant_steps_results.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["steps", "mean", "std", "log_path"])
        for r in results:
            w.writerow([r["steps"], r["mean"], r["std"], r["log"]])
    print(f"\nWrote CSV: {csv_path}")

    # plot
    xs   = [r["steps"] for r in results]
    ys   = [r["mean"]  for r in results]
    yerr = [r["std"]   for r in results]

    plt.figure(figsize=(7,4.5))
    plt.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=4)
    plt.xscale("log")
    plt.grid(True, alpha=0.4)
    plt.xlabel("Number of training steps (log scale)")
    plt.ylabel("BC Eval Return (mean ± std)")
    caption = textwrap.fill(
        "Ant-v2 Behavior Cloning: performance vs. number of gradient steps. "
        "Hyperparameters: MLP 2×128, lr=3e-4, train_batch=1024, n_iter=1, "
        f"eval_batch_size={args.eval_batch_size} (≥5 rollouts), seed={args.seed}.",
        width=70
    )
    plt.title("BC performance vs. training steps (Ant-v2)")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    plt.figtext(0.5, -0.15, caption + f"\nGenerated: {ts}", ha="center", va="top", fontsize=9)
    plt.tight_layout()

    png_path = os.path.join(args.outdir, "p4_ant_steps_curve.png")
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"Wrote figure: {png_path}")

if __name__ == "__main__":
    main()
