import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def get_section_results(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    steps_events = ea.Scalars('Train_EnvstepsSoFar')
    return_events = ea.Scalars('Eval_AverageReturn')   # aka eval_avg_return

    X = [e.value for e in steps_events]
    Y = [e.value for e in return_events]
    return X, Y


def get_latest_event_file(run_glob_pattern):
    run_dirs = glob.glob(run_glob_pattern)
    if not run_dirs:
        raise ValueError(f'No run dirs found for pattern: {run_glob_pattern}')

    run_dirs.sort()          # timestamp suffix => last is newest
    latest_run = run_dirs[-1]

    event_files = glob.glob(os.path.join(latest_run, 'events*'))
    if not event_files:
        raise ValueError(f'No event files in run dir: {latest_run}')

    return event_files[0]


def main():
    # directory where this script lives
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # rob831/data relative to this script
    base_dir = os.path.normpath(os.path.join(this_dir, '..', '..', 'data'))

    # figs/Q6 relative to this script: hw4/figs/Q6
    figs_dir = os.path.normpath(os.path.join(this_dir, '..', '..', '..', 'figs', 'Q6'))
    os.makedirs(figs_dir, exist_ok=True)

    exp_patterns = {
        # env1 = PointmassEasy-v0
        'env1_rnd': os.path.join(
            base_dir, 'hw4_part2_expl_q6_env1_rnd_PointmassEasy-v0_*'
        ),
        'env1_random': os.path.join(
            base_dir, 'hw4_part2_expl_q6_env1_random_PointmassEasy-v0_*'
        ),

        # env2 = PointmassHard-v0
        'env2_rnd': os.path.join(
            base_dir, 'hw4_part2_expl_q6_env2_rnd_PointmassHard-v0_*'
        ),
        'env2_random': os.path.join(
            base_dir, 'hw4_part2_expl_q6_env2_random_PointmassHard-v0_*'
        ),
    }

    results = {}
    for name, pattern in exp_patterns.items():
        try:
            eventfile = get_latest_event_file(pattern)
        except ValueError as e:
            print(f'Skipping {name}: {e}')
            continue

        X, Y = get_section_results(eventfile)
        results[name] = (X, Y)

        print(f'\n==== Q6 {name} ====')
        for i, (x, y) in enumerate(zip(X, Y)):
            print(f'Iteration {i:2d} | Train steps: {int(x):7d} | Eval return: {y: .3f}')
        if Y:
            print(f'Final eval return ({name}): {Y[-1]:.3f}')

    # ---------- make learning-curve figs for Q6 ----------
    if 'env1_rnd' in results and 'env1_random' in results:
        plt.figure()
        for key, label in [('env1_rnd', 'RND'), ('env1_random', 'Random ε-greedy')]:
            X, Y = results[key]
            plt.plot(X, Y, label=label)
        plt.xlabel('Train Env Steps')
        plt.ylabel('Eval Average Return')
        plt.title('Q6 – PointmassEasy-v0')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(figs_dir, 'q6_env1_eval_return.png')
        plt.savefig(out_path, dpi=150)
        print('Saved', out_path)

    if 'env2_rnd' in results and 'env2_random' in results:
        plt.figure()
        for key, label in [('env2_rnd', 'RND'), ('env2_random', 'Random ε-greedy')]:
            X, Y = results[key]
            plt.plot(X, Y, label=label)
        plt.xlabel('Train Env Steps')
        plt.ylabel('Eval Average Return')
        plt.title('Q6 – PointmassHard-v0')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(figs_dir, 'q6_env2_eval_return.png')
        plt.savefig(out_path, dpi=150)
        print('Saved', out_path)


if __name__ == '__main__':
    main()
