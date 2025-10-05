import numpy as np
import time

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # reset (support old/new API)
    reset_out = env.reset()
    ob = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                if hasattr(env, 'model') and hasattr(env.model, 'opt'):
                    time.sleep(env.model.opt.timestep)

        # policy action (KEEP FULL VECTOR, DO NOT ac[0])
        obs.append(ob)
        ac = policy.get_action(ob)  # expected shape (ac_dim,) or (1, ac_dim)
        if isinstance(ac, np.ndarray) and ac.ndim > 1 and ac.shape[0] == 1:
            ac = ac[0]
        ac = np.asarray(ac, dtype=np.float32)

        # clip to action space just in case
        if hasattr(env, "action_space") and hasattr(env.action_space, "low"):
            ac = np.clip(ac, env.action_space.low, env.action_space.high)

        acs.append(ac)

        # step (support old/new API)
        step_out = env.step(ac)
        if len(step_out) == 5:
            ob2, rew, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            ob2, rew, done, _info = step_out

        steps += 1
        next_obs.append(ob2)
        rewards.append(rew)

        rollout_done = bool(done or (steps >= max_path_length))
        terminals.append(1.0 if rollout_done else 0.0)

        ob = ob2
        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into the sampled_paths list.
    """
    sampled_paths = []

    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        sampled_paths.append(path)

    return sampled_paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

