import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # NEW: multiple gradient steps / minibatches
        self.num_gradient_steps = int(self.agent_params.get('num_gradient_steps', 1))
        self.update_batch_size = self.agent_params.get('update_batch_size', None)  # None => full batch

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):
        """
        Update the PG actor/policy using the given batch and return the train_log.
        Now supports multiple gradient steps per collection with optional mini-batches.
        """
        # 1) Monte-Carlo Q-values (aligned with obs/actions order)
        q_values = self.calculate_q_vals(rewards_list)

        # 2) Advantages (handles baseline and optional GAE)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_values, terminals
        )

        # 3) Multiple updates on the SAME data
        N = observations.shape[0]
        mb = self.update_batch_size or N  # full batch by default
        mb = int(min(max(1, mb), N))

        # pre-assemble indices once; reshuffle each epoch
        last_log = None
        for k in range(self.num_gradient_steps):
            perm = np.random.permutation(N) if mb < N else np.arange(N)
            # iterate mini-batches
            for start in range(0, N, mb):
                idx = perm[start:start+mb]
                last_log = self.actor.update(
                    observations=observations[idx],
                    actions=actions[idx],
                    advantages=advantages[idx],
                    q_values=q_values[idx] if self.nn_baseline else None,
                )

        # augment returned log with info about how many updates/batch size
        if last_log is None:
            last_log = {}
        last_log['NumPolicyUpdates'] = int(self.num_gradient_steps)
        last_log['UpdateBatchSize'] = int(mb)
        last_log['BatchN'] = int(N)
        return last_log

    # -------- rest of your class stays the same --------

    def calculate_q_vals(self, rewards_list):
        if not self.reward_to_go:
            q_per_traj = [self._discounted_return(rews) for rews in rewards_list]
        else:
            q_per_traj = [self._discounted_cumsum(rews) for rews in rewards_list]
        q_values = np.concatenate(q_per_traj, axis=0).astype(np.float32)
        return q_values

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):
        if self.nn_baseline:
            v_norm = self.actor.run_baseline_prediction(obs)  # shape [N]
            q_mean, q_std = np.mean(q_values), np.std(q_values) + 1e-8
            values = unnormalize(v_norm, q_mean, q_std)

            if self.gae_lambda is not None:
                rewards = np.concatenate(rewards_list).astype(np.float32)
                batch_size = obs.shape[0]
                values = np.append(values, 0.0).astype(np.float32)
                terminals = terminals.astype(np.float32)

                advantages = np.zeros(batch_size + 1, dtype=np.float32)
                for t in range(batch_size - 1, -1, -1):
                    not_done = 1.0 - terminals[t]
                    delta = rewards[t] + self.gamma * values[t + 1] * not_done - values[t]
                    advantages[t] = delta + self.gamma * self.gae_lambda * not_done * advantages[t + 1]
                advantages = advantages[:-1]
            else:
                advantages = q_values - values
        else:
            advantages = q_values.copy()

        if self.standardize_advantages:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = normalize(advantages, adv_mean, adv_std)

        return advantages

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    def _discounted_return(self, rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        T = rewards.shape[0]
        discounts = self.gamma ** np.arange(T, dtype=np.float32)
        total_return = np.sum(discounts * rewards, dtype=np.float32)
        discounted_returns = np.full(T, total_return, dtype=np.float32)
        return discounted_returns

    def _discounted_cumsum(self, rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        T = rewards.shape[0]
        discounted_cumsums = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + self.gamma * running
            discounted_cumsums[t] = running
        return discounted_cumsums
