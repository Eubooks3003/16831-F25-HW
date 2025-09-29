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
        """
        # 1) Monte-Carlo Q-values (aligned with obs/actions order)
        q_values = self.calculate_q_vals(rewards_list)

        # 2) Advantages (handles baseline and optional GAE)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_values, terminals
        )

        # 3) Policy (and baseline) update
        train_log = self.actor.update(
            observations=observations,
            actions=actions,
            advantages=advantages,
            q_values=q_values,  # used only if nn_baseline=True
        )
        return train_log



    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        if not self.reward_to_go:
            #use the whole traj for each timestep
            q_per_traj = [self._discounted_return(rews) for rews in rewards_list]

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_per_traj = [self._discounted_cumsum(rews) for rews in rewards_list]
        q_values = np.concatenate(q_per_traj, axis=0).astype(np.float32)

        return q_values  # return an array

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):
        """
        Computes advantages by (possibly) using GAE, or subtracting a baseline.
        """
        if self.nn_baseline:
            # Baseline net predicts *normalized* returns; unnormalize to Q scale
            v_norm = self.actor.run_baseline_prediction(obs)  # shape [N]
            q_mean, q_std = np.mean(q_values), np.std(q_values) + 1e-8
            values = unnormalize(v_norm, q_mean, q_std)       # back to Q scale

            if self.gae_lambda is not None:
                # Flatten rewards and prepare arrays
                rewards = np.concatenate(rewards_list).astype(np.float32)
                batch_size = obs.shape[0]

                # Add a dummy V_{T} for easier indexing; will be masked by terminals
                values = np.append(values, 0.0).astype(np.float32)
                terminals = terminals.astype(np.float32)

                advantages = np.zeros(batch_size + 1, dtype=np.float32)
                # GAE(λ): δ_t = r_t + γ V_{t+1}(1 − done_t) − V_t
                # A_t = δ_t + γλ(1 − done_t) A_{t+1}
                for t in range(batch_size - 1, -1, -1):
                    not_done = 1.0 - terminals[t]
                    delta = rewards[t] + self.gamma * values[t + 1] * not_done - values[t]
                    advantages[t] = delta + self.gamma * self.gae_lambda * not_done * advantages[t + 1]
                advantages = advantages[:-1]
            else:
                # Plain baseline: A = Q − V
                advantages = q_values - values
        else:
            advantages = q_values.copy()

        if self.standardize_advantages:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = normalize(advantages, adv_mean, adv_std)

        return advantages


    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
        Input: list of rewards r_0, …, r_T-1
        Output: length-T array where every entry equals sum_{t'=0}^{T-1} γ^{t'} r_{t'}
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        T = rewards.shape[0]
        discounts = self.gamma ** np.arange(T, dtype=np.float32)
        total_return = np.sum(discounts * rewards, dtype=np.float32)
        discounted_returns = np.full(T, total_return, dtype=np.float32)
        return discounted_returns

    def _discounted_cumsum(self, rewards):
        """
        Input: list of rewards r_0, …, r_T-1
        Output: length-T array with entry at t equal to sum_{t'=t}^{T-1} γ^{t'-t} r_{t'}
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        T = rewards.shape[0]
        discounted_cumsums = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + self.gamma * running
            discounted_cumsums[t] = running
        return discounted_cumsums