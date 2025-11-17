import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            
            random_action_sequences = np.random.uniform(
                low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim)
            )# TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # Cross-Entropy Method (iterative random-shooting with refinement)
            num_seq, H = num_sequences, horizon

            # Per-timestep action bounds
            low  = self.low
            high = self.high

            mean = None                   # (H, ac_dim)
            std  = None                   # (H, ac_dim)

            for it in range(self.cem_iterations):
                if it == 0:
                    # First iteration: uniform random in [low, high]
                    candidate_action_sequences = np.random.uniform(
                        low=low, high=high, size=(num_seq, H, self.ac_dim)
                    )
                else:
                    # Later iterations: sample from N(mean, std) and clip to bounds
                    samples = np.random.normal(
                        loc=mean[None, ...], scale=std[None, ...], size=(num_seq, H, self.ac_dim)
                    )
                    candidate_action_sequences = np.clip(samples, low, high)

                # Evaluate sequences using the current ensemble
                returns = self.evaluate_candidate_sequences(candidate_action_sequences, obs)  # (num_seq,)

                # Take top-K elites
                elite_idx = np.argsort(returns)[-self.cem_num_elites:]
                elites = candidate_action_sequences[elite_idx]  # (K, H, ac_dim)

                # Compute elite stats per time step
                elite_mean = elites.mean(axis=0)               # (H, ac_dim)
                elite_std  = elites.std(axis=0) + 1e-6         # (H, ac_dim)

                if it == 0:
                    mean = elite_mean
                    std  = elite_std
                else:
                    # Exponential moving average with alpha
                    mean = self.cem_alpha * elite_mean + (1.0 - self.cem_alpha) * mean
                    std  = self.cem_alpha * elite_std  + (1.0 - self.cem_alpha) * std

            # After final iteration, execute the mean action sequence
            cem_action = mean  # shape (H, ac_dim)
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # candidate_action_sequences: [N, H, act_dim]
        # obs: current observation, [obs_dim]
        per_model_returns = []
        for model in self.dyn_models:
            # sum of rewards for each sequence under this dynamics model → shape (N,)
            returns = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            per_model_returns.append(returns)

        # mean across ensemble → shape (N,)
        return np.mean(np.stack(per_model_returns, axis=0), axis=0)


    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            best_idx = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[best_idx]      # [H, act_dim]
            action_to_take = best_action_sequence[0]                         # [act_dim]
            return action_to_take[None]  # [1, act_dim]


    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """
        obs: [D_obs], candidate_action_sequences: [N, H, D_action]
        return: [N] sum of rewards for each sequence under `model`
        """
        N, H, _ = candidate_action_sequences.shape
        # tile the starting obs for all candidates
        obs_batch = np.repeat(obs[None, :], N, axis=0)  # [N, D_obs]
        sum_of_rewards = np.zeros(N, dtype=np.float32)

        for t in range(H):
            acs_t = candidate_action_sequences[:, t, :]  # [N, D_action]

            # reward at this step (vectorized over N)
            rew_t = self.env.get_reward(obs_batch, acs_t)
            # envs sometimes return (rewards, info); normalize shapes
            if isinstance(rew_t, tuple):
                rew_t = rew_t[0]
            rew_t = np.squeeze(rew_t)  # [N]
            sum_of_rewards += rew_t

            # roll the dynamics one step with the learned model
            next_obs_batch = model.get_prediction(obs_batch, acs_t, self.data_statistics)  # [N, D_obs]
            obs_batch = next_obs_batch

        return sum_of_rewards

