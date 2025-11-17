from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # 1) update critic multiple times
        critic_log = None
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # 2) compute advantages
        adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # 3) update actor multiple times
        actor_loss = None
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, adv_n)

        loss = OrderedDict()
        # be permissive about critic log key name
        loss['Loss_Critic'] = (critic_log.get('Training Loss Critic')
                            if isinstance(critic_log, dict) and 'Training Loss Critic' in critic_log
                            else critic_log if isinstance(critic_log, (int, float))
                            else 0.0)
        loss['Loss_Actor'] = actor_loss if actor_loss is not None else 0.0
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
        A(s,a) = r + gamma * V(s') * (1 - done) - V(s)
        """
        import numpy as np

        # helper to query critic as numpy
        if hasattr(self.critic, "forward_np"):
            V_s  = self.critic.forward_np(ob_no)          # [B]
            V_sp = self.critic.forward_np(next_ob_no)     # [B]
        elif hasattr(self.critic, "predict"):
            V_s  = self.critic.predict(ob_no)
            V_sp = self.critic.predict(next_ob_no)
        else:
            # generic fallback
            import torch
            with torch.no_grad():
                V_s  = self.critic(torch.as_tensor(ob_no, dtype=torch.float32)).cpu().numpy().squeeze()
                V_sp = self.critic(torch.as_tensor(next_ob_no, dtype=torch.float32)).cpu().numpy().squeeze()

        re_n = np.asarray(re_n).reshape(-1)
        terminal_n = np.asarray(terminal_n).reshape(-1)
        V_s = np.asarray(V_s).reshape(-1)
        V_sp = np.asarray(V_sp).reshape(-1)

        Q_sa = re_n + self.gamma * V_sp * (1 - terminal_n)
        adv_n = Q_sa - V_s

        if self.standardize_advantages:
            mu, std = adv_n.mean(), adv_n.std() + 1e-8
            adv_n = (adv_n - mu) / std
        return adv_n


    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
