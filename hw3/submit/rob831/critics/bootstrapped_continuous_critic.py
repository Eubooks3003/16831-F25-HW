from .base_critic import BaseCritic
from torch import nn
from torch import optim
import torch

from rob831.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):

        # to torch (on correct device)
        obs_t      = ptu.from_numpy(ob_no)
        next_obs_t = ptu.from_numpy(next_ob_no)
        rew_t      = ptu.from_numpy(reward_n)
        term_t     = ptu.from_numpy(terminal_n)

        steps_total = self.num_target_updates * self.num_grad_steps_per_target_update
        last_loss = None

        for t in range(steps_total):
            # recompute targets at the start of each target-update block
            if t % self.num_grad_steps_per_target_update == 0:
                with torch.no_grad():
                    v_next = self.forward(next_obs_t)              # [B]
                    v_next = v_next * (1 - term_t)                 # cut off at terminal
                    targets = rew_t + self.gamma * v_next          # [B]

            # critic prediction and MSE loss
            v_pred = self.forward(obs_t)                           # [B]
            loss = self.loss(v_pred, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            last_loss = loss.item()

        return last_loss

