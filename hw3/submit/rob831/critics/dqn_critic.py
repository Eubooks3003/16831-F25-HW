from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from rob831.infrastructure import pytorch_util as ptu


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Robust update: fix shapes, align batch sizes, slice features to self.ob_dim.
        """
        import numpy as np

        need = self.ob_dim

        # --- ensure numpy arrays ---
        ob_no = np.asarray(ob_no)
        next_ob_no = np.asarray(next_ob_no)
        ac_na = np.asarray(ac_na)
        reward_n = np.asarray(reward_n)
        terminal_n = np.asarray(terminal_n)

        # --- ensure obs are 2D [B, need] ---
        if ob_no.ndim == 1:
            ob_no = ob_no[None, ...]
        if next_ob_no.ndim == 1:
            next_ob_no = next_ob_no[None, ...]

        # slice feature dim to what the critic expects
        if ob_no.shape[-1] != need:
            ob_no = ob_no[..., -need:]
        if next_ob_no.shape[-1] != need:
            next_ob_no = next_ob_no[..., -need:]

        # --- ensure others are 1D [B] ---
        ac_na = ac_na.reshape(-1)
        reward_n = reward_n.reshape(-1)
        terminal_n = terminal_n.reshape(-1)

        # --- align batch sizes (take common B) ---
        B = min(ob_no.shape[0], next_ob_no.shape[0],
                ac_na.shape[0], reward_n.shape[0], terminal_n.shape[0])

        ob_no = ob_no[:B]
        next_ob_no = next_ob_no[:B]
        ac_na = ac_na[:B]
        reward_n = reward_n[:B]
        terminal_n = terminal_n[:B]

        # --- torch tensors ---
        ob_no = ptu.from_numpy(ob_no)                  # [B, need]
        ac_na = ptu.from_numpy(ac_na).to(torch.long)   # [B]
        next_ob_no = ptu.from_numpy(next_ob_no)        # [B, need]
        reward_n = ptu.from_numpy(reward_n)            # [B]
        terminal_n = ptu.from_numpy(terminal_n)        # [B]

        # Q(s, a)
        qa_t_values = self.q_net(ob_no)                                # [B, A]
        q_t_values = qa_t_values.gather(1, ac_na.unsqueeze(1)).squeeze(1)  # [B]

        # Q_target(s', ·)
        qa_tp1_values = self.q_net_target(next_ob_no)                  # [B, A]
        if self.double_q:
            qa_tp1_online = self.q_net(next_ob_no)                     # [B, A]
            a_tp1 = torch.argmax(qa_tp1_online, dim=1, keepdim=True)   # [B,1]
            q_tp1 = qa_tp1_values.gather(1, a_tp1).squeeze(1)          # [B]
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)                        # [B]

        # Bellman target
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape, f"{q_t_values.shape} vs {target.shape}"
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        return {'Training Loss': ptu.to_numpy(loss)}



    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
