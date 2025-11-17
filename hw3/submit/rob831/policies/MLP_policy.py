import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self.forward(observation)
        action = action_distribution.sample()
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        """
        Policy gradient step: maximize E[ log pi(a|s) * advantage ].
        Returns scalar loss.item().
        """
        import torch
        from torch.distributions import Categorical, Normal

        # to torch
        obs_t = torch.as_tensor(observations, dtype=torch.float32)
        act_t = torch.as_tensor(actions, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_n if adv_n is not None else 1.0, dtype=torch.float32)

        # if a single scalar is given, broadcast; otherwise flatten to [B]
        if adv_t.ndim == 0:
            adv_t = torch.ones(obs_t.shape[0], dtype=torch.float32) * adv_t
        else:
            adv_t = adv_t.reshape(-1)

        # build log-prob of taken actions under current policy
        if self.discrete:
            # parent MLPPolicy typically exposes a logits network for discrete action spaces
            logits = self.logits_na(obs_t) if hasattr(self, "logits_na") else self.network(obs_t)
            dist = Categorical(logits=logits)
            act_long = act_t.long().reshape(-1)
            logp = dist.log_prob(act_long)                      # [B]
        else:
            # diagonal Gaussian: mean from network, state-independent logstd parameter
            mean = self.mean_net(obs_t) if hasattr(self, "mean_net") else self.network(obs_t)
            std = torch.exp(self.logstd) if hasattr(self, "logstd") else torch.ones_like(mean)
            dist = Normal(mean, std)
            logp = dist.log_prob(act_t).sum(dim=-1)            # [B]

        # policy gradient loss (minus sign because we maximize)
        loss = -(logp * adv_t.detach()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

