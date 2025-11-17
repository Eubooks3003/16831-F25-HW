from rob831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_((m.weight))
        nn.init.uniform_(m.bias)

def init_method_2(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)
        nn.init.normal_(m.bias)

class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim      = hparams['ob_dim']           # expects flattened size
        self.output_size = hparams['rnd_output_size']
        self.n_layers    = hparams['rnd_n_layers']
        self.size        = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # Target network f (frozen)
        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu',
            output_activation='identity',
            init_method=init_method_1,
        ).to(ptu.device)
        for p in self.f.parameters():
            p.requires_grad = False
        self.f.eval()

        # Predictor network f_hat (trainable)
        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            activation='relu',
            output_activation='identity',
            init_method=init_method_2,
        ).to(ptu.device)

        # Optimizer
        opt_kwargs = getattr(self.optimizer_spec, 'kwargs',
                     getattr(self.optimizer_spec, 'optim_kwargs', {}))
        self.optimizer = self.optimizer_spec.constructor(self.f_hat.parameters(), **opt_kwargs)

        self.criterion = nn.MSELoss(reduction='none')

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure shape [B, ob_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1:] != (self.ob_dim,):
            x = x.view(x.shape[0], -1)
        return x

    def forward(self, ob_no):
        ob_no = self._flatten(ob_no.to(ptu.device))
        with torch.no_grad():
            target = self.f(ob_no).detach()
        pred = self.f_hat(ob_no)
        err = ((pred - target) ** 2).mean(dim=1)  # [B]
        return err

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        loss = self(ob_no).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
