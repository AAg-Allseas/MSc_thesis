"""
Main script for training a Latent SDE on the toy DP model. 

Most of the code is copied from latent_sde_lorentz.py - https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

Based on approach from:
Li, X., Wong, T.-K. L., Chen, R. T. Q., & Duvenaud, D. (2020). 
Scalable Gradients for Stochastic Differential Equations. 
Proceedings of the 23rd International Conference on Artificial Intelligence and Statistic, 108. 
https://doi.org/10.48550/arXiv.2001.01328

"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn, Tensor
from torch import optim
from torch.distributions import Normal

import torchsde
import logging
import os
from typing import Sequence


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val
    
class Encoder(nn.Module):
    """GRU encoder to encode observation data into context for the posterior network"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor) -> Tensor:
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder - used to encode observations to context for posterior
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        # Inital condition network - Generates a likely initial condition from context
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Neural SDEs
        # Posterior network, takes both latent state and context information of data
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # Prior network, takes just latent state
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        # Shared drift network, in this case diagonal
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )

        # Decoder - maps latent state to observed state
        self.projector = nn.Linear(latent_size, data_size)

        # Prior inital state distribution
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        # Data context
        self._ctx = None

    def contextualize(self, ctx: Tensor) -> None:
        """Function to update context"""
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    
    def f(self, t: float, y: Tensor) -> Tensor:
        """ Posterior drift network. Passes context along with the state"""
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t: float, y: Tensor) -> Tensor:
        """ Prior drift network"""
        return self.h_net(y)

    def g(self, t: float, y: Tensor) -> Tensor:  # Diagonal diffusion.
        """ Shared diffusion network. Current implementation requires diagonal noise"""
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs: Tensor, ts: Tensor, noise_std: Tensor, method: str="euler") -> tuple[float, float]:
        """ 
        Forward pass through Neural SDE.
        Steps:
            Encode the measurements into a context series
            Sample a starting point based on the context
            Integrate the SDE and its adjoint generating a posterior latent path (zs) and KL-divergence loss (log_ratio)
            Calculate the log likelihood loss of the decoded path (_xs) and measured path (xs): log_pxs
            Calculate the KL divergence loss of the inital condition

        Args:
            xs: Measurement data
            ts: Timestamps corresponding to measurements
            noise_std: Measurement noise standard deviation

        Returns:
            log_pxs: Log likelihood loss
            logqp0 + logqp_path: KL-divergence loss
        """
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        # Posterior initial condition
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)


        # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
        adjoint_params = (
                (ctx,) +
                tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
        )
        
        # Integrate SDE and adjoint
        zs, log_ratio = torchsde.sdeint_adjoint(
            self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)

        # Likelihood loss
        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        # Initial position KL divergence loss
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + logqp_path
    
    @torch.no_grad()
    def sample(self, batch_size: int, ts: Tensor, bm: torchsde._brownian.brownian_base.BaseBrownian=None):
        """ Sample batch of Neural SDEs and integrate """
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs

def train(
        batch_size=50,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=2.,
        lr_gamma=0.997,
        num_epochs=5000,
        kl_anneal_iters=1000,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./dump/',
        method="euler"
        ) -> None:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)  # Learning rate scheduler
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)  # KL annealing, start low

    # for epoch in tqdm.trange(1, num_epochs + 1):
    #     latent_sde.zero_grad()

    #     log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
    #     loss = -log_pxs + log_ratio * kl_scheduler.val

    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    #     kl_scheduler.step()
