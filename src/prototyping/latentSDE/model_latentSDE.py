from src.prototyping.latentSDE.utils import ContextEncoder, LipSwish
from src.prototyping.latentSDE.train_latentSDE import LOGGER


import torch
import torchsde
from torch import Tensor, nn
from torch.distributions import Normal


from typing import Optional, Tuple


class LatentSDE(torchsde.SDEIto):
    """Latent SDE model with learned drift, diffusion, and decoder."""

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        init_sigma: Optional[Tensor] = None,
    ) -> None:
        """Initialize networks and trainable parameters."""
        super(LatentSDE, self).__init__(noise_type="diagonal")
        # Encoder - used to encode observations to context for posterior
        self.encoder = ContextEncoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        # Inital condition network - Generates a likely initial condition from context
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Neural SDEs
        # Posterior network, takes both latent state and context information of data
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            LipSwish(),
            nn.Linear(hidden_size, hidden_size),
            LipSwish(),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh()
        )
        # Prior network, takes just latent state
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            LipSwish(),
            nn.Linear(hidden_size, hidden_size),
            LipSwish(),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh()
        )

        # Shared drift network, in this case diagonal
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    LipSwish(),
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

        # Noise standard deviation
        if init_sigma is None:
            init_sigma = torch.full((data_size, ), -1.0, dtype=torch.float32)
        elif init_sigma.shape != (data_size, 1):
            raise AttributeError("Inital log-sigma shape does not match data")

        self.log_sigma = nn.Parameter(init_sigma)

        # Data context
        self._ctx: Optional[tuple[Tensor, Tensor]] = None

    def contextualize(self, ctx: Tuple[Tensor, Tensor]) -> None:
        """Update cached context tensors."""
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).


    def f(self, t: float, y: Tensor) -> Tensor:
        """Posterior drift network. Passes context along with the state."""
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t: float, y: Tensor) -> Tensor:
        """Prior drift network."""
        return self.h_net(y)

    def g(self, t: float, y: Tensor) -> Tensor:  # Diagonal diffusion.
        """Shared diffusion network. Current implementation requires diagonal noise."""
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(
        self,
        xs: Tensor,
        ts: Tensor,
        adaptive: bool = False,
        method: str = "euler",
        dt: float = 0.01,
        bm: Optional[torchsde.BaseBrownian] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through Neural SDE.
        Steps:
            Encode the measurements into a context series
            Sample a starting point based on the context
            Integrate the SDE and its adjoint generating a posterior latent path (zs) and KL-divergence loss (log_ratio)
            Calculate the log likelihood loss of the decoded path (_xs) and measured path (xs): log_pxs
            Calculate the KL divergence loss of the inital condition

        Args:
            xs: Batch of measurement data. Dimensions: [Time, batches, state]
            ts: Timestamps corresponding to measurements

        Returns:
            log_pxs: Log likelihood loss
            logqp0 + logqp_path: KL-divergence loss
        """
        # Contextualization is only needed for posterior inference.
        # .contiguous() is required because torch.flip creates negative strides that cuDNN doesn't support.
        ctx = self.encoder(torch.flip(xs, dims=(0,)).contiguous())
        ctx = torch.flip(ctx, dims=(0,)).contiguous()
        self.contextualize((ts, ctx))

        # Posterior initial condition
        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        qz0_logstd = qz0_logstd.clamp(-5, 2)  # Prevent explosion
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
        adjoint_params = (
                (ctx,) +
                tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
        )

        # Integrate SDE and adjoint
        LOGGER.info("   - Integrating SDE")
        zs, log_ratio = torchsde.sdeint_adjoint(
            self, z0, ts, adjoint_params=adjoint_params, dt=dt, logqp=True, method=method, bm=bm)

        # Clamp log_sigma to prevent extreme values
        log_sigma_clamped = self.log_sigma.clamp(-4, 2)
        sigma = log_sigma_clamped.exp()
        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=sigma)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(2)).mean() # Sum over features, mean over batch and time

        # Initial position KL divergence loss
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.clamp(-5, 2).exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0) # Mean over batch and time, sum over features
        logqp_path = log_ratio.mean()  # Mean over batch and time
        return log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        ts: Tensor,
        bm: Optional[torchsde._brownian.brownian_base.BaseBrownian] = None,
        method: str = "euler",
    ) -> Tensor:
        """Sample batch of Neural SDEs and integrate."""
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm, method=method)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs