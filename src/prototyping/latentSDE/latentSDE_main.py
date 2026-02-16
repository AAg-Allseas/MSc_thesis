"""Main script for training a Latent SDE on the toy DP model.

Most of the code is copied from latent_sde_lorentz.py - https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

Based on approach from:
Li, X., Wong, T.-K. L., Chen, R. T. Q., & Duvenaud, D. (2020). 
Scalable Gradients for Stochastic Differential Equations. 
Proceedings of the 23rd International Conference on Artificial Intelligence and Statistic, 108. 
https://doi.org/10.48550/arXiv.2001.01328

"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tqdm
from torch import nn, Tensor
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

import torchsde

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset, prep_batch

LOGGER = logging.getLogger(__name__)

class LipSwish(nn.Module):
    """Lipschitz-constrained Swish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x) / 1.1
    
class LinearScheduler(object):
    """Linear scheduler that ramps a value from 0 to maxval."""

    def __init__(self, iters: int, maxval: float = 1.0) -> None:
        self._iters = max(1, iters)
        self._val = 0
        self._maxval = maxval

    def step(self) -> None:
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self) -> float:
        return self._val
    
class Encoder(nn.Module):
    """GRU encoder to encode observation data into context for the posterior network.
    
    Uses chunked processing to handle long sequences that exceed cuDNN's ~65k limit.
    """
    # cuDNN GRU has a max sequence length of ~65k 
    CUDNN_SEQ_LIMIT = 60000
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor) -> Tensor:
        """Encode a sequence into a context series."""
        seq_len = inp.size(0)
        
        # For short sequences, process directly
        if seq_len <= self.CUDNN_SEQ_LIMIT:
            self.gru.flatten_parameters()
            out, _ = self.gru(inp)
        else:
            # Process long sequences in chunks to avoid cuDNN limit
            outputs = []
            hidden = None
            for start in range(0, seq_len, self.CUDNN_SEQ_LIMIT):
                end = min(start + self.CUDNN_SEQ_LIMIT, seq_len)
                chunk = inp[start:end]
                self.gru.flatten_parameters()
                out_chunk, hidden = self.gru(chunk, hidden)
                outputs.append(out_chunk)
            out = torch.cat(outputs, dim=0)
        
        return self.lin(out)

class LatentSDE(torchsde.SDEIto):
    """Latent SDE model with learned drift, diffusion, and decoder."""

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        init_sigma: Optional[Tensor] = None,
    ) -> None:  # type: ignore
        """Initialize networks and trainable parameters."""
        super(LatentSDE, self).__init__(noise_type="diagonal")
        # Encoder - used to encode observations to context for posterior
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
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

def train(
    batch_size: int = 50,
    latent_size: int = 4,
    context_size: int = 64,
    hidden_size: int = 128,
    lr_init: float = 5e-3,
    t0: float = 0.0,
    t1: float = 2.0,
    lr_gamma: float = 0.997,
    num_epochs: int = 5000,
    kl_anneal_iters: int = 1000,
    pause_every: int = 1,
    noise_std: float = 0.1,
    adjoint: bool = False,
    train_dir: str = './dump/',
    method: str = "euler",
) -> None:
    """Train the Latent SDE model."""
    
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 1)
    
    sample_length = 5000
    feats = [
        'pos_eta_x',
        'pos_eta_y',
        'pos_eta_mz',
        'pos_nu_x',
        'pos_nu_y',
        'pos_nu_mz',
        'rpm_bow_fore',
        'rpm_bow_aft',
        'rpm_stern_fore',
        'rpm_stern_aft',
        'rpm_fixed_ps',
        'rpm_fixed_sb',
    ]

    scales = np.array([1, 1, 1, 1, 1, 1, 1/250, 1/250, 1/250, 1/250, 1/160, 1/160])
    dataset = ParquetDataset(files, 
                             columns=feats,
                             sample_length=sample_length,
                             scale_factors=scales)

    LOGGER.info("Training Parameters:")
    LOGGER.info(f"  batch_size: {batch_size}")
    LOGGER.info(f"  latent_size: {latent_size}")
    LOGGER.info(f"  context_size: {context_size}")
    LOGGER.info(f"  hidden_size: {hidden_size}")
    LOGGER.info(f"  lr_init: {lr_init}")
    LOGGER.info(f"  t0: {t0}, t1: {t1}")
    LOGGER.info(f"  lr_gamma: {lr_gamma}")
    LOGGER.info(f"  num_epochs: {num_epochs}")
    LOGGER.info(f"  kl_anneal_iters: {kl_anneal_iters}")
    LOGGER.info(f"  noise_std: {noise_std}")
    LOGGER.info(f"  method: {method}")

    LOGGER.info("Dataset Configuration:")
    LOGGER.info(f"  sample_length: {sample_length}")
    LOGGER.info(f"  columns: {feats}")
    LOGGER.info(f"  scale_factors: {scales}")
    LOGGER.info(f"  num_files: {len(files)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    LOGGER.info("Created DataLoader")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Running on {device}")

    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    
    # After model creation, initialize f_net's first layer so context columns are zero
    with torch.no_grad():
        # f_net[0] is Linear(latent_size + context_size, hidden_size)
        # h_net[0] is Linear(latent_size, hidden_size)
        # Copy h_net weights into the latent portion of f_net
        latent_sde.f_net[0].weight[:, :latent_size] = latent_sde.h_net[0].weight.clone()
        latent_sde.f_net[0].weight[:, latent_size:] = 0.0  # zero out context columns
        latent_sde.f_net[0].bias[:] = latent_sde.h_net[0].bias.clone()
        
        # Copy matching hidden layers
        for layer_idx in [2, 4]:  # second and third Linear layers
            latent_sde.f_net[layer_idx].weight[:] = latent_sde.h_net[layer_idx].weight.clone()
            latent_sde.f_net[layer_idx].bias[:] = latent_sde.h_net[layer_idx].bias.clone()

    # Override log_sigma to start at 0 so sigma = 1.0.
    with torch.no_grad():
        latent_sde.log_sigma.fill_(0.0)

    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)  # Learning rate scheduler
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)  # KL annealing, start low

    dt = 0.05
    kl_losses = np.empty((num_epochs, len(dataloader)))
    likelihood_losses = np.empty((num_epochs, len(dataloader)))
    

    try:
        for epoch in tqdm.trange(0, num_epochs):
            LOGGER.info(f"Epoch {epoch} - {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            LOGGER.info("-" * 25)
            for i, batch in enumerate(dataloader):
                
                LOGGER.info(f" - Batch {i} - {datetime.datetime.now().strftime('%H:%M:%S')}")
                ts, xs = prep_batch(batch, device)
                if xs.shape != (sample_length, batch_size, len(feats)):
                    LOGGER.info("Skipping batch - Inconsistent size")
                    continue
                # (Batch size, latent size + 1) to account for augmented state
                bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_size, latent_size + 1), dt=dt, device=device)
                latent_sde.zero_grad()

                log_pxs, log_ratio = latent_sde(xs, ts, method=method, dt=dt, bm=bm)
                loss = -log_pxs + log_ratio * kl_scheduler.val
                
                likelihood_losses[epoch, i] = -log_pxs.to("cpu").detach()
                kl_losses[epoch, i] = log_ratio.to("cpu").detach() * kl_scheduler.val


                # NaN guard
                if not torch.isfinite(loss):
                    LOGGER.warning(f"Non-finite loss at epoch {epoch}, batch {i}. Skipping update.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            kl_scheduler.step()
            
            if epoch % pause_every == 0:
                torch.save(latent_sde.state_dict(), f"./dump/latentsde/latentsde_{epoch}.pth")
                np.save("./dump/latentsde/kl_losses", kl_losses)
                np.save("./dump/latentsde/likelihood_losses", likelihood_losses)

                lr_now = optimizer.param_groups[0]['lr']
                LOGGER.info("=" * 50)
                LOGGER.info(f'Epoch {epoch:06d}\n lr: {lr_now:.5f}\n log_pxs: {log_pxs:.4f}\n log_ratio: {log_ratio:.4f}\n loss: {loss:.4f}')
                LOGGER.info("=" * 50)
            
    except Exception as e:
        LOGGER.error(f"Error at {datetime.datetime.now().strftime('%H:%M:%S')} \n {e}")
        try:
            LOGGER.info(f"Losses at time of error: \n log_pxs: {log_pxs:.4f}\n log_ratio: {log_ratio:.4f}\n loss: {loss:.4f}")
            LOGGER.info(f"Model Parameters: \n Log(Sigma): {latent_sde.log_sigma}")
        except Exception:
            pass
        raise e

def test_sample(
    batch_size: int = 50,
    latent_size: int = 4,
    context_size: int = 64,
    hidden_size: int = 128,
    ts: Optional[Tensor] = None,
    bm: Optional[torchsde.BaseBrownian] = None,
) -> Tensor:
    """Sample trajectories from a freshly initialized model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Running on {device}")

    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    if ts is None:
        ts = np.arange(0, 10800, 0.02)
        ts = torch.from_numpy(ts).to(device)
    if bm is None:
        bm = torchsde.BrownianInterval(0, 10800, size=(batch_size, latent_size), dt=0.02, device=device)

    LOGGER.info(f"Starting sampling - {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    sample = latent_sde.sample(batch_size=batch_size, ts=ts, bm=bm)
    LOGGER.info(f"Finished sampling - {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    return sample
if __name__ == "__main__":
    try:
        logging.basicConfig(filename="./logs/latentSDE.log", level=logging.INFO)
        LOGGER.info(f"Running {__file__}")
        train(batch_size=25)
        # test_sample()
        LOGGER.info(f"Finished running {__file__}")
    except KeyboardInterrupt:
        LOGGER.warning(f"Run interrupted {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")