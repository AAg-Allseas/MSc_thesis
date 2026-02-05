"""
Main script for training a Latent SDE on the toy DP model. 

Most of the code is copied from latent_sde_lorentz.py - https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

Based on approach from:
Li, X., Wong, T.-K. L., Chen, R. T. Q., & Duvenaud, D. (2020). 
Scalable Gradients for Stochastic Differential Equations. 
Proceedings of the 23rd International Conference on Artificial Intelligence and Statistic, 108. 
https://doi.org/10.48550/arXiv.2001.01328

"""

from pathlib import Path
import torch
import tqdm
from torch import nn, Tensor
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader

import torchsde
from typing import Optional

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.latentSDE.dataloader import ParquetDataset, prep_batch


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
    """LSTM encoder to encode observation data into context for the posterior network.
    
    Uses chunked processing to handle long sequences that exceed cuDNN's ~65k limit.
    """
    # cuDNN LSTM has a max sequence length of ~65k 
    CUDNN_SEQ_LIMIT = 60000
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor) -> Tensor:
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

class LatentSDE(torchsde.SDEStratonovich):

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__(noise_type="diagonal")
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
        self._ctx: Optional[tuple[Tensor, Tensor]] = None

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

    def forward(self, xs: Tensor, ts: Tensor, noise_std: Tensor, adaptive: bool=False, method: str="reversible_heun") -> tuple[float, float]:
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
            noise_std: Measurement noise standard deviation

        Returns:
            log_pxs: Log likelihood loss
            logqp0 + logqp_path: KL-divergence loss
        """
        # Contextualization is only needed for posterior inference.
        # .contiguous() is required because torch.flip creates negative strides that cuDNN doesn't support
        ctx = self.encoder(torch.flip(xs, dims=(0,)).contiguous())
        ctx = torch.flip(ctx, dims=(0,)).contiguous()
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
            self, z0, ts, adjoint_params=adjoint_params, dt=5e-2, logqp=True, method=method)

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
        method="reversible_heun"
        ) -> None:
    
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05)
    
    dataset = ParquetDataset(files, columns=['pos_eta_x', 
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
                                             'rpm_fixed_sb'])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

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

    for epoch in tqdm.trange(1, num_epochs + 1):
        for batch in dataloader:
            ts, xs = prep_batch(batch, device)

            latent_sde.zero_grad()

            log_pxs, log_ratio = latent_sde(xs, ts, noise_std, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val

            loss.backward()
            optimizer.step()
            scheduler.step()
            kl_scheduler.step()
        
        if epoch % pause_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:06d}, lr: {lr_now:.5f}, log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f}, loss: {loss:.4f}')

if __name__ == "__main__":
    train(batch_size=2)