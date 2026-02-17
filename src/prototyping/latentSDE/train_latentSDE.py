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
from typing import Optional

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader

import torchsde

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset, prep_batch
from src.prototyping.latentSDE.model_latentSDE import LatentSDE
from src.prototyping.latentSDE.utils import LinearScheduler

LOGGER = logging.getLogger(__name__)

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