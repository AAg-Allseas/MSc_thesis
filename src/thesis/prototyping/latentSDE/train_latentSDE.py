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
import os
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader

import torchsde

from thesis.prototyping.data_handling import find_parquet_files
from thesis.prototyping.dataloader import ParquetDataset, prep_batch
from thesis.prototyping.latentSDE.model_latentSDE import LatentSDE
from thesis.prototyping.latentSDE.utils import LinearScheduler

logger = logging.getLogger(__name__)

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
                             standardise=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on {device}")

    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    
    # After model creation, initialize f_net's first layer so context columns are zero
    with torch.no_grad():
        latent_sde.f_net[0].weight[:, :latent_size] = latent_sde.h_net[0].weight.clone()
        latent_sde.f_net[0].weight[:, latent_size:] = 0.0
        latent_sde.f_net[0].bias[:] = latent_sde.h_net[0].bias.clone()
        
        for layer_idx in [2, 4]:
            latent_sde.f_net[layer_idx].weight[:] = latent_sde.h_net[layer_idx].weight.clone()
            latent_sde.f_net[layer_idx].bias[:] = latent_sde.h_net[layer_idx].bias.clone()

    with torch.no_grad():
        latent_sde.log_sigma.fill_(0.0)

    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    dt = 0.05

    mlflow.set_experiment("latentSDE")

    with mlflow.start_run(run_name=f"latentsde_bs{batch_size}_lr{lr_init}"):
        mlflow.log_params({
            "model_type": "LatentSDE",
            "batch_size": batch_size,
            "latent_size": latent_size,
            "context_size": context_size,
            "hidden_size": hidden_size,
            "lr_init": lr_init,
            "lr_gamma": lr_gamma,
            "num_epochs": num_epochs,
            "kl_anneal_iters": kl_anneal_iters,
            "noise_std": noise_std,
            "method": method,
            "sample_length": sample_length,
            "dt": dt,
            "n_files": len(files),
        })

        global_step = 0

        try:
            for epoch in tqdm.trange(0, num_epochs):
                epoch_losses = []
                for i, batch in enumerate(dataloader):
                    ts, xs = prep_batch(batch, device)
                    if xs.shape != (sample_length, batch_size, len(feats)):
                        logger.warning("Skipping batch - Inconsistent size")
                        continue

                    bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_size, latent_size + 1), dt=dt, device=device)
                    latent_sde.zero_grad()

                    log_pxs, log_ratio = latent_sde(xs, ts, method=method, dt=dt, bm=bm)
                    loss = -log_pxs + log_ratio * kl_scheduler.val

                    mlflow.log_metric("batch_loss", loss.item(), step=global_step)
                    epoch_losses.append(loss.item())
                    global_step += 1

                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss at epoch {epoch}, batch {i}. Skipping update.")
                        optimizer.zero_grad()
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                kl_scheduler.step()

                mean_train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
                lr_now = optimizer.param_groups[0]['lr']

                mlflow.log_metrics({
                    "epoch_train_loss": mean_train_loss,
                    "learning_rate": lr_now,
                    "kl_weight": kl_scheduler.val,
                }, step=epoch)

                logger.info(f"Epoch {epoch:06d} | lr: {lr_now:.5f} | mean_loss: {mean_train_loss:.4f}")
                
                if epoch % pause_every == 0:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch}.pth")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": latent_sde.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        }, path)
                        mlflow.log_artifact(path, artifact_path="checkpoints")
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def test_sample(
    batch_size: int = 50,
    latent_size: int = 4,
    context_size: int = 64,
    hidden_size: int = 128,
    ts: Optional[Tensor] = None,
    bm: Optional[torchsde.BaseBrownian] = None,
) -> Tensor:
    """Sample trajectories from a freshly initialized model."""
    logger = logging.getLogger("train")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on {device}")

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

    logger.info(f"Starting sampling - {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    sample = latent_sde.sample(batch_size=batch_size, ts=ts, bm=bm)
    logger.info(f"Finished sampling - {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    return sample

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        train(batch_size=25)
    except KeyboardInterrupt:
        logger.warning(f"Run interrupted {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")