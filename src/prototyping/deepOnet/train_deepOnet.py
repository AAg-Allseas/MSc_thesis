from pathlib import Path
from typing import Any, Dict
from matplotlib import pyplot as plt
import numpy as np
from src.logging import EPOCH_LEVEL_NUM, LossTracker, create_run_folder, get_logger, save_checkpoint
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch import optim
import tqdm

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset
from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.utils import BranchConstructor, MLPConstructor

def prepare_batch(batch: tuple[Tensor, Tensor, Dict[str, Any]], 
                  sample_dataset: ParquetDataset,
                  n_samples: int = -1,
                  ordered: bool = False, 
                  device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                  ) -> tuple[Dict[str, Tensor], Tensor, Tensor]:
        _, sensors, metas = batch
        sensors = sensors.to(device)

        pos = metas["inital_pos"]
        initial_conditions = torch.vstack(pos if isinstance(pos[0], Tensor) else [torch.tensor(pos)]).T.to(device, dtype=torch.float32)

        # Extract file index from batch indices (available via DataLoader's sampler)
        idxs = metas["idx"]
        if isinstance(idxs, int):
             ts, samples, _ = sample_dataset[idxs]

        else:
            ts, samples = ([], [])
            for idx in idxs:
                t, sample, _ = sample_dataset[idx]
                ts.append(t)
                samples.append(sample)
            
            ts = torch.stack(ts)
            samples = torch.stack(samples)
        if n_samples == -1:
             n_samples = ts.shape[-1]

        if ordered:
             ts = ts[..., :n_samples].to(device)
             samples = samples[:, :n_samples, :].to(device)
        else:
            idx = torch.randperm(ts.size(1))[:n_samples] 
            ts = ts[:, idx].to(device)
            samples = samples[:, idx].to(device)

        x = ({"initial_conditions": initial_conditions,
              "surge_force": sensors[..., 0],
              "sway_force": sensors[..., 1],
              "yaw_moment": sensors[..., 2]}, 
              ts)
        
        return x, samples, ts

if __name__ == "__main__":
    
    run_path = create_run_folder(sub="prototypes/deepOnet/testing")
    logger = get_logger(run_path, name="train")
    tracker = LossTracker(run_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.epoch(f"Running on {device}")
    # Latent dimension (shared by all branch outputs).
    latent_dim = 64
    # Number of features to predict.
    output_dim = 12


    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[3, 100, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="surge_force",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="sway_force",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
        BranchConstructor(
            name="yaw_moment",
            layer_sizes=[1000, 250, 250, latent_dim],
            activation="gelu"
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 250, 250, latent_dim],
        activation="gelu"
    )
        
    mionet = MIONet(branches, trunk, output_dim).to(device)
    logger.epoch(f"Initialised model")
    logger.debug(str(mionet))

    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 10)
    
    sample_length = 20000
    logger.epoch(f"Using {len(files)} with sample length {sample_length}")
    logger.debug(files)

    # Sensor measurements, resampled to 1Hz
    feats_sensors = [
        'tau_ext_x',
        'tau_ext_y',
        'tau_ext_mz'
    ]
    scales_sensors = np.array([1/75e3, 1/75e3, 1/100e3])
    dataset_sensors = ParquetDataset(files, 
                            columns=feats_sensors,
                            sample_length=sample_length,
                            scale_factors=scales_sensors,
                            resample_dt=1)
    
    # State samples at original resolution
    feats_samples = [
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
        'rpm_fixed_sb',]

    scales_samples = np.array([1, 1, 1, 1, 1, 1, 1/250, 1/250, 1/250, 1/250, 1/160, 1/160])
    dataset_samples = ParquetDataset(files, 
                             columns=feats_samples,
                             sample_length=sample_length,
                             scale_factors=scales_samples,
                             )
    

    n_samples = 1024
    batch_size = 24
    n_epochs = 10000
    
    dataloader = DataLoader(dataset_sensors, batch_size=batch_size, shuffle=True, pin_memory=True)

    rng = torch.Generator()

    optimiser = torch.optim.Adam(params=mionet.parameters(), lr=0.0005)
    scheduler= torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=n_epochs, pct_start=0.01)
    loss_fn = nn.MSELoss()

    logger.epoch(f"Starting training\n Training parameters: \n - {n_samples} samples \n - {batch_size} batch size \n - {n_epochs} epochs")
    try:
        for epoch in tqdm.trange(n_epochs):
            running_loss = 0
            logger.epoch(f"Epoch {epoch}")
            logger.epoch("-" * 25)

            for batch in dataloader:
                optimiser.zero_grad()
                
                x, samples, _ = prepare_batch(batch, dataset_samples, n_samples=n_samples, device=device)
                loss = loss_fn(mionet(x), samples)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mionet.parameters(), max_norm=1.0)

                optimiser.step()
                scheduler.step()

                tracker.log_batch(loss.item())
                logger.batch(f"Batch loss: {loss.item():.4f}")

            mean_loss = tracker.end_epoch(epoch)
            logger.epoch(f"Epoch {epoch} complete | Mean loss: {mean_loss}")
            save_checkpoint(mionet, run_path, epoch, optimiser, scheduler)
    except Exception as e:
         logger.error(e)
         

    # for batch in dataloader:
    #     with torch.no_grad():
    #         x, samples, ts = prepare_batch(batch, dataset_samples, ordered=True, device=device)
    #         plt.plot(ts.to("cpu").T, samples.to("cpu")[..., 0].T)
    #         plt.plot(ts.to("cpu").T, mionet(x).to("cpu")[..., 1].T)
    #         plt.show()
    #         break

