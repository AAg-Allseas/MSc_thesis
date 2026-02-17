from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset
from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.utils import BranchConstructor, MLPConstructor


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 1)
    
    sample_length = 20000

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
    

    n_samples = 512
    batch_size = 2

    dataloader = DataLoader(dataset_sensors, batch_size=batch_size)

    rng = torch.Generator()

    for batch in dataloader:
        _, sensors, metas = batch
        sensors = sensors.to(device)

        initial_conditions = torch.vstack(metas["inital_pos"]).T.to(device, dtype=torch.float32)

        # Extract file index from batch indices (available via DataLoader's sampler)
        idxs = metas["idx"]

        ts, samples = ([], [])
        for idx in idxs:
            t, sample, _ = dataset_samples[idx]
            ts.append(t)
            samples.append(sample)
        
        ts = torch.stack(ts)
        samples = torch.stack(samples)

        idx = torch.randperm(ts.size(1))[:n_samples] 
        ts = ts[:, idx].to(device)
        samples = samples[:, idx].to(device)

        x = ({"initial_conditions": initial_conditions,
              "surge_force": sensors[..., 0],
              "sway_force": sensors[..., 1],
              "yaw_moment": sensors[..., 2]}, 
              ts)
        pred = mionet(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, samples)
        print(loss)