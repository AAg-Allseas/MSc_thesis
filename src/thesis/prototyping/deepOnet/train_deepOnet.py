"""Training script for MIONet (Multiple-Input Operator Network).

This module provides training and testing routines for the DeepONet-based
model, including loss tracking, checkpointing, and learning rate scheduling.

Example:
    Run training from the command line::

        python -m thesis.prototyping.deepOnet.train_deepOnet
"""
import logging
from pathlib import Path
from typing import Optional
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from thesis.prototyping.data_handling import find_parquet_files
from thesis.prototyping.dataloader import ParquetDataset
from thesis.prototyping.deepOnet.model_deepOnet import MIONet
from thesis.prototyping.deepOnet.utils import BranchConstructor, CNN1DBranchConstructor, MLPConstructor, prepare_batch
from thesis.utils import save_checkpoint_artifact

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_samples: ParquetDataset,
    input_features: dict[str, int],
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_samples: int,
    max_errors: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int = 0,
) -> int:
    """Run one training epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing input batches.
        dataset_samples: Dataset providing target samples.
        optimiser: Optimizer for updating model parameters.
        loss_fn: Loss function (e.g., MSELoss).
        device: Torch device (CPU or CUDA).
        n_samples: Number of time samples per batch.
        max_errors: Maximum consecutive non-finite losses before stopping.
        lr_scheduler: Learning rate scheduler.
        global_step: Current global batch step counter for MLflow logging.

    Returns:
        Updated global_step counter.

    Raises:
        ValueError: If non-finite losses exceed max_errors threshold.
    """
    model.train()
    error_count = 0
    n_batches = len(dataloader)
    log_interval = n_batches // 3

    for i, batch in enumerate(dataloader):
        optimiser.zero_grad()
                
        x, samples, _ = prepare_batch(batch, dataset_samples, input_features,  n_samples=n_samples, device=device)
        loss = loss_fn(model(x), samples)

        # Early detection of numerical instability
        if not torch.isfinite(loss):
            error_count += 1
            logger.warning(f"Non-finite loss detected: {loss.item()}, current error count: {error_count}")
            if error_count >= max_errors:
                raise ValueError(f"Training diverged with loss={loss.item()}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        if i % log_interval == 0:
            mlflow.log_metric("batch_train_loss", loss.item(), step=global_step)

        global_step += 1

        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step()

    return global_step


def test(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_samples: ParquetDataset,
    input_features: dict[str, int], 
    loss_fn: nn.Module,
    device: torch.device,
    n_samples: int,
    global_step: int = 0,
) -> tuple[int, float]:
    """Evaluate model on test data.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader providing test batches.
        dataset_samples: Dataset providing target samples.
        loss_fn: Loss function (e.g., MSELoss).
        device: Torch device (CPU or CUDA).
        n_samples: Number of time samples per batch.
        global_step: Current global test batch step counter for MLflow logging.

    Returns:
        Tuple of (updated global_step, mean test loss).
    """
    model.eval()
    test_losses: list[float] = []
    n_batches = len(dataloader)
    log_interval = n_batches // 3

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, samples, _ = prepare_batch(batch, dataset_samples, input_features, n_samples=n_samples, device=device)
            loss = loss_fn(model(x), samples)
            
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite test loss: {loss.item()}")
                continue
            
            test_losses.append(loss.item())
            if i % log_interval == 0:
                mlflow.log_metric("batch_test_loss", loss.item(), step=global_step)

            global_step += 1

    mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0.0
    return global_step, mean_test_loss


def load_samples_sensors(
    files: list[Path],
    sample_length: int,
    feats_sensors: list[str],
    feats_samples: list[str],
    batch_size: int,
    sample_dt: float=1.0, 
    scale: bool = True,
    shuffel: bool = True,
) -> tuple[ParquetDataset, DataLoader]:
    """Load sensor and sample datasets and create a DataLoader.

    Args:
        files: List of parquet file paths.
        sample_length: Number of timesteps per sample.
        feats_sensors: Column names for sensor features (resampled to 1Hz).
        scales_sensors: Scale factors for sensor features.
        feats_samples: Column names for sample features.
        scales_samples: Scale factors for sample features.
        batch_size: Batch size for the DataLoader.
        shuffel: Whether to shuffle data (default True).

    Returns:
        Tuple of (sample dataset, DataLoader for sensors).
    """
    dataset_sensors = ParquetDataset(files, 
                        columns=feats_sensors,
                        sample_length=sample_length,
                        standardise=scale,
                        resample_dt=sample_dt)
    dataset_samples = ParquetDataset(files, 
                             columns=feats_samples,
                             sample_length=sample_length,
                             standardise=scale,
                             )

    dataloader = DataLoader(dataset_sensors, batch_size=batch_size, shuffle=shuffel, pin_memory=True)

    return dataset_samples,dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on {device}")

    latent_dim = 128
    output_dim = 12

    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[12, 100, latent_dim],
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="surge_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="sway_force",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
        CNN1DBranchConstructor(
            name="yaw_moment",
            in_channels=1,
            channels=[32, 64, 128],
            kernel_sizes=[7, 5, 3],
            output_dim=latent_dim,
            activation="gelu",
            dropout=0.1
        ),
    ]
    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu",
        dropout=0.1
    )
        
    mionet = MIONet(branches, trunk, output_dim).to(device)

    logger.info("Initialised model")

    files_training = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] <= 40)
    files_testing = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] > 40)
    
    sample_length = 10000
    logger.info(f"Using {len(files_training)} files with sample length {sample_length} for training")
    logger.info(f"Using {len(files_testing)} files with sample length {sample_length} for testing")

    feats_sensors = [
        'tau_ext_x',
        'tau_ext_y',
        'tau_ext_mz'
    ]
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
    
    input_features = {"surge_force": 0,
                      "surge_force": 1,
                      "yaw_moment": 2}

    n_samples = 2048
    batch_size = 16
    n_epochs = 10000
    max_errors = 3
    


    dataset_training_samples, dataloader_training = load_samples_sensors(files_training, sample_length, feats_sensors, feats_samples, batch_size, sample_dt=0.5)
    dataset_testing_samples, dataloader_testing = load_samples_sensors(files_testing, sample_length, feats_sensors, feats_samples, batch_size, sample_dt=0.5)

    last_lr = 1e-3
    warmup_epochs = 100
    warmup_steps = warmup_epochs * len(dataloader_training)

    optimiser = torch.optim.Adam(params=mionet.parameters(), lr=last_lr)

    linear_warmup_schedule = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, min_lr=1e-6)
    scheduler = linear_warmup_schedule

    loss_fn = nn.MSELoss()

    mlflow.set_experiment("deepOnet_testing")

    with mlflow.start_run(run_name=f"mionet_bs{batch_size}_lr{last_lr}"):
        mlflow.log_params({
            "model_type": "MIONet",
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "batch_size": batch_size,
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "lr_init": last_lr,
            "warmup_epochs": warmup_epochs,
            "sample_length": sample_length,
            "loss_fn": "MSELoss",
            "optimizer": "Adam",
            "grad_clip": 1.0,
            "n_training_files": len(files_training),
            "n_testing_files": len(files_testing),
        })

        global_train_step = 0
        global_test_step = 0

        try:
            for epoch in tqdm.tqdm(range(n_epochs)):
                global_train_step = train(mionet, dataloader_training, dataset_training_samples, input_features, optimiser, loss_fn, device, n_samples, max_errors, scheduler, global_step=global_train_step)
                global_test_step, mean_test_loss = test(mionet, dataloader_testing, dataset_testing_samples, input_features, loss_fn, device, n_samples, global_step=global_test_step)

                if epoch == warmup_epochs:
                    scheduler = plateau_scheduler

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(mean_test_loss)
                
                lr = optimiser.param_groups[0]['lr']

                mlflow.log_metrics({
                    "epoch_test_loss": mean_test_loss,
                    "learning_rate": lr,
                }, step=epoch)

                logger.info(f"Epoch {epoch} | test_loss: {mean_test_loss:.4f} | lr: {lr:.6f}")

                if epoch % 50 == 0:
                    save_checkpoint_artifact(mionet, epoch, optimiser, scheduler)

                last_lr = lr

            mlflow.pytorch.log_model(mionet, artifact_path="model")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
