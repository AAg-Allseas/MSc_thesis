"""Training script for MIONet (Multiple-Input Operator Network).

This module provides training and testing routines for the DeepONet-based
model, including loss tracking, checkpointing, and learning rate scheduling.

Example:
    Run training from the command line::

        python -m src.prototyping.deepOnet.train_deepOnet
"""
from pathlib import Path
import numpy as np
from src.logging import LossTracker, create_run_folder, get_logger, save_checkpoint
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset
from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.utils import BranchConstructor, MLPConstructor, prepare_batch


def train(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_samples: ParquetDataset,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_samples: int,
    max_errors: int,
    logger,
    tracker: LossTracker,
) -> None:
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
        logger: Logger instance for recording metrics.
        tracker: LossTracker for accumulating batch losses.

    Raises:
        ValueError: If non-finite losses exceed max_errors threshold.
    """
    model.train()
    error_count = 0
    for batch in dataloader:
        optimiser.zero_grad()
                
        x, samples, _ = prepare_batch(batch, dataset_samples, n_samples=n_samples, device=device)
        loss = loss_fn(model(x), samples)

        # Early detection of numerical instability
        if not torch.isfinite(loss):
            error_count += 1
            logger.warning(f"Non-finite loss detected: {loss.item()}, current error count: {error_count}")
            logger.batch(f"Batch loss: {loss.item():.4f}")
            if error_count >= max_errors:
                logger.error(f"Errors exceded threshold of {max_errors}. Stopping training")
                raise ValueError(f"Training diverged with loss={loss.item()}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()

        tracker.log_batch(loss.item())


def test(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_samples: ParquetDataset,
    loss_fn: nn.Module,
    device: torch.device,
    n_samples: int,
    logger,
    tracker: LossTracker,
) -> None:
    """Evaluate model on test data.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader providing test batches.
        dataset_samples: Dataset providing target samples.
        loss_fn: Loss function (e.g., MSELoss).
        device: Torch device (CPU or CUDA).
        n_samples: Number of time samples per batch.
        logger: Logger instance for recording metrics.
        tracker: LossTracker for accumulating batch losses.
    """
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, samples, _ = prepare_batch(batch, dataset_samples, n_samples=n_samples, ordered=True, device=device)
            loss = loss_fn(model(x), samples)
            
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite test loss: {loss.item()}")
                continue
            
            tracker.log_batch(loss.item(), "test")
            logger.batch(f"Batch testing loss: {loss.item():.4f}")


def load_samples_sensors(
    files: list[Path],
    sample_length: int,
    feats_sensors: list[str],
    scales_sensors: np.ndarray,
    feats_samples: list[str],
    scales_samples: np.ndarray,
    batch_size: int,
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
                        scale_factors=scales_sensors,
                        resample_dt=1)
    dataset_samples = ParquetDataset(files, 
                             columns=feats_samples,
                             sample_length=sample_length,
                             scale_factors=scales_samples,
                             )

    dataloader = DataLoader(dataset_sensors, batch_size=batch_size, shuffle=shuffel, pin_memory=True)

    return dataset_samples,dataloader

if __name__ == "__main__":
    
    run_path = create_run_folder(sub="prototypes/deepOnet/testing")
    logger = get_logger(run_path, name="train")
    loss_tracker = LossTracker(run_path)

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

    mionet.load_state_dict(torch.load(r"runs\prototypes\deepOnet\testing\starting_models\2026-02-17_17-22-31_epoch_1700.pth")["model_state_dict"])
    mionet.eval()
    logger.epoch("Loaded model from saved file")

    logger.epoch("Initialised model")
    logger.debug(str(mionet))

    files_training = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] <= 40)
    files_testing = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] > 40)
    
    sample_length = 20000
    logger.epoch(f"Using {len(files_training)} with sample length {sample_length} for training")
    logger.debug(files_training)

    logger.epoch(f"Using {len(files_testing)} with sample length {sample_length} for testing")
    logger.debug(files_testing)

    # Sensor measurements, resampled to 1Hz
    feats_sensors = [
        'tau_ext_x',
        'tau_ext_y',
        'tau_ext_mz'
    ]
    scales_sensors = np.array([1/75e3, 1/75e3, 1/100e3])

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

    n_samples = 2048
    batch_size = 64
    n_epochs = 10000
    max_errors = 3
    
    dataset_training_samples, dataloader_training = load_samples_sensors(files_training, sample_length, feats_sensors, scales_sensors, feats_samples, scales_samples, batch_size)
    dataset_testing_samples, dataloader_testing = load_samples_sensors(files_testing, sample_length, feats_sensors, scales_sensors, feats_samples, scales_samples, batch_size)
    
    rng = torch.Generator()

    last_lr = 0.005
    optimiser = torch.optim.Adam(params=mionet.parameters(), lr=last_lr)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    logger.epoch(f"Starting training\n Training parameters: \n - {n_samples} samples \n - {batch_size} batch size \n - {n_epochs} epochs")
    try:
        for epoch in tqdm.trange(n_epochs):
            logger.epoch(f"Epoch {epoch}")
            logger.epoch("-" * 25)
            
            train(mionet, dataloader_training, dataset_training_samples, optimiser, loss_fn, device, n_samples, max_errors, logger, loss_tracker)
            test(mionet, dataloader_testing, dataset_testing_samples, loss_fn, device, n_samples, logger, loss_tracker)
            
            mean_training_loss, mean_testing_loss = loss_tracker.end_epoch(epoch)
            scheduler.step(mean_training_loss)  # Step scheduler once per epoch with mean loss
            lr = optimiser.param_groups[0]['lr']

            logger.epoch(f"Epoch {epoch} complete | Mean loss (training): {mean_training_loss:.4f} | Mean loss (testing): {mean_testing_loss:.4f}")
            if lr != last_lr:
                 logger.epoch(f"Learning rate reduced. \n Old LR: {last_lr} | New LR: {lr}")
            
            save_checkpoint(mionet, run_path, epoch, optimiser, scheduler)
            logger.epoch("Model saved")
            logger.epoch("-" * 25)

            last_lr = lr
    except Exception as e:
         logger.error(e)
    
