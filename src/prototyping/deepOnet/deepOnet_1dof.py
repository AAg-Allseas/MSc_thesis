"""Training script for 1-DOF DeepONet (MIONet) model."""

import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn

from src.utils import is_databricks, save_checkpoint_artifact
from src.prototyping.data_handling import find_parquet_files
from src.prototyping.deepOnet.train_deepOnet import load_samples_sensors, train, test
from src.prototyping.deepOnet.model_deepOnet import MIONet
from src.prototyping.deepOnet.utils import BranchConstructor, CNN1DBranchConstructor, MLPConstructor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if is_databricks():
        base_path = Path("/Volumes/main_udev/ai_labs/aaperghis")
        mlflow.set_experiment("/Users/aag@allseas.com/experiments/DeepOnet_1dof")
    else:
        base_path = Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data")
        mlflow.set_experiment("DeepOnet_1dof")

    files_training = find_parquet_files(base_path / "1_dof",
                                   lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 80)
    files_testing = find_parquet_files(base_path / "1_dof",
                                   lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] > 80)

    sample_length = 10000

    feats_sensors = ['tau_ext_x']
    feats_samples = ['pos_eta_x']

    n_samples = 2048
    batch_size = 16
    n_epochs = 10000
    max_errors = 3

    dataset_training_samples, dataloader_training = load_samples_sensors(
        files_training, sample_length, feats_sensors, feats_samples, batch_size, sample_dt=0.5)
    dataset_testing_samples, dataloader_testing = load_samples_sensors(
        files_testing, sample_length, feats_sensors, feats_samples, batch_size, sample_dt=0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_dim = 128
    output_dim = 1
    input_dim = 1

    branches = [
        BranchConstructor(
            name="initial_conditions",
            layer_sizes=[input_dim, 100, latent_dim],
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
    ]

    trunk = MLPConstructor(
        layer_sizes=[1, 125, 250, 250, latent_dim],
        activation="gelu",
        dropout=0.1
    )

    mionet = MIONet(branches, trunk, output_dim).to(device)

    input_features = {"surge_force": 0}

    last_lr = 1e-2
    warmup_epochs = 0
    warmup_steps = warmup_epochs * len(dataloader_training)

    optimiser = torch.optim.Adam(params=mionet.parameters(), lr=last_lr)

    linear_warmup_schedule = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=1.0, total_iters=warmup_steps)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, min_lr=1e-6)
    scheduler = linear_warmup_schedule

    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name=f"1dof_bs{batch_size}_lr{last_lr}"):
        mlflow.log_params({
            "model_type": "MIONet",
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "input_dim": input_dim,
            "batch_size": batch_size,
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "lr_init": last_lr,
            "warmup_epochs": warmup_epochs,
            "sample_length": sample_length,
            "sample_dt": 0.5,
            "loss_fn": "MSELoss",
            "optimizer": "Adam",
            "grad_clip": 1.0,
            "n_training_files": len(files_training),
            "n_testing_files": len(files_testing),
            "features_sensors": str(feats_sensors),
            "features_samples": str(feats_samples),
        })
        mlflow.set_tags({
            "model_type": "MIONet",
            "variant": "1_dof",
            "device": str(device),
        })

        global_train_step = 0
        global_test_step = 0

        try:
            for epoch in range(n_epochs):
                global_train_step = train(
                    mionet, dataloader_training, dataset_training_samples, input_features,
                    optimiser, loss_fn, device, n_samples, max_errors, scheduler,
                    global_step=global_train_step)
                global_test_step, mean_test_loss = test(
                    mionet, dataloader_testing, dataset_testing_samples, input_features,
                    loss_fn, device, n_samples, global_step=global_test_step)

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


if __name__ == "__main__":
    main()
