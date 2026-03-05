import os
from typing import Optional

import mlflow
import torch
from torch import nn
import tempfile

import mlflow
import torch
from torch import nn
import tempfile

def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def save_checkpoint_artifact(
    model: nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> None:
    """Save a model checkpoint as an MLflow artifact.

    Args:
        model: The model to save.
        epoch: Current epoch number (used in filename).
        optimizer: Optional optimizer to save state from.
        scheduler: Optional scheduler to save state from.
    """
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        mlflow.log_artifact(path, artifact_path="checkpoints")

def databricks_test_func() -> None:
    print("Test v2")