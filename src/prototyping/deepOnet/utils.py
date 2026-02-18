"""Helper functions and classes for DeepONet model construction.

Based on:
    Moya, C., Zhang, S., Lin, G., & Yue, M. (2023).
    DeepONet-grid-UQ: A trustworthy deep operator framework for predicting
    the power grid's post-fault trajectories.
    Neurocomputing, 535, 166-182. https://doi.org/10.1016/j.neucom.2023.03.015
"""
from dataclasses import dataclass
from typing import Any, Dict
from torch import nn, Tensor
import torch

from src.prototyping.dataloader import ParquetDataset

# get activation function from str
def get_activation(identifier: str) -> nn.Module:
    """Return a PyTorch activation module by string key.

    Args:
        identifier: Activation name key.

    Returns:
        Instantiated activation module.

    Raises:
        KeyError: If the identifier is not in the activation map.
    """
    return{
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "Rrelu": nn.RReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "Mish": nn.Mish(),
            "identity": nn.Identity(),
    }[identifier]

@dataclass
class MLPConstructor:
    """Configuration container for the MLP builder.

    Attributes:
        layer_sizes: List of layer sizes, including input and output.
        activation: Activation name used between linear layers.
    """
    layer_sizes: list[int]
    activation: str


@dataclass
class BranchConstructor(MLPConstructor):
    """Configuration for a named branch network in MIONet.

    Attributes:
        layer_sizes: List of layer sizes, including input and output.
        activation: Activation name used between linear layers.
        name: Unique identifier for this branch (used as model attribute name).
    """

    name: str


class MLP(nn.Module):
    """Simple multi-layer perceptron assembled from a constructor configuration.

    Attributes:
        net: ModuleList containing Linear and activation layers.
    """

    def __init__(self, constructor: MLPConstructor) -> None:
        """Build the MLP from a constructor configuration.

        Args:
            constructor: MLPConstructor describing layer sizes and activation.
        """

        super(MLP, self).__init__()
        self.net = nn.ModuleList()

        for k in range(len(constructor.layer_sizes) - 2):
            self.net.append(nn.Linear(constructor.layer_sizes[k], constructor.layer_sizes[k+1], bias=True))
            self.net.append(get_activation(constructor.activation))

        self.net.append(nn.Linear(constructor.layer_sizes[-2], constructor.layer_sizes[-1], bias=True))
        self.net.apply(self._init_weights)
    
    def _init_weights(self, m: Any) -> None:
        """Initialize linear layers with Xavier normal weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all layers.
        """
        y = x
        for k in range(len(self.net)):
            y = self.net[k](y)
        return y


def prepare_batch(
    batch: tuple[Tensor, Tensor, Dict[str, Any]],
    sample_dataset: ParquetDataset,
    n_samples: int = -1,
    ordered: bool = False,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Prepare a batch of data for MIONet forward pass.

    Extracts sensor inputs, initial conditions, and target samples from a batch,
    then formats them for the model. Optionally subsamples time points.

    Args:
        batch: Tuple of (unused, sensors, metadata) from the DataLoader.
        sample_dataset: Dataset providing target samples at full resolution.
        n_samples: Number of time samples to use (-1 for all).
        ordered: If True, take first n_samples; if False, randomly permute.
        device: Torch device to move tensors to.

    Returns:
        Tuple of:
            - x: Dict of branch inputs and time tensor for trunk.
            - samples: Target output tensor of shape (batch, n_samples, features).
            - ts: Time points tensor of shape (batch, n_samples).
    """
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


