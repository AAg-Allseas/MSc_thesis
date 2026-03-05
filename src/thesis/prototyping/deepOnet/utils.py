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

from thesis.prototyping.dataloader import ParquetDataset

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
            "sin": sin_act(),
            "Mish": nn.Mish(),
            "identity": nn.Identity(),
    }[identifier]

# sin activation function
class sin_act(nn.Module):
    def __init__(self):
        super(sin_act, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


@dataclass
class MLPConstructor:
    """Configuration container for the MLP builder.

    Attributes:
        layer_sizes: List of layer sizes, including input and output.
        activation: Activation name used between linear layers.
        dropout: Dropout probability applied after each hidden activation (0.0 = no dropout).
    """
    layer_sizes: list[int]
    activation: str
    dropout: float = 0.0


@dataclass
class BranchConstructor:
    """Configuration for a named branch network in MIONet.

    Attributes:
        name: Unique identifier for this branch (used as model attribute name).
        layer_sizes: List of layer sizes, including input and output.
        activation: Activation name used between linear layers.
        dropout: Dropout probability applied after each hidden activation (0.0 = no dropout).
    """
    name: str
    layer_sizes: list[int]
    activation: str
    dropout: float = 0.0


@dataclass
class CNN1DBranchConstructor:
    """Configuration for a 1D CNN branch network.

    Attributes:
        name: Unique identifier for this branch.
        in_channels: Number of input channels (1 for univariate series).
        channels: List of output channels per conv layer.
        kernel_sizes: List of kernel sizes per conv layer.
        output_dim: Final output dimension (latent dim).
        activation: Activation name used between layers.
        dropout: Dropout probability applied after each activation (0.0 = no dropout).
    """
    name: str
    in_channels: int
    channels: list[int]
    kernel_sizes: list[int]
    output_dim: int
    activation: str = "gelu"
    dropout: float = 0.0


class CNN1D(nn.Module):
    """1D CNN that maps a time series to a fixed-size latent vector."""

    def __init__(self, constructor: CNN1DBranchConstructor) -> None:
        super().__init__()
        layers = []
        in_ch = constructor.in_channels
        for out_ch, ks in zip(constructor.channels, constructor.kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2))
            layers.append(get_activation(constructor.activation))
            if constructor.dropout > 0:
                layers.append(nn.Dropout(constructor.dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, constructor.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels[-1])
        return self.fc(x)


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
            if constructor.dropout > 0:
                self.net.append(nn.Dropout(constructor.dropout))

        self.net.append(nn.Linear(constructor.layer_sizes[-2], constructor.layer_sizes[-1], bias=True))

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
    input_features: dict[str, int],
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
        input_features: dictionary of strings defining the names of the input features and their corresponding index in the state vector. 
                        Initial conditions are added automatically.
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

    # Initial conditions from the first timestep of each sample (first 3 features = positions)
    initial_conditions = samples[:, 0, :].to(device, dtype=torch.float32)

    # Compute fixed t_max from the full time vector before subsampling
    t_max = ts.max()

    if n_samples == -1:
        n_samples = ts.shape[-1]

    if ordered:
        idx_0 = torch.randint(0, ts.shape[-1] - n_samples + 1, size=(1,)).item()
        ts = ts[..., idx_0 : idx_0 + n_samples].to(device)
        samples = samples[:, idx_0 : idx_0 + n_samples, :].to(device)
        
    else:
        idx = torch.randperm(ts.size(1))[:n_samples]
        ts = ts[:, idx].to(device)
        samples = samples[:, idx].to(device)

    # Normalize time to [0, 1]
    if t_max > 0:
        ts = ts / t_max
    input_dict = {"initial_conditions": initial_conditions}
    for feat, idx in input_features.items():
        input_dict[feat] = sensors[..., idx]
    x = (input_dict, ts)

    return x, samples, ts


