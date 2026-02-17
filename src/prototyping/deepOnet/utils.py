""" Helper functions and classes taken from"
Moya, C., Zhang, S., Lin, G., & Yue, M. (2023). 
DeepONet-grid-UQ: A trustworthy deep operator framework for predicting the power grid's post-fault trajectories. 
Neurocomputing, 535, 166-182. https://doi.org/10.1016/j.neucom.2023.03.015
 """
from dataclasses import dataclass
from typing import Any
from torch import nn
import torch

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
    """Configuration for a named branch network."""
    name: str


class MLP(nn.Module):
    """Simple MLP assembled from a constructor configuration.

    Args:
        constructor: MLPConstructor describing sizes and activation.
    """
    def __init__(self, constructor: MLPConstructor) -> None:

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
        """Run a forward pass through the MLP."""
        y = x
        for k in range(len(self.net)):
            y = self.net[k](y)
        return y


