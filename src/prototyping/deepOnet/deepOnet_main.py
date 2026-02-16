"""DeepONet based on:
Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). 
Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. 
Nature Machine Intelligence, 3(3), 218-229. https://doi.org/10.1038/s42256-021-00302-5

Using the multiple input formulation from:
Jin, P., Meng, S., & Lu, L. (2022).
 MIONet: Learning Multiple-Input Operators via Tensor Product. 
 SIAM Journal on Scientific Computing, 44(6), A3490-A3514. https://doi.org/10.1137/22M1477751
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from torch import nn
import torch

from src.prototyping.deepOnet.utils import MLP, MLPConstructor

@dataclass
class BranchConstructor(MLPConstructor):
    """Configuration for a named branch network."""

    name: str


class MIONet(nn.Module):
    """Multiple-input operator network with branch and trunk MLPs."""

    def __init__(
        self,
        branches: List[BranchConstructor],
        trunk: MLPConstructor,
        use_bias: bool = True,
    ) -> None:
        """Build the branch and trunk subnetworks.

        Args:
            branches: Branch constructors, one per input stream.
            trunk: Trunk constructor for the shared input.
            use_bias: Whether to include a learned scalar bias.
        """
        super().__init__()

        for branch in branches:
            if branch.layer_size[-1] != trunk.layer_size[-1]:
                raise AttributeError("Branches and trunk networks must have same output dimension")
            
            setattr(self, branch.name, MLP(branch))
        
        self.trunk = MLP(trunk)
        self.use_bias = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Compute the MIONet output for a batch.

        Args:
            x: Tuple of branch input dict and trunk input tensor.

        Returns:
            Predicted output tensor.
        """
        u, y = x
        bs: List[torch.Tensor] = []
        for key, value in u.items():
            bs.append(getattr(self, key)(value))
        T = self.trunk(y)
        B = torch.stack(bs)

        # Combine branch outputs with trunk output using a tensor product.
        s = torch.prod(B * T.unsqueeze(0), dim=0)
        s = torch.sum(s, dim=-1)

        if self.use_bias:
            s = s + self.tau
        return s

if __name__ == "__main__":
    branches = [BranchConstructor(name="inital_conditions",layer_size=[12, 100, 100], activation="gelu"),
                BranchConstructor(name="measurements",layer_size=[1000, 250, 250, 100], activation="gelu")]
    trunk = MLPConstructor(layer_size=[1000, 250, 250, 100], activation="gelu")
        
    mionet = MIONet(branches, trunk)

