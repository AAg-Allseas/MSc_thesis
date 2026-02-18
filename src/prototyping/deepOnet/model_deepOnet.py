"""DeepONet based on:
Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). 
Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. 
Nature Machine Intelligence, 3(3), 218-229. https://doi.org/10.1038/s42256-021-00302-5

Using the multiple input formulation from:
Jin, P., Meng, S., & Lu, L. (2022).
 MIONet: Learning Multiple-Input Operators via Tensor Product. 
 SIAM Journal on Scientific Computing, 44(6), A3490-A3514. https://doi.org/10.1137/22M1477751
"""
from typing import Dict, List, Tuple

from torch import nn
import torch


from src.prototyping.deepOnet.utils import CNN1D, CNN1DBranchConstructor, MLP, BranchConstructor, MLPConstructor


class MIONet(nn.Module):
    """Multiple-input operator network with branch and trunk MLPs."""

    def __init__(
        self,
        branches: list,
        trunk: MLPConstructor,
        output_dim: int,
        use_bias: bool = True,
    ) -> None:
        """Build the branch and trunk subnetworks.

        Args:
            branches: Branch constructors (BranchConstructor or CNN1DBranchConstructor).
            trunk: Trunk constructor for the shared input.
            use_bias: Whether to include a learned scalar bias.
        """
        super().__init__()
        latent_dim = trunk.layer_sizes[-1]
        self.branch_names = []

        for branch in branches:
            if isinstance(branch, CNN1DBranchConstructor):
                if branch.output_dim != latent_dim:
                    raise AttributeError(
                        f"CNN branch '{branch.name}' output_dim={branch.output_dim} "
                        f"must match trunk latent dim={latent_dim}"
                    )
                setattr(self, branch.name, CNN1D(branch))
            else:
                if branch.layer_sizes[-1] != latent_dim:
                    raise AttributeError(
                        f"All branches must output {latent_dim} dimensions "
                        f"(trunk's penultimate layer), got {branch.layer_sizes[-1]}"
                    )
                setattr(self, branch.name, MLP(branch))
            self.branch_names.append(branch.name)
        
        self.trunk = MLP(trunk)
        # Output MLP takes concatenated branch + trunk representations
        self.output = MLP(MLPConstructor(
            layer_sizes=[latent_dim * 2, latent_dim, output_dim],
            activation="gelu"
        ))

    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Compute the MIONet output for a batch.

        Args:
            x: Tuple of branch input dict and trunk input tensor.
               Trunk input y can be (batch,), (batch, 1), or (batch, n_samples).

        Returns:
            Predicted output tensor of shape (batch, output_dim) or (batch, n_samples, output_dim).
        """
        u, y = x
        
        # Ensure y has trailing dimension of 1 for the trunk input
        if y.dim() == 1:
            y = y.unsqueeze(-1)  # (batch,) -> (batch, 1)
        elif y.dim() == 2 and y.size(-1) != 1:
            y = y.unsqueeze(-1)  # (batch, n_samples) -> (batch, n_samples, 1)
        
        bs: List[torch.Tensor] = []
        for name in self.branch_names:
            bs.append(getattr(self, name)(u[name]))
        
        T = self.trunk(y)  # (batch, latent_dim) or (batch, n_samples, latent_dim)
        B = torch.stack(bs)  # (n_branches, batch, latent_dim)

        # If T has n_samples dimension, expand B to broadcast
        if T.dim() == 3:
            B = B.unsqueeze(2)  # (n_branches, batch, 1, latent_dim)

        # Sum branch outputs, then concatenate with trunk
        B_combined = torch.sum(B, dim=0)  # (batch, [n_samples,] latent_dim)
        s = torch.cat([B_combined.expand_as(T), T], dim=-1)  # (batch, [n_samples,] 2*latent_dim)

        return self.output(s)