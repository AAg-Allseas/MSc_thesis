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


from thesis.prototyping.deepOnet.utils import CNN1D, CNN1DBranchConstructor, MLP, BranchConstructor, MLPConstructor


class MIONet(nn.Module):
    """Multiple-input operator network with branch and trunk MLPs.

    Each output degree-of-freedom gets its own trunk network that shares the
    same query-point input.  The forward pass computes, for each output j:

        G_j(u_1, …, u_m)(y) = Σ_k (∏_i b_i^k(u_i)) · t_j^k(y)

    following the original MIONet formulation (Jin et al., 2022).
    """

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
            trunk: Trunk constructor for the shared input.  One trunk MLP is
                   created per output DOF, all sharing the same architecture.
            output_dim: Number of output degrees-of-freedom (one trunk each).
            use_bias: Whether to include a learned scalar bias per output DOF.
        """
        super().__init__()
        latent_dim = trunk.layer_sizes[-1]
        self.branch_names: List[str] = []
        self.output_dim = output_dim

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

        # One trunk per output DOF — same architecture, independent weights
        self.trunks = nn.ModuleList([MLP(trunk) for _ in range(output_dim)])

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Compute the MIONet output for a batch.

        Args:
            x: Tuple of (branch_inputs, trunk_input).
               branch_inputs: dict mapping branch name -> tensor.
               trunk_input y: (batch,), (batch, 1), or (batch, n_samples).

        Returns:
            Predicted output tensor of shape (batch, output_dim) or
            (batch, n_samples, output_dim).
        """
        u, y = x

        # Ensure y has a trailing feature dim of 1 for the trunk MLP
        if y.dim() == 1:
            y = y.unsqueeze(-1)           # (batch,) -> (batch, 1)
        elif y.dim() == 2 and y.size(-1) != 1:
            y = y.unsqueeze(-1)           # (batch, n_samples) -> (batch, n_samples, 1)

        # --- Branches: element-wise product across all branch outputs ---
        # Each branch output: (batch, latent_dim)
        B = torch.ones_like(getattr(self, self.branch_names[0])(u[self.branch_names[0]]))
        for name in self.branch_names:
            B = B * getattr(self, name)(u[name])  # (batch, latent_dim)

        # --- Trunks: one per output DOF, dot-product with combined branch ---
        # T_j(y): (batch, latent_dim) or (batch, n_samples, latent_dim)
        outputs = []
        for trunk in self.trunks:
            T = trunk(y)
            if T.dim() == 2 and B.dim() == 2:
                # (batch, latent_dim) * (batch, latent_dim) -> sum -> (batch,)
                outputs.append((B * T).sum(dim=-1))
            else:
                # B: (batch, latent_dim) -> (batch, 1, latent_dim)
                B_exp = B.unsqueeze(1) if B.dim() == 2 else B
                outputs.append((B_exp * T).sum(dim=-1))  # (batch, n_samples)

        # Stack along last dim: (batch, output_dim) or (batch, n_samples, output_dim)
        out = torch.stack(outputs, dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out