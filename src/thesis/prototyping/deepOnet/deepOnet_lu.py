"""Stochastic ODE DeepONet from Lu et al. (2021).

Recreates the stochastic ODE example from:
    Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadkis, G. E. (2021).
    Learning nonlinear operators via DeepONet based on the universal
    approximation theorem of operators.
    Nature Machine Intelligence, 3(3), 218-229.
    GitHub: https://github.com/lululxvi/deeponet

Stochastic ODE:
    dX_t = σ(t) X_t dW_t,  X_0 = 1

where σ(t) is a Gaussian random field with Matérn-1/2 (absolute exponential)
kernel and random length scale l ∈ [1, 2].

The operator maps the KL expansion of σ(·) to the statistical mean E[X_t].

Closed-form mean:
    E[X_t] = exp(σ²/2),  σ² = 2lt + 2l²(exp(-t/l) - 1)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from thesis.prototyping.deepOnet.utils import MLP, MLPConstructor


# ──────────────────────────────────────────────────────────────
# Data generation (ported from https://github.com/lululxvi/deeponet)
# ──────────────────────────────────────────────────────────────


def _eig(kernel, num, Nx):
    """Eigenvalues and eigenfunctions of a kernel on [0, 1].

    Uses the Nyström method with the trapezoidal rule.

    Args:
        kernel: sklearn kernel callable.
        num: Number of eigenvalues/functions to keep.
        Nx: Number of quadrature points.

    Returns:
        Tuple of (eigenvalues, eigenfunctions) arrays.
    """
    h = 1 / (Nx - 1)
    c = kernel(np.linspace(0, 1, num=Nx)[:, None])[0] * h
    A = np.empty((Nx, Nx))
    for i in range(Nx):
        A[i, i:] = c[: Nx - i]
        A[i, i::-1] = c[: i + 1]
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5

    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    for i in range(num):
        eigvec[:, i] /= np.trapezoid(eigvec[:, i] ** 2, dx=h) ** 0.5
    return eigval, eigvec


class _GRF_KL:
    """Gaussian Random Field via Karhunen-Loève expansion on [0, 1]."""

    def __init__(self, kernel, length_scale, num_eig, N, interp):
        from scipy import interpolate
        from sklearn.gaussian_process import kernels as gp_kernels

        if kernel == "AE":
            k = gp_kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "RBF":
            k = gp_kernels.RBF(length_scale=length_scale)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        eigval, eigvec = _eig(k, num_eig, N)
        eigvec *= eigval**0.5
        x = np.linspace(0, 1, num=N)
        self.eigfun = [
            interpolate.interp1d(x, y, kind=interp, copy=False, assume_sorted=True)
            for y in eigvec.T
        ]

    def bases(self, sensors):
        """Evaluate KL basis functions at sensor locations."""
        return np.array([np.ravel(f(sensors)) for f in self.eigfun])


class _GRFs:
    """Collection of GRFs with random length scales in [l_min, l_max]."""

    def __init__(self, kernel, l_min, l_max, N=100, interp="linear"):
        self.kernel = kernel
        self.l_min = l_min
        self.l_max = l_max
        self.N = N
        self.interp = interp

    def random(self, n):
        """Sample n random length scales."""
        return (self.l_max - self.l_min) * np.random.rand(n, 1) + self.l_min

    def eval_KL_bases(self, ls, sensors, M):
        """Evaluate KL bases for each length scale, flattened per sample."""
        results = []
        for l in ls:
            grf = _GRF_KL(self.kernel, l[0], M, self.N, self.interp)
            results.append(np.ravel(grf.bases(sensors)))
        return np.vstack(results)


class SODESystem:
    """Stochastic ODE: dX_t = σ(t) X_t dW_t, X_0 = y0.

    σ(t) is a GRF parameterized by length scale l.
    Learns the operator σ(·) → E[X_t].
    """

    def __init__(self, T: float = 1.0, y0: float = 1.0):
        self.T = T
        self.y0 = y0

    def _eval_mean(self, l, t):
        """Closed-form mean E[X_t] for GRF with length scale l.

        σ² = 2lt + 2l²(exp(-t/l) - 1)
        E[X_t] = y0 * exp(σ²/2)
        """
        sigma2 = 2 * l * t + 2 * l**2 * (np.exp(-t / l) - 1)
        return self.y0 * np.exp(0.5 * sigma2)

    def gen_operator_data(self, space, Nx, M, num):
        """Generate operator dataset.

        Each sample: (KL_bases, t) → E[X_t]

        Args:
            space: GRFs instance for sampling.
            Nx: Number of sensor locations.
            M: Number of KL modes.
            num: Number of samples to generate.

        Returns:
            branch_input: (num, Nx*M) KL bases.
            trunk_input: (num, 1) random time values.
            targets: (num, 1) exact mean E[X_t].
        """
        print(f"Generating operator data (n={num})...", flush=True)
        features = space.random(num)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        branch_input = space.eval_KL_bases(features, sensors, M)
        trunk_input = self.T * np.random.rand(num)[:, None]
        targets = self._eval_mean(features, trunk_input)
        return branch_input, trunk_input, targets

    def gen_example_data(self, space, l, Nx, M, num=100):
        """Generate example data for a fixed length scale.

        Returns evenly-spaced time points for plotting.
        """
        features = np.full((num, 1), l)
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        branch_input = space.eval_KL_bases(features, sensors, M)
        trunk_input = np.linspace(0, self.T, num=num)[:, None]
        targets = self._eval_mean(features, trunk_input)
        return branch_input, trunk_input, targets


# ──────────────────────────────────────────────────────────────
# DeepONet model
# ──────────────────────────────────────────────────────────────


class DeepONet(nn.Module):
    """Unstacked DeepONet with dot-product output.

    Architecture from Lu et al. (2021):
        Branch net: [m, 100, 100] with relu activation
        Trunk net:  [dim_x, 100, 100] with relu activation
        Output:     Σ_k branch_k * trunk_k + bias
    """

    def __init__(
        self,
        branch_cfg: MLPConstructor,
        trunk_cfg: MLPConstructor,
        use_bias: bool = True,
    ):
        super().__init__()
        if branch_cfg.layer_sizes[-1] != trunk_cfg.layer_sizes[-1]:
            raise ValueError(
                f"Branch ({branch_cfg.layer_sizes[-1]}) and trunk "
                f"({trunk_cfg.layer_sizes[-1]}) output dims must match"
            )
        self.branch = MLP(branch_cfg)
        self.trunk = MLP(trunk_cfg)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None
        self._init_weights()

    def _init_weights(self):
        """Xavier normal initialization (Glorot normal) to match the paper."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_branch: torch.Tensor, x_trunk: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_branch: (batch, m) branch input.
            x_trunk: (batch, dim_x) trunk input.

        Returns:
            (batch, 1) predicted output.
        """
        b = self.branch(x_branch)
        t = self.trunk(x_trunk)
        out = torch.sum(b * t, dim=-1, keepdim=True)
        if self.bias is not None:
            out = out + self.bias
        return out


# ──────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────


def _to_tensors(*arrays, device):
    """Convert numpy arrays to float32 tensors on device."""
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, xt, y in dataloader:
        xb, xt, y = xb.to(device), xt.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb, xt), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, xt, y in dataloader:
        xb, xt, y = xb.to(device), xt.to(device), y.to(device)
        loss = loss_fn(model(xb, xt), y)
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def predict(model, branch_input, trunk_input, device):
    model.eval()
    xb, xt = _to_tensors(branch_input, trunk_input, device=device)
    return model(xb, xt).cpu().numpy()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Paper parameters ──
    # SODE: dX_t = σ(t) X_t dW_t, X_0 = 1
    # σ drawn from GRF with AE (Matérn-1/2) kernel, length scale l ∈ [1, 2]
    Nx = 20  # sensor locations for KL bases
    M = 5  # KL modes
    m = Nx * M  # branch input dim = 100
    p = 100  # latent dim (shared branch/trunk output)
    dim_x = 1  # trunk input dim (time)
    n_train = 1_000_000
    n_test = 1_000_000
    lr = 0.001
    epochs = 20_000
    batch_size = 0  # 0 = full batch (paper default)

    print(f"m={m}, p={p}, Nx={Nx}, M={M}")
    print(f"Train: {n_train}, Test: {n_test}, Epochs: {epochs}")

    # ── Data generation ──
    system = SODESystem(T=1, y0=1)
    space = _GRFs(kernel="AE", l_min=1, l_max=2, N=100, interp="linear")

    xb_train, xt_train, y_train = system.gen_operator_data(space, Nx, M, n_train)
    xb_test, xt_test, y_test = system.gen_operator_data(space, Nx, M, n_test)
    print(f"Branch input shape: {xb_train.shape}")
    print(f"Trunk  input shape: {xt_train.shape}")
    print(f"Output shape:       {y_train.shape}")

    # ── DataLoaders ──
    train_ds = TensorDataset(
        torch.tensor(xb_train, dtype=torch.float32),
        torch.tensor(xt_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(xb_test, dtype=torch.float32),
        torch.tensor(xt_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    bs = len(train_ds) if batch_size == 0 else batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))

    # ── Model (matches paper architecture) ──
    branch_cfg = MLPConstructor(layer_sizes=[m, p, p], activation="relu")
    trunk_cfg = MLPConstructor(layer_sizes=[dim_x, p, p], activation="relu")
    model = DeepONet(branch_cfg, trunk_cfg, use_bias=True).to(device)
    n_params = sum(par.numel() for par in model.parameters())
    print(f"\nModel ({n_params:,} parameters):\n{model}")

    # ── Training ──
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_test_loss = float("inf")
    best_epoch = 0

    print(f"\nTraining for {epochs} epochs, lr={lr}")
    print("=" * 60)

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        if epoch % 1000 == 0 or epoch == 1:
            test_loss = evaluate(model, test_loader, loss_fn, device)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
            tqdm.tqdm.write(
                f"Epoch {epoch:>5d} | Train MSE: {train_loss:.6e} | "
                f"Test MSE: {test_loss:.6e}"
            )

    print(f"\nBest model at epoch {best_epoch}: Test MSE = {best_test_loss:.6e}")

    # ── Example evaluations ──
    print("\n" + "=" * 60)
    print("Example predictions for different length scales:")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(10):
        l = 0.2 + 0.2 * i
        xb_ex, xt_ex, y_ex = system.gen_example_data(space, l, Nx, M)
        y_pred = predict(model, xb_ex, xt_ex, device)
        mse = np.mean((y_ex - y_pred) ** 2)
        mse_outlier = np.mean(
            np.sort((y_ex - y_pred).ravel() ** 2)[: -max(1, len(y_ex) // 1000)]
        )
        print(f"  l={l:.1f}  MSE={mse:.6e}  MSE(w/o outliers)={mse_outlier:.6e}")

        ax = axes[i // 5, i % 5]
        t_plot = xt_ex.ravel()
        ax.plot(t_plot, y_ex.ravel(), "k-", linewidth=2, label="Exact")
        ax.plot(t_plot, y_pred.ravel(), "r--", linewidth=2, label="DeepONet")
        ax.set_title(f"l = {l:.1f}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\mathbb{E}[X_t]$")
        if i == 0:
            ax.legend()

    plt.suptitle(
        r"DeepONet: $\sigma(\cdot) \mapsto \mathbb{E}[X_t]$ for stochastic ODE",
        fontsize=14,
    )
    plt.tight_layout()
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "deeponet_sode_lu.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {out_dir / 'deeponet_sode_lu.png'}")
