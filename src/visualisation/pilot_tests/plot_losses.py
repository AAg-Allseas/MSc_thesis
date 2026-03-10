from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.visualisation.general_plotting.config import GOLDEN_RATIO


def plotLatentSDE() -> None:
    """
    Function used to create a plot of the training losses of the LatentSDE approach
    Creates both a regular and log scaled plot.

    Args:
        likelihood np.ndarray: [Time, batch] arranged array containing the (positive) log-likelihood losses
        kl_loss np.ndarray: [Time, batch] arranged array containing the KL losses
    """
    likelihood = np.load(Path("dump/latentsde/proof_of_concept/likelihood_losses.npy"))[
        :, 0
    ].reshape(-1, 1)
    kl_loss = np.load(Path("dump/latentsde/proof_of_concept/kl_losses.npy"))[
        :, 0
    ].reshape(-1, 1)
    fig, ax = plt.subplots(figsize=GOLDEN_RATIO * 0.75)
    likelihood = likelihood[:491]
    kl_loss = kl_loss[:491]

    ax.plot(np.mean(likelihood, axis=1), label="Negative Log-Likelihood")
    ax.plot(np.mean(kl_loss, axis=1), label="KL-Divergence loss")
    ax.legend()

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")

    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(linewidth=0.25)
    return fig


def plotDeepONet() -> None:
    """
    Function used to create a plot of the training losses of the DeepONet approach
    Creates both a regular and log scaled plot.

    Args:
        mse np.ndarray: [Time] arranged array containing the MSE losses per epoch
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=GOLDEN_RATIO * 0.75)
    data = pd.read_csv(Path(r"dump\deeponet\epoch_test_loss.csv"))

    # First 100 steps to see beginning
    mse = data["value"].to_numpy()[:100]
    epoch = data["step"].to_numpy()[:100]

    ax.plot(epoch, mse)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(linewidth=0.25)

    ax.set_yscale("log")
    return fig


if __name__ == "__main__":
    plotLatentSDE()
    plotDeepONet()
    plt.show()
