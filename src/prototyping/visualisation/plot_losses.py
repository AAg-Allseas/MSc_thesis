
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np



def plotLatentSDE(likelihood: np.ndarray, kl_loss: np.ndarray, start: int=0, end: int=-1) -> None:
    """ 
    Function used to create a plot of the training losses of the LatentSDE approach
    Creates both a regular and log scaled plot.

    Args:
        likelihood np.ndarray: [Time, batch] arranged array containing the (positive) log-likelihood losses
        kl_loss np.ndarray: [Time, batch] arranged array containing the KL losses
    """
    fig, axs = plt.subplots(ncols=1, nrows=2)
    likelihood = likelihood[start:end]
    kl_loss = kl_loss[start:end]

    for i in range(2):
        axs[i].plot(likelihood[:end], color="blue", alpha=0.1)
        axs[i].plot(np.mean(likelihood, axis=1), color="blue", label="Negative Log-Likelihood")
        axs[i].fill_between(np.linspace(0, len(likelihood), len(likelihood)), np.mean(likelihood, axis=1), color="blue", alpha=0.1)
        kl_loss_stacked = kl_loss + likelihood
        axs[i].plot(kl_loss_stacked, color="orange", alpha=0.1)
        axs[i].plot(np.mean(kl_loss_stacked, axis=1), color="orange", label="KL-Divergence loss")
        axs[i].fill_between(np.linspace(0, len(kl_loss_stacked), len(kl_loss_stacked)), np.mean(likelihood, axis=1), np.mean(kl_loss_stacked, axis=1), color="orange", alpha=0.1)
        axs[i].legend()

    axs[1].set_yscale("symlog", linthresh=0.1)



if __name__ == "__main__":
    likelihood_losses =  np.load(Path("dump/latentsde/likelihood_losses.npy"))[:, 0].reshape(-1, 1)
    kl_losses = np.load(Path("dump/latentsde/kl_losses.npy"))[:, 0].reshape(-1, 1)
    plotLatentSDE(likelihood_losses, kl_losses, end=491)
    plt.show()