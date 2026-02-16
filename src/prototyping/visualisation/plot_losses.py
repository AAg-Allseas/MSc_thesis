
from matplotlib import pyplot as plt
import numpy as np



def plotLatentSDE(likelihood: np.ndarray, kl_loss: np.ndarray) -> None:
    """ 
    Function used to create a plot of the training losses of the LatentSDE approach
    Creates both a regular and log scaled plot.

    Args:
        likelihood np.ndarray: [Time, batch] arranged array containing the (positive) log-likelihood losses
        kl_loss np.ndarray: [Time, batch] arranged array containing the KL losses
    """
    fig, axs = plt.subplots(ncols=1, nrows=2)

    for i in range(2):
        axs[i].plot(likelihood, color="blue", alpha=0.1)
        axs[i].plot(np.mean(likelihood, axis=1), color="blue", label="Negative Log-Likelihood")
        axs[i].fill_between(np.linspace(0, len(likelihood), len(likelihood)), np.mean(likelihood, axis=1), color="blue", alpha=0.1)
        kl_loss_stacked = kl_loss + likelihood
        axs[i].plot(kl_loss_stacked, color="orange", alpha=0.1)
        axs[i].plot(np.mean(kl_loss_stacked, axis=1), color="orange", label="KL-Divergence loss")
        axs[i].fill_between(np.linspace(0, len(kl_loss_stacked), len(kl_loss_stacked)), np.mean(likelihood, axis=1), np.mean(kl_loss_stacked, axis=1), color="orange", alpha=0.1)

    axs[1].set_yscale("log")



if __name__ == "__main__":
    test_kl = np.zeros([1000, 5])
    test_like = np.zeros([1000, 5])

    def kl_func(x: float):
        return 100* x ** (-0.003)
    def like_func(x: float): 
        return 100 * np.exp(-x * 0.003)

    rng = np.random.default_rng()
    test_kl[0] = 100
    test_like[0] = 100
    for i in range(1, 1000):
        test_kl[i] = np.clip(kl_func(i) + rng.normal(loc=0, scale=test_kl[i-1]/20, size=5), 0, a_max=200)
        test_like[i] = np.clip(like_func(i) + rng.normal(loc=0, scale=test_like[i-1]/20, size=5), 0, a_max=200)
    
    plotLatentSDE(test_like, test_kl)
    plt.show()