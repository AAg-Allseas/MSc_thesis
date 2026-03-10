from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path

from src.visualisation.pilot_tests.plot_deepOnet import main as plot_DeepONet
from src.visualisation.pilot_tests.plot_latent_sde import main as plot_LatentSDE
from src.visualisation.pilot_tests.timestep_difference import (
    plot_cumulative_error_summary,
)
import src.visualisation.pilot_tests.plot_losses as pilot_losses
import src.visualisation.general_plotting.weather_report as weather_report

PLOTS = Path("plots")


def save_plot(name: str, fig: plt.Figure) -> None:
    fig.savefig(PLOTS / name)
    print(f"Saved {name}")


if __name__ == "__main__":
    matplotlib.style.use(r"src\visualisation\thesis.mplstyle")
    save_plot("pilot_deepOnet_1dof", plot_DeepONet())
    save_plot("pilot_latentSDE", plot_LatentSDE())
    plot_cumulative_error_summary()

    save_plot("pilot_losses_latentsde", pilot_losses.plotLatentSDE())
    save_plot("pilot_losses_deepOnet", pilot_losses.plotDeepONet())
    weather_report.main()
