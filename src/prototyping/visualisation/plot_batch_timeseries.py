
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset


def plot_timetraces(series: np.ndarray | Tensor, timesteps: np.ndarray | Tensor) -> None:

    fig, ax = plt.subplots()

    # Surge
    ax.plot(timesteps, series[..., 0], alpha=0.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.set_title("Surge position")
    
    plt.show()


if __name__ == "__main__":
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05)
    
    feats = ['pos_eta_x', 
            'pos_eta_y', 
            'pos_eta_mz', 
            'pos_nu_x', 
            'pos_nu_y', 
            'pos_nu_mz', 
            'rpm_bow_fore', 
            'rpm_bow_aft', 
            'rpm_stern_fore', 
            'rpm_stern_aft', 
            'rpm_fixed_ps', 
            'rpm_fixed_sb']
    
    dataset = ParquetDataset(files, 
                            columns=feats,
                            scale_factors=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    samples = []
    for i in range(len(dataset)):
        times, sample = dataset[i]
        samples.append(sample)
    samples = np.array(samples).swapaxes(0, 1)
    plot_timetraces(samples, times)
