
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np

from thesis.prototyping.data_handling import find_parquet_files
from thesis.prototyping.dataloader import ParquetDataset


def plot_timetraces(dataset: ParquetDataset, 
                    color: Optional[str]=None,
                    figsize: Optional[tuple]=None,
                    key: Optional[str]=None) -> None:
    samples = []
    for i in range(len(dataset)):
        times, sample, _ = dataset[i]
        samples.append(sample)
    samples = np.array(samples).swapaxes(0, 1)


    fig, ax = plt.subplots(figsize=figsize)

    # Surge
    legend_handlers = ax.plot(times, samples[..., 0], alpha=0.1, color=color)
    
    if key is not None:
        ax.legend([legend_handlers[0]], [key])

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.set_title("Surge position")

    return fig, ax


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
                            standardise=False)


    plot_timetraces(dataset)

    plt.show()