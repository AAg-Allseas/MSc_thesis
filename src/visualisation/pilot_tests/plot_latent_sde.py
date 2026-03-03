
from cProfile import Profile
from pathlib import Path
import pstats

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset, prep_batch
from src.prototyping.latentSDE.model_latentSDE import LatentSDE
from src.prototyping.latentSDE.test_latentSDE import load_and_sample
from src.visualisation.general_plotting.config import FLATTER
from src.visualisation.pilot_tests.plot_batch_timeseries import plot_timetraces


def plot_latentsde(dataset: ParquetDataset, 
                   sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    fig, ax = plot_timetraces(dataset, color="b", figsize=FLATTER, key="Training paths")
    data, ts = sample_data

    handles_gen = ax.plot(ts, data[..., 0], alpha=0.6, color="r")

    leg = ax.get_legend()
    handles, labels = [], []
    if leg is not None:
        handles = list(leg.legend_handles)
        labels = [t.get_text() for t in leg.texts]
    handles.append(handles_gen[0])
    labels.append("Generated paths")
    ax.legend(handles, labels)

    return fig

def main() -> None:
    sample_ts = np.load("dump/latentsde/sample_time.npy")
    sample_data = np.load("dump/latentsde/sample_data.npy")

    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 2)
    
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
                            sample_length=5000,
                            standardise=False)
    return plot_latentsde(dataset, (sample_data, sample_ts))

if __name__ == "__main__":
    # files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
    #                            lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 2)
    
    # feats = ['pos_eta_x', 
    #         'pos_eta_y', 
    #         'pos_eta_mz', 
    #         'pos_nu_x', 
    #         'pos_nu_y', 
    #         'pos_nu_mz', 
    #         'rpm_bow_fore', 
    #         'rpm_bow_aft', 
    #         'rpm_stern_fore', 
    #         'rpm_stern_aft', 
    #         'rpm_fixed_ps', 
    #         'rpm_fixed_sb']
    
    # dataset = ParquetDataset(files, 
    #                         columns=feats,
    #                         sample_length=5000,
    #                         standardise=True)
    
    # print("Starting Sampling")
    # with Profile() as profile:
    #     sample_data = load_and_sample(Path("dump/latentsde/proof_of_concept/latentsde_491.pth"),
    #                                     batch_size=5,
    #                                     data_size=12,
    #                                     latent_size=4,
    #                                     context_size=64,
    #                                     hidden_size=128,
    #                                     timestep=0.05,
    #                                     end_time=250)
    #     profile.dump_stats(Path("dump/latentsde/profile/sample.stats"))
    #     pstats.Stats(profile) \
    #                 .strip_dirs()\
    #                 .sort_stats("cumulative") \
    #                 .print_stats(30)
        
    # print("Finished Sampling")
    # data, ts = sample_data
    # np.save("dump/latentsde/sample_data", data)
    # np.save("dump/latentsde/sample_time", ts)
    # plot_latentsde(dataset, sample_data)

    main()
    plt.show()