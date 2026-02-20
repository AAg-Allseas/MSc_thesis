
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
from src.prototyping.visualisation.plot_batch_timeseries import plot_timetraces


def plot_latentsde(dataset: ParquetDataset, 
                   sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    fig, ax = plot_timetraces(dataset, color="b")
    data, ts = sample_data

    ax.plot(ts, data[..., 0], alpha=0.4, color="r")

    plt.show()

if __name__ == "__main__":
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
    
    # dataset = ParquetDataset(files, 
    #                         columns=feats,
    #                         sample_length=5000,
    #                         scale_factors=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    
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
    device = torch.device("cuda")
    latent_sde = LatentSDE(
        data_size=12,
        latent_size=4,
        context_size=64,
        hidden_size=128,
        ).to(device)
    dataset = ParquetDataset(files, 
                            columns=feats,
                            sample_length=5,
                            )
    dataloader = DataLoader(dataset)
    ts, xs, _ = prep_batch(list(dataloader)[0])
    print(torch.export.export(latent_sde, args=(xs, ts)))