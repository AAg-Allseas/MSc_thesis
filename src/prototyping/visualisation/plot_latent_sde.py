
from cProfile import Profile
from pathlib import Path
import pstats

from matplotlib import pyplot as plt
import numpy as np
import torch
import torchsde

from src.prototyping.data_handling import find_parquet_files
from src.prototyping.dataloader import ParquetDataset
from src.prototyping.latentSDE.latentSDE_main import LatentSDE
from src.prototyping.visualisation.plot_batch_timeseries import plot_timetraces


def load_and_sample(path: Path, 
                    batch_size: int, 
                    data_size: int, 
                    latent_size: int, 
                    context_size: int, 
                    hidden_size: int,
                    timestep: float,
                    start_time: float = 0,
                    end_time: float = 10800) -> np.ndarray:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    
    latent_sde.load_state_dict(torch.load(path))
    latent_sde.eval()
    
    ts = np.arange(start_time, end_time, timestep)
    ts = torch.from_numpy(ts).to(device)
    
    bm = torchsde.BrownianInterval(start_time, end_time, size=(batch_size, latent_size), dt=timestep, device=device)
    return latent_sde.sample(batch_size=batch_size, ts=ts, bm=bm).to("cpu"), ts.to("cpu")

def plot_latentsde(dataset: ParquetDataset, 
                   sample_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    fig, ax = plot_timetraces(dataset, color="b")
    data, ts = sample_data

    ax.plot(ts, data[..., 0], alpha=0.4, color="r")

    plt.show()

if __name__ == "__main__":
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                               lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 1)
    
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
                            scale_factors=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    
    print("Starting Sampling")
    with Profile() as profile:
        sample_data = load_and_sample(Path("dump/latentsde/proof_of_concept/latentsde_491.pth"),
                                        batch_size=5,
                                        data_size=12,
                                        latent_size=4,
                                        context_size=64,
                                        hidden_size=128,
                                        timestep=0.05,
                                        end_time=250)
        profile.dump_stats(Path("dump/latentsde/profile/sample.stats"))
        pstats.Stats(profile) \
                    .strip_dirs()\
                    .sort_stats("cumulative") \
                    .print_stats(30)
        
    print("Finished Sampling")
    data, ts = sample_data
    np.save("dump/latentsde/sample_data", data)
    np.save("dump/latentsde/sample_time", ts)
    plot_latentsde(dataset, sample_data)