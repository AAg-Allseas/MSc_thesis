

from cProfile import Profile
from pathlib import Path
import pstats

import numpy as np
import torch
import torchsde

from src.prototyping.latentSDE.model_latentSDE import LatentSDE

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


def integrate_sde(sde: LatentSDE, ts: torch.Tensor, bm: torch.Tensor, z0: torch.Tensor) -> None:
    zs = torch.zeros_like(bm, device=bm.device)
    zs[0] = z0
    with torch.no_grad():
        for i in range(1, ts.shape[0]):
            zs[i] = sde.h(ts[i], zs[i-1]) * ts[i] + sde.g(ts[i], zs[i-1]) * bm[i]
    return zs

if __name__ == "__main__":
    path = Path("dump/latentsde/proof_of_concept/latentsde_491.pth")

    latent_size = 4
    context_size = 64
    hidden_size = 128

    batch_size = 50
    dt = 0.05
    T = 10800
    series_length = int(T // dt + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bm = torch.rand(size=(series_length, batch_size, latent_size), device=device) * dt

    latent_sde = LatentSDE(
        data_size=12,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ).to(device)
    
    latent_sde.load_state_dict(torch.load(path))
    latent_sde.eval()

    ts = np.arange(0, T, dt)
    ts = torch.from_numpy(ts).to(device)

    eps = torch.randn(size=(batch_size, *latent_sde.pz0_mean.shape[1:]), device=latent_sde.pz0_mean.device)
    z0 = latent_sde.pz0_mean + latent_sde.pz0_logstd.exp() * eps

    with Profile() as profile:
        integrate_sde(latent_sde, ts, bm, z0)
    
        profile.dump_stats(Path("dump/latentsde/profile/speed_latentsde.stats"))
        pstats.Stats(profile) \
                    .strip_dirs()\
                    .sort_stats("cumulative") \
                    .print_stats(30)


