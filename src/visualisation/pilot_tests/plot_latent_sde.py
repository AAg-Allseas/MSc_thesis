from cProfile import Profile
from pathlib import Path
import pstats

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

from thesis.prototyping.data_handling import find_parquet_files
from thesis.prototyping.dataloader import ParquetDataset, prep_batch
from thesis.prototyping.latentSDE.test_latentSDE import load_and_sample
from src.visualisation.general_plotting.config import FLATTER
from src.visualisation.pilot_tests.plot_batch_timeseries import plot_timetraces


def plot_latentsde(
    dataset: ParquetDataset,
    sample_prior: tuple[torch.Tensor, torch.Tensor],
    sample_posterior: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    fig, ax = plot_timetraces(dataset, color="b", figsize=FLATTER, key="Training paths")

    leg = ax.get_legend()
    handles, labels = [], []
    if leg is not None:
        handles = list(leg.legend_handles)
        labels = [t.get_text() for t in leg.texts]

    data_prior, ts_prior = sample_prior
    handles_gen_prior = ax.plot(ts_prior, data_prior[..., 0], alpha=0.6, color="r")
    handles.append(handles_gen_prior[0])
    labels.append("Generated paths (Prior)")

    if sample_posterior is not None:
        data_posterior, ts_posterior = sample_posterior
        handles_gen_posterior = ax.plot(
            ts_posterior, data_posterior[..., 0], alpha=0.6, color="g"
        )
        handles.append(handles_gen_posterior[0])
        labels.append("Generated paths (Posterior)")

    ax.legend(handles, labels)

    return fig


def main() -> None:
    sample_ts = np.load("dump/latentsde/sample_time.npy")
    sample_data = np.load("dump/latentsde/sample_data.npy")

    files = find_parquet_files(
        Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
        lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 2,
    )

    feats = [
        "pos_eta_x",
        "pos_eta_y",
        "pos_eta_mz",
        "pos_nu_x",
        "pos_nu_y",
        "pos_nu_mz",
        "rpm_bow_fore",
        "rpm_bow_aft",
        "rpm_stern_fore",
        "rpm_stern_aft",
        "rpm_fixed_ps",
        "rpm_fixed_sb",
    ]

    dataset = ParquetDataset(
        files, columns=feats, sample_length=5000, standardise=False
    )
    return plot_latentsde(dataset, (sample_data, sample_ts))


if __name__ == "__main__":
    files = find_parquet_files(
        Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
        lambda m: m["end_time"] == 10800 and m["timestep"] == 0.05 and m["seed"] < 2,
    )

    feats = [
        "pos_eta_x",
        "pos_eta_y",
        "pos_eta_mz",
        "pos_nu_x",
        "pos_nu_y",
        "pos_nu_mz",
        "rpm_bow_fore",
        "rpm_bow_aft",
        "rpm_stern_fore",
        "rpm_stern_aft",
        "rpm_fixed_ps",
        "rpm_fixed_sb",
    ]

    dataset = ParquetDataset(
        files, columns=feats, sample_length=5000, standardise=False
    )

    # Sample 5 dataset states for posterior sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset_states = []
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        # batch is a tuple (time, states, meta) from __getitem__
        _, states = prep_batch(batch, device=str(device))
        dataset_states.append(states)

    print("Starting Sampling")

    with Profile() as profile:
        sample_data_prior = load_and_sample(
            Path("dump/latentsde/proof_of_concept/latentsde_491.pth"),
            batch_size=5,
            data_size=12,
            latent_size=4,
            context_size=64,
            hidden_size=128,
            timestep=0.05,
            custom_sampling=False,
            end_time=250,
        )

        # Batch the 5 dataset states and sample posterior once
        batched_data_states = torch.cat(
            dataset_states, dim=1
        )  # (time_steps, batch_size, features)
        sample_data_posterior = load_and_sample(
            Path("dump/latentsde/proof_of_concept/latentsde_491.pth"),
            batch_size=5,
            data_size=12,
            latent_size=4,
            context_size=64,
            hidden_size=128,
            timestep=0.05,
            end_time=250,
            network="posterior",
            custom_sampling=False,
            data_sample=batched_data_states,
        )

        profile.dump_stats(Path("dump/latentsde/profile/sample.stats"))
        pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)

    # print("Finished Sampling")
    # data, ts = sample_data_prior
    # np.save("dump/latentsde/prior_sample_data", data)
    # np.save("dump/latentsde/prior_sample_time", ts)

    # # Save posterior samples
    # post_data, post_ts = sample_data_posterior
    # np.save("dump/latentsde/posterior_sample_data", post_data)
    # np.save("dump/latentsde/posterior_sample_time", post_ts)

    # plot_latentsde(dataset, sample_data_prior, sample_data_posterior)

    # # main()
    # plt.show()
