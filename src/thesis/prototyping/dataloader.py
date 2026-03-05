"""Parquet-backed dataset utilities.

Example usage:
    files = find_parquet_files(Path(filepath), lambda m: m["seed"] < 10)
    dataset = ParquetDataset(files)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, pin_memory=True, num_workers=2)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from thesis.prototyping.data_handling import find_parquet_files

class ParquetDataset(Dataset):
    """Dataset for reading time series from parquet files."""

    def __init__(
        self,
        files: list[Path],
        columns: Optional[list[str]] = None,
        sample_length: Optional[int] = None,
        resample_every: Optional[int] = None,
        resample_dt: Optional[float] = None,
        standardise: bool = True,
        standardise_dict: Optional[Dict[str, np.ndarray]] = None,
        meta_key: str = "run_params",
        device: str = "cpu",
    ) -> None:
        """Initialize the dataset and metadata cache.

        Args:
            files: List of parquet files to read.
            columns: Columns to load; time is prepended if missing.
            sample_length: Number of time steps per sample.
            resample_every: Downsample by fixed step size.
            resample_dt: Downsample by target timestep.
            scale_factors: Per-feature scaling factors (simple multiplication).
            standardize: Dict with 'mean' and 'std' arrays for (x - mean) / std
                normalization. Takes precedence over scale_factors.
            meta_key: Parquet metadata key with JSON payload.
            device: Device hint for downstream usage.
        """
        if len(files) == 0:
            raise AttributeError("File list must include at least one entry")
        
        self.files = files
        if not columns:
            pass
        elif "time" in columns:
            columns.remove("time")
            columns = ["time"] + columns
        else:
            columns = ["time"] + columns

        self.columns = columns
        self.device = device
        if standardise:
            if standardise_dict is not None:
                self.standardise = standardise_dict
            else:
                self.standardise = self.compute_statistics()
        else:
            n_feats = len(self.columns) - 1 if self.columns else 0
            self.standardise = {"mean": np.zeros(n_feats, dtype=np.float32), "std": np.ones(n_feats, dtype=np.float32)}

        series_length = len(pd.read_parquet(self.files[0]))
        self.sample_length = sample_length if sample_length else series_length

        if resample_every is not None and resample_dt is not None:
            raise ValueError("Provide only one of resample_every or resample_dt")
        self.resample_every = resample_every
        self.resample_dt = resample_dt
        
        self.n_per_series = int(np.floor(series_length / self.sample_length))

        self.metas: list[Dict[str, Any]] = []
        for file in self.files:

            schema = pq.read_schema(file)
            meta = schema.metadata or {}

            meta_key_bytes = meta_key.encode("utf-8")
            if meta_key_bytes not in meta:
                continue
            params = json.loads(meta[meta_key_bytes].decode("utf-8"))
            self.metas.append(params)



    def __len__(self) -> int:
        """Return the number of samples across all files.

        Returns:
            Total number of samples in the dataset.
        """
        return self.n_per_series * len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Load a sample window, resample if needed, and return time and states.

        Args:
            idx: Sample index into the virtual concatenated dataset.

        Returns:
            Tuple of time tensor, state tensor, and metadata dict.
        """
        file_idx = idx // self.n_per_series
        states_time = pd.read_parquet(self.files[file_idx], columns=self.columns)
        states_time = np.ascontiguousarray(states_time.values, dtype=np.float32)
        states_time = states_time[(idx % self.n_per_series) * self.sample_length: (idx % self.n_per_series + 1) * self.sample_length, ...]

        if self.resample_every is not None or self.resample_dt is not None:
            if states_time.shape[0] < 2:
                raise ValueError("Series must have at least two timesteps for resampling")

            if self.resample_every is not None:
                step = int(self.resample_every)
            else:
                base_dt = float(states_time[1, 0] - states_time[0, 0])
                if base_dt <= 0:
                    raise ValueError("Base timestep must be positive for resampling")
                step = int(round(float(self.resample_dt) / base_dt))

            if step <= 0:
                raise ValueError("Resample step must be >= 1")
            states_time = states_time[::step]
        
        states = states_time[:, 1:]

        states = (states - self.standardise["mean"]) / self.standardise["std"]

        states = torch.from_numpy(states).to(dtype=torch.float32)

        time = torch.from_numpy(np.round(states_time[:, 0] - states_time[0, 0], 2)).to(dtype=torch.float32)
        meta = self.metas[file_idx].copy()
        meta["idx"] = idx  # Add file index
        return time, states, meta

    def compute_statistics(self) -> None:
        """Compute per-feature mean and std from a list of parquet files.

        Args:

        Returns:
            Dict with 'mean' and 'std' arrays of shape (n_features,).
        """
        n = 0
        mean = np.zeros(len(self.columns) - 1, dtype=np.float64)
        m2 = np.zeros_like(mean)

        for f in self.files:
            data = pd.read_parquet(f, columns=self.columns).values.astype(np.float64)
            values = data[:, 1:]
            for row in values:
                n += 1
                delta = row - mean
                mean += delta / n
                delta2 = row - mean
                m2 += delta * delta2

        std = np.sqrt(m2 / n).astype(np.float32)
        std[std < 1e-8] = 1.0  # avoid division by zero for constant features
        return {"mean": mean.astype(np.float32), "std": std}

    def inverse_scale(self, states: Tensor) -> Tensor:
        """Reverse the scaling applied in __getitem__ to recover original values.

        Args:
            states: Scaled tensor of shape (..., n_features).

        Returns:
            Tensor in original (unscaled) space.
        """
        mean = torch.as_tensor(self.standardise["mean"], dtype=states.dtype, device=states.device)
        std = torch.as_tensor(self.standardise["std"], dtype=states.dtype, device=states.device)
        return states * std + mean


def prep_batch(batch: Tuple[Tensor, Tensor, Dict[str, Any]], device: str = "cpu") -> Tuple[Tensor, Tensor]:
    """Move batch to device and validate aligned timesteps.

    Args:
        batch: Tuple of time, state, and metadata tensors.
        device: Target device for tensors.

    Returns:
        Tuple of time and state tensors on the target device.
    """
    ts, xs, _ = batch
    t = ts[0, :]
    if not torch.allclose(ts, t, atol=0.005):
        raise ValueError("Timesteps between runs are different. Please ensure they are equal")

    xs = torch.permute(xs, (1, 0, 2)).contiguous().to(device, non_blocking=True)
    t = t.to(device, non_blocking=True)
    return t, xs
    


if __name__ == "__main__":
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                                lambda m: m["seed"] < 10)

    dataset = ParquetDataset(files, sample_length=5000)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, pin_memory=True, num_workers=2)
    
    for batch in dataloader:
        ts, xs = prep_batch(batch)
        print(xs.shape)
