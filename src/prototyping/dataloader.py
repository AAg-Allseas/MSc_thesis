""" 
Example usage 
files = find_parquet_files(Path(filepath,
                            lambda m: m["seed"] < 10)

dataset = ParquetDataset(files)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, pin_memory=True, num_workers=2)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from src.prototyping.data_handling import find_parquet_files

class ParquetDataset(Dataset):
    """ Dataloader for reading in time series from parquet"""
    def __init__(self, files: list[Path], 
                 columns: Optional[list[str]]=None, 
                 sample_length: Optional[int]=None,
                 resample_every: Optional[int]=None,
                 resample_dt: Optional[float]=None,
                 scale_factors: Optional[np.ndarray] = None,
                 meta_key: str="run_params",
                 device: str="cpu"):
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
        self.meta_key = meta_key
        self.device = device
        if scale_factors is not None:
            self.scale_factors = scale_factors

        series_length = len(pd.read_parquet(self.files[0]))
        self.sample_length = sample_length if sample_length else series_length

        if resample_every is not None and resample_dt is not None:
            raise ValueError("Provide only one of resample_every or resample_dt")
        self.resample_every = resample_every
        self.resample_dt = resample_dt
        
        self.n_per_series = int(np.floor(series_length / self.sample_length))


    def __len__(self):
        return self.n_per_series * len(self.files)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """ 
        __getitem__ method used by dataloader. 
        Opens files based on the index and splits it into a sample. 
        Scales the states and adjusts the time series to start from 0.
        """
        states_time = pd.read_parquet(self.files[idx // self.n_per_series], columns=self.columns)
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
        if self.scale_factors is not None:
            states = states * self.scale_factors
        states = torch.from_numpy(states).to(dtype=torch.float32)

        time = torch.from_numpy(np.round(states_time[:, 0] - states_time[0, 0], 2)).to(dtype=torch.float32)
        return time, states

def prep_batch(batch: tuple[Tensor, Tensor], device: str="cpu") -> tuple[Tensor, Tensor]:
    """ Function to move data to device as well as verifying the timestep and length"""
    ts, xs = batch
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
