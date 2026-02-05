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
                 meta_key: str="run_params",
                 device: str="cpu"):
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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        states_time = pd.read_parquet(self.files[idx], columns=self.columns)
        states_time = np.ascontiguousarray(states_time.values, dtype=np.float32)
        states = torch.from_numpy(states_time[:, 1:])
        time = torch.from_numpy(states_time[:, 0])
        return time, states

def prep_batch(batch: tuple[Tensor, Tensor], device: str="cpu") -> tuple[Tensor, Tensor]:
    """ Function to move data to device as well as verifying the timestep and length"""
    ts, xs = batch
    t = ts[0, :]
    if not torch.allclose(ts, t):
        raise ValueError("Timesteps between runs are different. Please ensure they are equal")

    xs = torch.permute(xs, (1, 0, 2)).contiguous().to(device, non_blocking=True)
    t = t.to(device, non_blocking=True)
    return t, xs
    


if __name__ == "__main__":
    files = find_parquet_files(Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"),
                                lambda m: m["seed"] < 10)

    dataset = ParquetDataset(files)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, pin_memory=True, num_workers=2)
    
    for batch in dataloader:
        prep_batch(batch)
        print(batch.shape)
