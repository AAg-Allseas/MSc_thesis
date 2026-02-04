from dataclasses import asdict, dataclass, field
import datetime
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd

import json
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ['time', 'pos_eta_x', 'pos_eta_y', 'pos_eta_mz', 'pos_nu_x', 'pos_nu_y', 'pos_nu_mz', 'tau_control_x', 'tau_control_y', 'tau_control_mz', 'tau_actual_x', 'tau_actual_y', 'tau_actual_mz', 'tau_ext_x', 'tau_ext_y', 'tau_ext_mz', 'gain_P_x', 'gain_P_y', 'gain_P_mz', 'gain_I_x', 'gain_I_y', 'gain_I_mz', 'gain_D_x', 'gain_D_y', 'gain_D_mz', 'rpm_bow_fore', 'rpm_bow_aft', 'rpm_stern_fore', 'rpm_stern_aft', 'rpm_fixed_ps', 'rpm_fixed_sb']

@dataclass(frozen=True)
class ParquetMetadata:
    model: str
    version: str
    seed: int
    timestep: float
    end_time: float
    n_steps: int
    mean_force: list[float]
    var_force:  list[float]
    inital_pos: tuple[float, float, float]
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

def make_df(N: int) -> pd.DataFrame:
    d = len(COLUMNS)
    return pd.DataFrame(np.empty([N+1, d]), columns=COLUMNS)

def update_df(df: pd.DataFrame,
              idx: int, 
              t: float, 
              eta: np.ndarray, 
              nu: np.ndarray, 
              tau_control: np.ndarray, 
              tau_actual: np.ndarray, 
              f_ext: np.ndarray, 
              pid_gains: np.ndarray, 
              rpms: np.ndarray) -> None:
    """ Function to update data storage dataframe"""
    
    df.iloc[idx, 0] = t

    # Save eta and nu, only 3DOF
    df.iloc[idx, 1:4] = eta[[0, 1, 5]]
    df.iloc[idx, 4:7] = nu[[0, 1, 5]]

    # Forces
    df.iloc[idx, 7:16] = np.r_[tau_control, tau_actual, f_ext]

    # PID gains
    df.iloc[idx, 16:19] = pid_gains[0]  # P
    df.iloc[idx, 19:22] = pid_gains[1]  # I
    df.iloc[idx, 22:25] = pid_gains[2]  # D

    # Thrusters
    df.iloc[idx, 25:] = rpms


def save_df_to_parquet(df: pd.DataFrame,
                       metadata: ParquetMetadata, 
                       base_name: str = "dp_sim",
                       path: Optional[Path]=None) -> None:
    """
    Function to save a dataframe as a Parquet file. Encodes metadata into the file for efficient indexing. 

    File is saved with format: "{base_name}_{metadata.end_time}_{metadata.timestep}_{metadata.seed}.parquet"
    
    :param df: Dataframe to be saved
    :type df: pd.DataFrame
    :param metadata: Metadata dataclass containing information about the simulation to be saved
    :type metadata: ParquetMetadata
    :param base_name: Base name used to save the file
    :type base_name: str
    :param path: Folder to save the Parquet file in
    :type path: Optional[Path]
    """
    table = pa.Table.from_pandas(df)
    meta = dict(table.schema.metadata or {})
    meta_key = b"run_params"
    meta[meta_key] = json.dumps(asdict(metadata)).encode("utf-8")
    
    table = table.replace_schema_metadata(meta)

    filename = f"{base_name}_{metadata.end_time}_{metadata.timestep}_{metadata.seed}.parquet"
    if path:
        path = path / filename
    else:
        path = filename
    
    pq.write_table(table, path)


def find_parquet_files(
    folder: Path,
    filter_fn: Callable=lambda meta: True,
    meta_key: str="run_params"
    ) -> list[Path]:
    """
    Function to search for Parquet files based on their metadata.

    Example filter functions:
        filter_fn = lambda m: m["seed"] == 123 and m["version"] == "0.4"
        filter_fn = lambda m: m["model"] == "V3" and m["lr"] < 1e-3
    
    :param folder: Folder to search for Parquet files in
    :type folder: Path
    :param filter_fn: Lambda function used to match metadata
    :type filter_fn: Callable
    :param meta_key: Key under which the parameters to be found are indexed in the metadata
    :type meta_key: str
    """    
    folder = Path(folder)
    matches = []

    for path in folder.glob("*.parquet"):
        try:
            schema = pq.read_schema(path)
            meta = schema.metadata or {}

            meta_key_bytes = meta_key.encode("utf-8")
            if meta_key_bytes not in meta:
                continue

            params = json.loads(meta[meta_key_bytes].decode("utf-8"))

            if filter_fn(params):
                matches.append((path, params))

        except Exception:
            pass  # Handle corrupt files etc.

    return matches
