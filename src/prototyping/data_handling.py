from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

COLUMNS = ['time', 'pos_eta_x', 'pos_eta_y', 'pos_eta_mz', 'pos_nu_x', 'pos_nu_y', 'pos_nu_mz', 'tau_control_x', 'tau_control_y', 'tau_control_mz', 'tau_actual_x', 'tau_actual_y', 'tau_actual_mz', 'tau_ext_x', 'tau_ext_y', 'tau_ext_mz', 'gain_P_x', 'gain_P_y', 'gain_P_mz', 'gain_I_x', 'gain_I_y', 'gain_I_mz', 'gain_D_x', 'gain_D_y', 'gain_D_mz', 'rpm_bow_fore', 'rpm_bow_aft', 'rpm_stern_fore', 'rpm_stern_aft', 'rpm_fixed_ps', 'rpm_fixed_sb']



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
                       rng_seed: int, 
                       dt: float, 
                       T: float,
                       base_name: str = "dp_sim",
                       path: Optional[Path]=None) -> None:

    filename = f"{base_name}_{T}_{dt}_{rng_seed}.parquet"
    df.to_parquet(path / filename)



