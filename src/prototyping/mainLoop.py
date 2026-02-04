#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""

from pathlib import Path
import time
from typing import Generator, Optional
import numpy as np
import matplotlib.pyplot as plt
from src.prototyping.visualisation import plotTimeSeries
from src.prototyping.data_handling import make_df, save_df_to_parquet, update_df
from src.prototyping.model.ornstein_uhlenbeck import ou_generate_uniform, resample_from_base
from src.prototyping.model.supply import SupplyVessel
from src.prototyping.model.gnc import attitudeEuler

import numpy as np
import hashlib
import struct

###############################################################################    
# Function printVehicleinfo(vehicle)
###############################################################################
def printInfo(vehicle, sampleTime, N): 
    """ 
    Function to print vessel and simulation paramters

    args:
        vehicle
        sampleTime
        N
    """
    print('---------------------------------------------------------------------------------------')
    print('%s' % (vehicle.name))
    print('Length: %s m' % (vehicle.L))
    print('%s' % (vehicle.controlDescription))  
    print('Sampling frequency: %s Hz' % round(1 / sampleTime))
    print('Simulation time: %s seconds' % round(N * sampleTime))
    print('---------------------------------------------------------------------------------------')
    
def simulate(N: int, 
             sampleTime: float, 
             vessel: SupplyVessel,
             f_ext: np.ndarray,
             f_ext_dt: float,
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    
    DOF = 6                     # degrees of freedom
    t = 0                       # initial simulation time

    # Initial state vectors
    eta = np.array([0, 0, 0, 0, 0, 0], float)    # position/attitude, user editable
    nu = vessel.nu                              # velocity, defined by vehicle class
    u_actual = vessel.u_actual                  # actual inputs, defined by vehicle class
    
    df = make_df(N)

    f_ext = resample_from_base(f_ext, f_ext_dt, sampleTime, N * sampleTime)
    # Main simulation loop
    for i in range(0,N+1):
        
        t = i * sampleTime      # simulation time
        if t % 60 == 0:
            print(f"time: {t}s")
   
        u_control = vessel.DPcontrol(eta,nu,sampleTime)

        tau_control = vessel.B @ (np.abs(u_control) * u_control)
        tau_actual = vessel.B @ (np.abs(u_actual) * u_actual)

        update_df(df, i, t, eta, nu, tau_control, tau_actual, f_ext[i], vessel.gains, u_actual)
        
        # Propagate vehicle and attitude dynamics
        [nu, u_actual]  = vessel.dynamics(eta,nu,u_actual,u_control,sampleTime, f_external=f_ext[i])
        eta = attitudeEuler(eta,nu,sampleTime)

        # if t == 120:
        #     vessel.thrusterFailure(5) # Thruster failure of main propeller
    save_df_to_parquet(df, seed, sampleTime, (N+1) * sampleTime, path=Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"))    

def R2D(value):  # radians to degrees
    return value * 180 / np.pi

if __name__ == "__main__":
    vehicle = SupplyVessel('DPcontrol')
    runtime = 1000 # seconds
    seeds = range(0, 11)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        noise_dt = 0.01
        external_forces = ou_generate_uniform(int(runtime // noise_dt), noise_dt, mu=np.array([75e3, 75e3, 100e3]), sigma=np.array([50e3, 50e3, 50e3]), rng=rng)

        sampleTimes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

        for sampleTime in sampleTimes:

            N = int(runtime // sampleTime)     # number of samples

            start_time = time.time_ns()
            printInfo(vehicle, sampleTime, N)
            simulate(N, sampleTime, vehicle, f_ext=external_forces, f_ext_dt=noise_dt, seed=seed)
            end_time = time.time_ns()
            print(f"Time taken: {(end_time-start_time)/1e9:.3f}s")
