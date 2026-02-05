#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""

from concurrent import futures
from pathlib import Path
import time
import numpy as np
from src.prototyping.data_handling import ParquetMetadata, make_df, save_df_to_parquet, update_df
from src.prototyping.model.ornstein_uhlenbeck import ou_generate_uniform, resample_from_base
from src.prototyping.model.supply import SupplyVessel
from src.prototyping.model.gnc import attitudeEuler


MODEL="ToyDPModel"
VERSION="1.0"

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
             meta: ParquetMetadata
            ) -> tuple[np.ndarray, np.ndarray]:
    
    t = 0                       # initial simulation time

    # Initial state vectors
    eta = vessel.eta                            # position/attitude, user editable
    nu = vessel.nu                              # velocity, defined by vehicle class
    u_actual = vessel.u_actual                  # actual inputs, defined by vehicle class
    
    df = make_df(N)


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
    save_df_to_parquet(df, 
                       metadata=meta,
                       path=Path(r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\data"))    

def R2D(value):  # radians to degrees
    return value * 180 / np.pi

def run_sim(seed: int, 
            runtime: float, 
            mu_f: np.ndarray, 
            sigma_f: np.ndarray, 
            vessel: SupplyVessel, 
            sampleTime: float) -> None:
    
    rng = np.random.default_rng(seed)
    N = int(runtime // sampleTime)     # number of samples

    # UO process for external forces - independent of timestep
    noise_dt = 0.05
    external_forces = ou_generate_uniform(int(runtime // noise_dt), noise_dt, mu= mu_f, sigma=sigma_f, rng=rng)
    external_forces = resample_from_base(external_forces, noise_dt, sampleTime, N * sampleTime)

    # Initial condition
    x_0, y_0 = rng.uniform(low=-3, high=3, size=2)
    psi_0 = rng.uniform(low=-0.15, high=0.15)
    pos_0 = (x_0, y_0, psi_0)
    vessel.eta[0] = x_0
    vessel.eta[1] = y_0
    vessel.eta[5] = psi_0

    u_0 = vessel.DPcontrol(vessel.eta, vessel.nu, sampleTime)
    vessel.u_actual = u_0



    start_time = time.time_ns()
    printInfo(vessel, sampleTime, N)

    meta = ParquetMetadata(model=MODEL,
                            version=VERSION,
                            timestep=sampleTime,
                            end_time=runtime,
                            seed=seed,
                            n_steps=N,
                            mean_force=list(mu_f),
                            var_force=list(sigma_f),
                            inital_pos=pos_0)
    simulate(N, sampleTime, vessel, f_ext=external_forces, meta=meta)
    end_time = time.time_ns()
    print(f"Time taken: {(end_time-start_time)/1e9:.3f}s")

if __name__ == "__main__":
    vessel = SupplyVessel('DPcontrol')
    runtime = 3600 * 3 # seconds
    sampleTime = 0.05 # Seconds [s]

    # Number of runs and seeds
    seeds = range(0, 50)

    # External forces
    mu_f = np.array([75e3, 75e3, 100e3])  # Mean
    sigma_f = np.array([50e3, 50e3, 50e3])  # Variance

    with futures.ProcessPoolExecutor() as pool:
        fs = []
        for seed in seeds:
            print(seed)
            fs.append(
                pool.submit(run_sim,
                            seed=seed,
                            runtime=runtime,
                            mu_f=mu_f,
                            sigma_f=sigma_f,
                            vessel=vessel,
                            sampleTime=sampleTime)
                )
        for future in futures.as_completed(fs):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")
