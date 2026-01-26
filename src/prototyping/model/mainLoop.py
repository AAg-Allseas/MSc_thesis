#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""

import numpy as np
import matplotlib.pyplot as plt
from src.prototyping.model import plotTimeSeries
from src.prototyping.model.supply import SupplyVessel
from src.prototyping.model.gnc import attitudeEuler


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
             random_mean: np.ndarray | float=np.zeros(3), 
             random_std: np.ndarray | float=np.zeros(3)) -> tuple[np.ndarray, np.ndarray]:
    
    DOF = 6                     # degrees of freedom
    t = 0                       # initial simulation time

    # Initial state vectors
    eta = np.array([0, 0, 0, 0, 0, 0], float)    # position/attitude, user editable
    nu = vessel.nu                              # velocity, defined by vehicle class
    u_actual = vessel.u_actual                  # actual inputs, defined by vehicle class
    
    # Initialization of table used to store the simulation data
    simData = np.empty( [0, 2*DOF + 9], float)

    rng = np.random.default_rng()
    lambda_ = 0.01 # Speed of noise
    f_ext = random_mean + random_std * rng.standard_normal(3)
    rt_dt = np.sqrt(sampleTime)

    # Main simulation loop
    for i in range(0,N+1):
        
        t = i * sampleTime      # simulation time
        if t % 60 == 0:
            print(f"time: {t}s")
   
        u_control = vessel.DPcontrol(eta,nu,sampleTime)
        f_ext = f_ext - lambda_*(f_ext - random_mean) * sampleTime + random_std * rt_dt * rng.standard_normal(3)

        # Store simulation data in simData
        tau_control = vessel.B @ u_control ** 2
        tau_actual = vessel.B @ u_actual ** 2
        signals = np.hstack([eta, nu, tau_control, tau_actual, f_ext])
        simData = np.vstack( [simData, signals] ) 

        # Sample random forcing from normal distribution

        # Propagate vehicle and attitude dynamics
        [nu, u_actual]  = vessel.dynamics(eta,nu,u_actual,u_control,sampleTime, f_external=f_ext)
        eta = attitudeEuler(eta,nu,sampleTime)

        if t == 120:
            vessel.thrusterFailure(5) # Thruster failure of main propeller

    # Store simulation time vector
    simTime = np.arange(start=0, stop=t+sampleTime, step=sampleTime)[:, None]

    return(simTime,simData)

def R2D(value):  # radians to degrees
    return value * 180 / np.pi


if __name__ == "__main__":
    vehicle = SupplyVessel('DPcontrol', V_current=1)

    # Simulation parameters
    sampleTime = 0.02                   # sample time [seconds]
    N = 10000                         # number of samples

    printInfo(vehicle, sampleTime, N)
    [simTime, simData] = simulate(N, sampleTime, vehicle, np.array([0, 0, 0]), np.array([0, 0, 0]))
    plotTimeSeries.plotVehicleStates(simTime, simData, 1)    
    plt.show()