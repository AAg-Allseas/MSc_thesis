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

    # Debug data storage
    # 3x3 controller gains. 3x estimated position, 3x e_int, 6x rpm actuators = 9+3+3+6 = 21
    debugData = np.zeros([N+1, 21])

    rng = np.random.default_rng()
    lambda_ = 0.01 # Speed of noise
    f_ext = random_mean + random_std * rng.standard_normal(3)
    sqrt_dt = np.sqrt(sampleTime)

    # Main simulation loop
    for i in range(0,N+1):
        
        t = i * sampleTime      # simulation time
        if t % 60 == 0:
            print(f"time: {t}s")
   
        u_control = vessel.DPcontrol(eta,nu,sampleTime)
        # u_control = np.zeros(6)

        # Sample random forcing from normal distribution
        f_ext = f_ext - lambda_*(f_ext - random_mean) * sampleTime + random_std * sqrt_dt * rng.standard_normal(3)

        # Store simulation data in simData
        tau_control = vessel.B @ (np.abs(u_control) * u_control)
        tau_actual = vessel.B @ (np.abs(u_actual) * u_actual)
        signals = np.hstack([eta, nu, tau_control, tau_actual, f_ext])
        simData = np.vstack( [simData, signals] )
        
        for j in range(3):
            debugData[i, 3 * j : 3 * (j + 1)] = vessel.gains[j]
            debugData[i, 9+j] = vessel.pos_est[j]
        debugData[i, 12:15] = vessel.e_int
        debugData[i, 15:21] = u_actual


        # Propagate vehicle and attitude dynamics
        [nu, u_actual]  = vessel.dynamics(eta,nu,u_actual,u_control,sampleTime, f_external=f_ext)
        eta = attitudeEuler(eta,nu,sampleTime)

        # if t == 120:
        #     vessel.thrusterFailure(5) # Thruster failure of main propeller

    # Store simulation time vector
    simTime = np.arange(start=0, stop=t+sampleTime, step=sampleTime)[:, None]

    return(simTime,simData, debugData)

def R2D(value):  # radians to degrees
    return value * 180 / np.pi


if __name__ == "__main__":
    vehicle = SupplyVessel('DPcontrol')

    # Simulation parameters
    sampleTime = 0.02                   # sample time [seconds]
    N = 50000                        # number of samples

    printInfo(vehicle, sampleTime, N)
    [simTime, simData, debugData] = simulate(N, sampleTime, vehicle, np.array([100e3, 0, 0]), np.array([80e3, 0, 0]))
    plotTimeSeries.displayPlot(simTime, simData)    
    plotTimeSeries.debugPlot(simTime, simData, debugData, vehicle)
    plt.show()