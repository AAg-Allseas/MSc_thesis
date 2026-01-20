# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from src.prototyping.model.gnc import ssa
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

legendSize = 10  # legend size
figSize1 = [15, 10]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 300  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, figNo):

    # Time vector
    t = simTime

    # State vectors

    #eta
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))

    #nu
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])

    # Actuator forces
    u_x_control = simData[:, 12]
    u_y_control = simData[:, 13]
    u_r_control = simData[:, 14]
    u_x_actual = simData[:, 15]
    u_y_actual = simData[:, 16]
    u_r_actual = simData[:, 17]
    f_ext_x = simData[:, 18]
    f_ext_y = simData[:, 19]
    f_ext_r = simData[:, 20]


    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    beta_c  = R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    alpha_c = R2D(ssa(np.arctan2(w,u)))   # flight path angle
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    fig, ax = plt.subplots(3, 3, figsize=figSize1)
    ax[0, 0].plot(y, x, label="North-East positions (m)")
    ax[0, 0].legend(fontsize=legendSize)
    ax[0, 0].grid()

    fig.suptitle("Vehicle states", fontsize=12)

    ax[1, 0].plot(t, U, label="Speed (m/s)")
    ax[1, 0].legend(fontsize=legendSize)
    ax[1, 0].grid()


    ax[0, 1].plot(t, u, label="Surge velocity (m/s)")
    ax[0, 1].plot(t, v, label="Sway velocity (m/s)")
    ax[0, 1].set_xlabel("Time (s)", fontsize=12)
    ax[0, 1].legend(fontsize=legendSize)
    ax[0, 1].grid()

    ax[1, 1].plot(t, r, label="Yaw rate (deg/s)")
    ax[1, 1].set_xlabel("Time (s)", fontsize=12)
    ax[1, 1].legend(fontsize=legendSize)
    ax[1, 1].grid()

    ax[0, 2].plot(t, chi, label="Course angle (deg)")
    ax[0, 2].legend(fontsize=legendSize)
    ax[0, 2].grid()

    ax[1, 2].plot(t, psi, label="Yaw angle (deg)")
    ax[1, 2].plot(t, beta_c, label="Crab angle (deg)")
    ax[1, 2].set_xlabel("Time (s)", fontsize=12)
    ax[1, 2].legend(fontsize=legendSize)
    ax[1, 2].grid()

    ax[2, 0].plot(t, u_x_control, label="Surge force")
    ax[2, 0].plot(t, u_y_control, label="Sway force")
    ax[2, 0].plot(t, u_r_control, label="Yaw moment")

    ax[2, 1].plot(t, u_x_actual, label="Surge force")
    ax[2, 1].plot(t, u_y_actual, label="Sway force")
    ax[2, 1].plot(t, u_r_actual, label="Yaw moment")

    ax[2, 2].plot(t, f_ext_x, label="Surge force")
    ax[2, 2].plot(t, f_ext_y, label="Sway force")
    ax[2, 2].plot(t, f_ext_r, label="Yaw moment")


# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo
def plotControls(simTime, simData, vehicle, figNo):

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(
        figNo, figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])), dpi=dpiValue
    )

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    # Plot the vehicle.dimU active control inputs
    for i in range(0, vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]  # control input, commands
        u_actual = simData[:, 2 * DOF + vehicle.dimU + i]  # actual control input

        if vehicle.controls[i].find("deg") != -1:  # convert angles to deg
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend(
            [vehicle.controls[i] + ", command", vehicle.controls[i] + ", actual"],
            fontsize=legendSize,
        )
        plt.xlabel("Time (s)", fontsize=12)
        plt.grid()


# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position (x, y, z) in 3D
# in figure no. figNo
def plot3D(simData,numDataPoints,FPS,filename,figNo):
        
    # State vectors
    x = simData[:,0]
    y = simData[:,1]
    z = simData[:,2]
    
    # down-sampling the xyz data points
    N = y[::len(x) // numDataPoints];
    E = x[::len(x) // numDataPoints];
    D = z[::len(x) // numDataPoints];
    
    # Animation function
    def anim_function(num, dataSet, line):
        
        line.set_data(dataSet[0:2, :num])    
        line.set_3d_properties(dataSet[2, :num])    
        ax.view_init(elev=10.0, azim=-120.0)
        
        return line
    
    dataSet = np.array([N, E, -D])      # Down is negative z
    
    # Attaching 3D axis to the figure
    fig = plt.figure(figNo,figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
               dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax) 
    
    # Line/trajectory plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0] 

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlim3d([-100, 20])                   # default depth = -100 m
    
    if np.amax(z) > 100.0:
        ax.set_zlim3d([-np.amax(z), 20])
        
    ax.set_zlabel('-Z / Down')

    # Plot 2D surface for z = 0
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # Title of plot
    ax.set_title('North-East-Down')
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                         anim_function, 
                         frames=numDataPoints, 
                         fargs=(dataSet,line),
                         interval=200, 
                         blit=False,
                         repeat=True)
    
    # Save the 3D animation as a gif file
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))  

