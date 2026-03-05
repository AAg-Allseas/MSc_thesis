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
from thesis.prototyping.model.gnc import ssa
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from thesis.prototyping.model.supply import SupplyVessel

legendSize = 10  # legend size
figSize1 = [15, 10]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 96  # figure dpi value


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def displayPlot(simTime: np.ndarray, simData: np.ndarray) -> None:

    # Time vector
    t = simTime

    # State vectors

    #eta
    x = simData[:, 0]
    y = simData[:, 1]
    simData[:, 2]
    R2D(ssa(simData[:, 3]))
    R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))

    #nu
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    R2D(simData[:, 9])
    R2D(simData[:, 10])
    R2D(simData[:, 11])

    # Actuator forces
    simData[:, 12]
    simData[:, 13]
    simData[:, 14]
    u_x_actual = simData[:, 15]
    u_y_actual = simData[:, 16]
    simData[:, 17]
    f_ext_x = simData[:, 18]
    f_ext_y = simData[:, 19]
    simData[:, 20]


    # Speed
    np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    R2D(ssa(np.arctan2(w,u)))   # flight path angle
    R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    
    fig, axs = plt.subplots(3, 2, figsize=figSize1, dpi=dpiValue)
    fig.suptitle("Vehicle states")
    axs[0, 0].set_title("North-East positions (m)")
    axs[0, 0].set_xlabel("East position (m)")
    axs[0, 0].set_ylabel("North position (m)")
    axs[0, 0].plot(y, x)
    axs[0, 0].axis("equal")
    t_flat = t.reshape(-1)
    sample_times = np.arange(float(t_flat[0]), float(t_flat[-1]) + 1e-9, 25.0)
    marker_idx = np.unique([np.abs(t_flat - ts).argmin() for ts in sample_times])
    for i in marker_idx:
        axs[0, 0].scatter(
            y[i],
            x[i],
            s=60,
            marker=(3, 0, psi[i]),
            facecolor="tab:red",
            edgecolor="k",
            zorder=3,
        )
    axs[0, 0].grid()


    axs[0, 1].set_title("Velocity")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].plot(t, u, label="Surge")
    axs[0, 1].plot(t, v, label="Sway")
    axs[0, 1].legend(fontsize=legendSize)
    axs[0, 1].grid()

    axs[1, 0].set_title("Thruster force")
    axs[1, 0].set_ylabel("Force (kN)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].plot(t, u_x_actual / 1e3, label="Surge")
    axs[1, 0].plot(t, u_y_actual / 1e3, label="Sway")
    axs[1, 0].legend(fontsize=legendSize)
    axs[1, 0].grid()

    axs[1, 1].set_title("External force")
    axs[1, 1].set_ylabel("Force (kN)")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].plot(t, f_ext_x / 1e3, label="Surge")
    axs[1, 1].plot(t, f_ext_y / 1e3, label="Sway")
    axs[1, 1].legend(fontsize=legendSize)
    axs[1, 1].grid()

    gs = axs[2, 0].get_gridspec()
    for ax in axs[2, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, :])
    axbig.plot(t, x, label="Surge")
    axbig.plot(t, y, label="Sway")
    axbig.set_xlabel("Time (s)")
    axbig.set_ylabel("Displacement (m)")
    axbig.legend(fontsize=legendSize)
    axbig.grid()

    fig.tight_layout()
    fig.savefig("plots/vehicleState.png")


def debugPlot(simTime: np.ndarray, simData: np.ndarray, debugData: np.ndarray, vehicle: SupplyVessel) -> None:

    t = simTime

    #eta
    x = simData[:, 0]
    y = simData[:, 1]
    simData[:, 2]
    R2D(ssa(simData[:, 3]))
    R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))

    pos = (x, y, psi)

    #nu
    simData[:, 6]
    simData[:, 7]
    simData[:, 8]
    R2D(simData[:, 9])
    R2D(simData[:, 10])
    R2D(simData[:, 11])

    # Actuator forces
    simData[:, 12]
    simData[:, 13]
    simData[:, 14]
    simData[:, 15]
    simData[:, 16]
    simData[:, 17]
    simData[:, 18]
    simData[:, 19]
    simData[:, 20]

    p_gains = debugData[:, :3]
    i_gains = debugData[:, 3:6]
    d_gains = debugData[:, 6:9]
    gains = (p_gains, i_gains, d_gains)

    pos_est = debugData[:, 9:12]
    e_int = debugData[:, 12:15]
    rpm_actual = debugData[:, 15:21]
    dof_labels = ["Surge", "Sway", "Yaw"]
    pid_labels = ["Proportional", "Integral", "Derivative"]
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=figSize1, dpi=dpiValue)

    for i in range(3): # DOF
        for j in range(3): # PID
            axs[0, i].plot(t, gains[j][:, i], label=pid_labels[j])
        axs[0, i].set_title(f"PID gains in {dof_labels[i]}")
        axs[0, i].set_xlabel("Gain (-)")
        axs[0, i].set_ylabel("Time (s)")
        axs[0, i].legend()

        axs[1, 0].plot(t, pos_est[:, i], label=dof_labels[i])
        axs[1, 1].plot(t, pos[i], label=dof_labels[i])
        axs[1, 2].plot(t, e_int[:, i], label=dof_labels[i])

    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[1, 2].legend()

    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[1, 2].grid()


    thruster_labels = ["Bow Thruster 1", "Bow Thruster 2", "Bow Thruster 3", "Bow Thruster 4", "Main Propeller 1", "Main Propeller 2"]

    for i in range(6):
        y = i % 2 + 2
        x = (i) // 2
        print(f"i: {i} -> ({x}, {y})")
        axs[y, x].plot(t, rpm_actual[:, i])
        axs[y, x].set_title(f"{thruster_labels[i]} RPS")
        axs[y, x].axhline(vehicle.n_max[i], color="k", linestyle="--")
        axs[y, x].axhline(-vehicle.n_max[i], color="k", linestyle="--")
        axs[y, x].set_xlabel("Time (s)")
        axs[y, x].set_ylabel("RPS")

    fig.tight_layout()


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
    N = y[::len(x) // numDataPoints]
    E = x[::len(x) // numDataPoints]
    D = z[::len(x) // numDataPoints]
    
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

