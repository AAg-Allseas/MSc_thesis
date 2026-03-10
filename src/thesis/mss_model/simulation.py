"""
OSV dynamic positioning simulation runner.

Ties together the vessel model, PID controller, thruster allocation,
and environmental loads into a time-domain simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from thesis.mss_model.gnc import ssa, rk4
from thesis.mss_model.thruster import thruster_config, alloc_pseudoinverse, optimal_alloc
from thesis.mss_model.control import PIDNonlinearMIMO
from thesis.mss_model.vessel import OSV


@dataclass
class SimConfig:
    """Configuration for the OSV DP simulation."""

    T_final: float = 250.0
    h: float = 0.05
    x_ref: float = 0.0
    y_ref: float = 0.0
    psi_ref: float = 0.0
    Vc: float = 0.5
    betaVc: float = np.deg2rad(-140)
    Hs: float = 0.0
    Tp: float = 8.0
    beta_wave: float = 0.0
    alloc_dynamic: bool = True


def simulate_osv(cfg: SimConfig | None = None) -> dict[str, NDArray]:
    """Run the full OSV DP simulation.

    Returns a dict with keys: ``t``, ``eta``, ``nu``, ``n``, ``alpha``.
    """
    if cfg is None:
        cfg = SimConfig()

    vessel = OSV()
    v = vessel.params

    # Constant azimuth angles
    alpha0 = np.deg2rad(np.array([-28.0, 28.0]))

    # Thruster limits for allocation
    K_max = np.diag([300e3, 300e3, 655e3, 655e3])
    n_max = np.array([140.0, 140.0, 150.0, 150.0])
    l_x = [37.0, 35.0, -42.0, -42.0]
    l_y = [0.0, 0.0, 7.0, -7.0]

    T_thr = thruster_config(["T", "T", alpha0[0], alpha0[1]], l_x, l_y)

    az_max = np.deg2rad(60)
    lb = np.array([-az_max, -az_max, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf])
    ub = np.array([az_max, az_max, 1, 1, 1, 1, np.inf, np.inf, np.inf])

    alpha_old = alpha0.copy()
    u_old = np.zeros(4)

    # PID controller
    M = v.M
    wn = 0.1 * np.diag([1.0, 1.0, 3.0])
    zeta = 1.0 * np.diag([1.0, 1.0, 1.0])
    T_f = 30.0
    pid = PIDNonlinearMIMO()

    # Initial state
    eta = np.array([5.0, 5.0, 0.0, np.deg2rad(5), np.deg2rad(2), 0.0])
    nu = np.zeros(6)
    x = np.concatenate([nu, eta])

    eta_ref = np.array([cfg.x_ref, cfg.y_ref, cfg.psi_ref])

    t = np.arange(0, cfg.T_final + cfg.h / 2, cfg.h)
    n_steps = len(t)

    sim_eta = np.zeros((n_steps, 6))
    sim_nu = np.zeros((n_steps, 6))
    sim_n = np.zeros((n_steps, 4))
    sim_alpha = np.zeros((n_steps, 2))

    for i in range(n_steps):
        # Sensor noise
        eta[0] += 0.0001 * np.random.randn()
        eta[1] += 0.0001 * np.random.randn()
        eta[5] += 0.0001 * np.random.randn()

        # Setpoint change at t > 50 s
        if t[i] > 50:
            eta_ref = np.array([cfg.x_ref, cfg.y_ref, np.deg2rad(40)])

        # PID controller
        tau = pid(eta, nu, eta_ref, M, wn, zeta, T_f, cfg.h)

        # Control allocation
        if not cfg.alloc_dynamic:
            alpha_c = alpha0.copy()
            u_c = alloc_pseudoinverse(K_max, T_thr, np.eye(4), tau[[0, 1, 5]])
        else:
            alpha_c, u_c, _ = optimal_alloc(
                tau[[0, 1, 5]], lb, ub, alpha_old, u_old,
                l_x, l_y, K_max, cfg.h)
            alpha_old = alpha_c.copy()
            u_old = u_c.copy()

        # Scale to propeller speeds
        u_c = n_max**2 * u_c
        n_c = np.sign(u_c) * np.sqrt(np.abs(u_c))
        ui = np.concatenate([n_c, alpha_c])

        sim_eta[i] = eta
        sim_nu[i] = nu
        sim_n[i] = n_c
        sim_alpha[i] = alpha_c

        # RK4 step
        x = rk4(vessel, cfg.h, x, ui, cfg.Vc, cfg.betaVc,
                 cfg.Hs, cfg.Tp, cfg.beta_wave)
        nu = x[:6]
        eta = x[6:]

    return {"t": t, "eta": sim_eta, "nu": sim_nu, "n": sim_n, "alpha": sim_alpha}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    results = simulate_osv()
    t = results["t"]
    eta = results["eta"]
    nu = results["nu"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(eta[:, 1], eta[:, 0])
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("North-East positions")
    axes[0].grid(True)

    axes[1].plot(t, np.rad2deg(np.vectorize(ssa)(eta[:, 5])))
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Heading (deg)")
    axes[1].grid(True)

    U = np.sqrt(nu[:, 0] ** 2 + nu[:, 1] ** 2)
    axes[2].plot(t, U)
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Speed (m/s)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
