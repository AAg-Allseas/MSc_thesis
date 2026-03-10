"""
Thruster configuration and control allocation.

Supports tunnel thrusters, main propellers, and azimuth thrusters.
Includes both pseudoinverse and constrained (SQP) allocation methods.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def thruster_config(
    alpha: list[str | float],
    l_x: list[float],
    l_y: list[float],
) -> NDArray:
    """Thruster configuration matrix T_thr (3 x n_thrusters).

    Parameters
    ----------
    alpha : list
        Thruster types: ``'T'`` (tunnel), ``'M'`` (main propeller),
        or a float (azimuth angle in rad).
    l_x, l_y : list
        Longitudinal and lateral positions of each thruster.
    """
    n = len(alpha)
    T = np.zeros((3, n))
    for i in range(n):
        a = alpha[i]
        if a == "T":
            T[:, i] = [0, 1, l_x[i]]
        elif a == "M":
            T[:, i] = [1, 0, -l_y[i]]
        else:
            az = float(a)
            T[:, i] = [np.cos(az), np.sin(az),
                       l_x[i] * np.sin(az) - l_y[i] * np.cos(az)]
    return T


def alloc_pseudoinverse(
    K: NDArray,
    T: NDArray,
    W: NDArray,
    tau: NDArray,
) -> NDArray:
    """Unconstrained control allocation using weighted pseudoinverse."""
    Winv = np.diag(1.0 / np.diag(W))
    Kinv = np.diag(1.0 / np.diag(K))
    return Kinv @ Winv @ T.T @ np.linalg.solve(T @ Winv @ T.T, tau)


def optimal_alloc(
    tau: NDArray,
    lb: NDArray,
    ub: NDArray,
    alpha_old: NDArray,
    u_old: NDArray,
    l_x: list[float],
    l_y: list[float],
    K_thr: NDArray,
    h: float,
) -> tuple[NDArray, NDArray, float]:
    """Constrained control allocation via SQP (scipy SLSQP).

    Returns ``(alpha_opt, u_opt, slack_norm)``.
    """
    max_rate_alpha = 0.3
    max_rate_u = 0.1

    def objective(x):
        alpha, u, s = x[:2], x[2:6], x[6:9]
        w1, w2, w3, w4 = 1, 100, 1, 0.1
        return (w1 * np.dot(u, u) + w2 * np.dot(s, s)
                + w3 * np.linalg.norm(alpha - alpha_old) ** 2
                + w4 * np.linalg.norm(u - u_old) ** 2)

    def eq_constraint(x):
        alpha, u, s = x[:2], x[2:6], x[6:9]
        T_a = thruster_config(["T", "T", alpha[0], alpha[1]], l_x, l_y)
        return T_a @ K_thr @ u - tau + s

    def ineq_constraints(x):
        alpha, u = x[:2], x[2:6]
        c = np.empty(12)
        for j in range(2):
            da = (alpha[j] - alpha_old[j]) / h
            c[2 * j] = max_rate_alpha - da
            c[2 * j + 1] = max_rate_alpha + da
        for j in range(4):
            du = (u[j] - u_old[j]) / h
            c[4 + 2 * j] = max_rate_u - du
            c[4 + 2 * j + 1] = max_rate_u + du
        return c

    x0 = np.array([np.deg2rad(-28), np.deg2rad(28),
                    0, 0, 0, 0, 0, 0, 0])
    result = minimize(
        objective, x0, method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints=[
            {"type": "eq", "fun": eq_constraint},
            {"type": "ineq", "fun": ineq_constraints},
        ],
        options={"disp": False, "maxiter": 200},
    )
    xo = result.x
    return xo[:2], xo[2:6], float(np.linalg.norm(xo[6:9]))
