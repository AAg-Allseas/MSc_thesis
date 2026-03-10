"""
DP controllers for marine vessels.

MIMO nonlinear PID controller (Fossen 2021, Algorithm 15.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from thesis.mss_model.gnc import ssa


class PIDNonlinearMIMO:
    """MIMO nonlinear PID controller for dynamic positioning.

    Operates in the horizontal plane (surge, sway, yaw) with a
    first-order reference-model filter for smooth setpoint tracking.
    """

    def __init__(self) -> None:
        self.z_int = np.zeros(3)
        self.eta_d = np.zeros(3)

    def __call__(
        self,
        eta: NDArray,
        nu: NDArray,
        eta_ref: NDArray,
        M: NDArray,
        wn: NDArray,
        zeta: NDArray,
        T_f: float,
        h: float,
    ) -> NDArray:
        eta_ref = eta_ref.ravel()
        dof6 = len(nu) == 6

        if dof6:
            eta3 = np.array([eta[0], eta[1], eta[5]])
            nu3 = np.array([nu[0], nu[1], nu[5]])
            M3 = M[np.ix_([0, 1, 5], [0, 1, 5])]
        else:
            eta3, nu3, M3 = eta, nu, M

        R = np.array([
            [np.cos(eta3[2]), -np.sin(eta3[2]), 0],
            [np.sin(eta3[2]),  np.cos(eta3[2]), 0],
            [0, 0, 1],
        ])

        M_diag = np.diag(np.diag(M3))
        Kp = M_diag @ wn @ wn
        Kd = M_diag @ (2 * zeta @ wn)
        Ki = 0.1 * Kp @ wn

        e = eta3 - self.eta_d
        e[2] = ssa(e[2])
        tau_PID = -R.T @ (Kp @ e + Ki @ self.z_int) - Kd @ nu3

        if dof6:
            tau = np.array([tau_PID[0], tau_PID[1], 0, 0, 0, tau_PID[2]])
        else:
            tau = tau_PID

        self.z_int += h * (eta3 - self.eta_d)
        self.eta_d += h * (eta_ref - self.eta_d) / T_f
        return tau
