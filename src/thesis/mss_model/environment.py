"""
Environmental loads: ocean current and wave drift forces.

Ocean current model from Fossen (2021, Ch. 8).
Wave spectrum models translated from the MSS toolbox ``waveSpectrum.m``.
Mean wave drift forces via numerical integration of drift force
coefficients over the wave energy spectrum (Fossen 2021, Eq. 8.87):

    F_drift_i = 2 * integral{ S(w) * T_i(w, beta) dw }

where S(w) is the wave energy spectrum and T_i(w, beta) is the
drift force transfer function for DOF *i* and wave heading *beta*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from thesis.mss_model.gnc import smtrx


# ---------------------------------------------------------------------------
# Ocean current
# ---------------------------------------------------------------------------

def ocean_current(
    Vc: float,
    betaVc: float,
    psi: float,
    nu_ang: NDArray,
) -> tuple[NDArray, NDArray]:
    """Irrotational ocean current in the body frame.

    Parameters
    ----------
    Vc : float
        Current speed (m/s).
    betaVc : float
        Current direction in NED (rad).
    psi : float
        Vessel heading (rad).
    nu_ang : NDArray
        Angular body velocities ``[p, q, r]``.

    Returns
    -------
    nu_c : NDArray
        6-DOF current velocity in body frame.
    nu_c_dot : NDArray
        Time derivative of ``nu_c`` (due to vessel rotation in current).
    """
    v_c = np.array([
        Vc * np.cos(betaVc - psi),
        Vc * np.sin(betaVc - psi),
        0.0,
    ])
    nu_c = np.concatenate([v_c, np.zeros(3)])
    nu_c_dot = np.concatenate([-smtrx(nu_ang) @ v_c, np.zeros(3)])
    return nu_c, nu_c_dot


# ---------------------------------------------------------------------------
# Wave spectra (translated from MSS waveSpectrum.m)
# ---------------------------------------------------------------------------

def jonswap_spectrum(
    omega: NDArray,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
) -> NDArray:
    """JONSWAP wave energy spectrum.

    Parameters
    ----------
    omega : NDArray
        Wave frequencies (rad/s).
    Hs : float
        Significant wave height (m).
    Tp : float
        Peak period (s).
    gamma : float
        Peak enhancement factor (default 3.3).

    Returns
    -------
    S : NDArray
        Spectral density values (m^2 s / rad).
    """
    wp = 2 * np.pi / Tp
    # Goda (1999) normalisation constant
    sigma = np.where(omega <= wp, 0.07, 0.09)
    A_gamma = 1 - 0.287 * np.log(gamma)
    alpha = (Hs**2 / 16) * (wp**4) / (A_gamma * 0.3125)  # 5/16

    S_pm = (alpha / omega**5) * np.exp(-1.25 * (wp / omega) ** 4)
    G = gamma ** np.exp(-0.5 * ((omega - wp) / (sigma * wp)) ** 2)
    return S_pm * G


def pierson_moskowitz_spectrum(omega: NDArray, Hs: float, Tp: float) -> NDArray:
    """Pierson-Moskowitz (ITTC modified) wave spectrum."""
    wp = 2 * np.pi / Tp
    A = (5 / 16) * Hs**2 * wp**4
    B = 1.25 * wp**4
    return A / omega**5 * np.exp(-B / omega**4)


# ---------------------------------------------------------------------------
# Wave drift forces
# ---------------------------------------------------------------------------

@dataclass
class WaveDriftCoefficients:
    """Drift-force transfer functions T_i(omega, beta) for the OSV.

    The coefficients represent non-dimensional mean drift force amplitudes
    per unit wave amplitude squared, for each DOF (surge, sway, yaw)
    at discrete frequencies.

    If no hydrodynamic solver data (VERES/WAMIT) is available, simplified
    Newman-type approximations can be used by providing approximate
    coefficients.

    Attributes
    ----------
    omega : NDArray
        Discrete frequencies at which coefficients are defined (rad/s).
    T_surge : NDArray
        Drift force coefficient in surge (N/m^2) at each frequency.
    T_sway : NDArray
        Drift force coefficient in sway (N/m^2) at each frequency.
    T_yaw : NDArray
        Drift moment coefficient in yaw (Nm/m^2) at each frequency.
    """

    omega: NDArray = field(default_factory=lambda: np.array([]))
    T_surge: NDArray = field(default_factory=lambda: np.array([]))
    T_sway: NDArray = field(default_factory=lambda: np.array([]))
    T_yaw: NDArray = field(default_factory=lambda: np.array([]))


def default_osv_drift_coefficients() -> WaveDriftCoefficients:
    """Simplified drift force coefficients for a typical OSV.

    These are approximate coefficients based on empirical relations for a
    blunt-body vessel (Faltinsen 1990, Ch. 6). They should be replaced
    with data from a hydrodynamic solver (e.g. WAMIT, VERES) for
    production use.

    The surge coefficient uses the Stokes drift reflection approximation:
        T_surge ≈ rho * g * B * T_draft / 2

    Sway and yaw coefficients are scaled proportionally.
    """
    rho, g = 1025.0, 9.81
    B, T_draft, L = 18.0, 5.0, 83.0

    omega = np.linspace(0.2, 2.0, 20)

    # Frequency-dependent scaling (peaks near vessel natural frequencies)
    freq_shape = 1 - np.exp(-3 * omega)

    T_surge = -rho * g * B * T_draft / 2 * freq_shape * np.ones_like(omega)
    T_sway = -rho * g * L * T_draft / 3 * freq_shape * np.ones_like(omega)
    T_yaw = -rho * g * L**2 * T_draft / 24 * freq_shape * np.ones_like(omega)

    return WaveDriftCoefficients(
        omega=omega,
        T_surge=T_surge,
        T_sway=T_sway,
        T_yaw=T_yaw,
    )


def mean_wave_drift_force(
    Hs: float,
    Tp: float,
    beta_wave: float,
    psi: float,
    drift_coeffs: WaveDriftCoefficients | None = None,
    spectrum: str = "jonswap",
    gamma: float = 3.3,
) -> NDArray:
    """Compute mean wave drift forces in the body frame.

    Implements Fossen (2021, Eq. 8.87):

        F_i = 2 * integral{ S(w) * T_i(w, beta_rel) dw }

    for surge, sway, and yaw, projected into the body frame.

    Parameters
    ----------
    Hs : float
        Significant wave height (m). If 0, returns zero forces.
    Tp : float
        Peak period (s).
    beta_wave : float
        Wave propagation direction in NED frame (rad).
    psi : float
        Vessel heading (rad).
    drift_coeffs : WaveDriftCoefficients, optional
        Drift force transfer functions. Uses default OSV values if None.
    spectrum : str
        Wave spectrum type: ``'jonswap'`` or ``'pm'``.
    gamma : float
        JONSWAP peak enhancement factor.

    Returns
    -------
    tau_drift : NDArray
        6-DOF mean wave drift force/moment vector in body frame.
    """
    if Hs <= 0:
        return np.zeros(6)

    if drift_coeffs is None:
        drift_coeffs = default_osv_drift_coefficients()

    omega = drift_coeffs.omega
    if len(omega) == 0:
        return np.zeros(6)

    # Wave spectrum
    if spectrum == "pm":
        S = pierson_moskowitz_spectrum(omega, Hs, Tp)
    else:
        S = jonswap_spectrum(omega, Hs, Tp, gamma)

    # Relative wave heading (encounter angle in body frame)
    beta_rel = beta_wave - psi

    # Direction cosines for projecting NED-aligned drift into body axes
    cb = np.cos(beta_rel)
    sb = np.sin(beta_rel)

    # Numerical integration (trapezoidal rule)
    F_surge = 2.0 * np.trapz(S * drift_coeffs.T_surge, omega) * cb
    F_sway = 2.0 * np.trapz(S * drift_coeffs.T_sway, omega) * sb
    N_yaw = 2.0 * np.trapz(S * drift_coeffs.T_yaw, omega) * sb

    return np.array([F_surge, F_sway, 0.0, 0.0, 0.0, N_yaw])
