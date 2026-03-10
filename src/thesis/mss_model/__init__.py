"""
OSV 6-DOF simulation package — ``mss_model``.

Translated from the MSS toolbox (Fossen 2021).
"""

from .vessel import OSV, OSVParams
from .simulation import SimConfig, simulate_osv
from .environment import (
    WaveDriftCoefficients,
    mean_wave_drift_force,
    jonswap_spectrum,
    pierson_moskowitz_spectrum,
    ocean_current,
)
from .control import PIDNonlinearMIMO
from .thruster import thruster_config, alloc_pseudoinverse, optimal_alloc
from .gnc import ssa, rk4, eulerang, Rzyx, Tzyx, smtrx, hmtrx

__all__ = [
    "OSV",
    "OSVParams",
    "SimConfig",
    "simulate_osv",
    "WaveDriftCoefficients",
    "mean_wave_drift_force",
    "jonswap_spectrum",
    "pierson_moskowitz_spectrum",
    "ocean_current",
    "PIDNonlinearMIMO",
    "thruster_config",
    "alloc_pseudoinverse",
    "optimal_alloc",
    "ssa",
    "rk4",
    "eulerang",
    "Rzyx",
    "Tzyx",
    "smtrx",
    "hmtrx",
]
