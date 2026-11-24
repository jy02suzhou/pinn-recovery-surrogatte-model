"""Material properties and Kennard-Yovanovich rarefied He micro-gap conductance.

Self-contained NumPy / PyTorch compatible physics constants for 3D ESC + 300 mm wafer.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STEFAN_BOLTZMANN: float = 5.670374419e-8  # W / (m^2 * K^4)

# Reference Helium properties at T0 = 300 K, P0 = 760 Torr
T0_HE: float = 300.0
P0_HE_TORR: float = 760.0
LAMBDA0_HE: float = 175e-9   # Mean free path [m]
K0_HE: float = 0.152         # Thermal conductivity [W/(m*K)]
GAMMA_HE: float = 5.0 / 3.0
PR_HE: float = 2.0 / 3.0


@dataclass(frozen=True)
class MaterialProp:
    name: str
    k_ref: float      # Thermal conductivity at 300 K [W/(m*K)]
    emissivity: float = 0.0


MATERIALS: dict[str, MaterialProp] = {
    "Si": MaterialProp("Silicon", k_ref=148.0, emissivity=0.65),
    "Al2O3": MaterialProp("Alumina Ceramic", k_ref=30.0, emissivity=0.0),
    "AlN": MaterialProp("Aluminum Nitride Ceramic", k_ref=160.0, emissivity=0.0),
    "Al": MaterialProp("Aluminum Base", k_ref=167.0, emissivity=0.0),
}


def h_eff_interface(
    P_torr: np.ndarray | float,
    d_gap: float = 10e-6,
    T_g: np.ndarray | float = 315.0,
    h_solid: float = 50.0,
    alpha_T1: float = 0.35,
    alpha_T2: float = 0.35,
) -> np.ndarray | float:
    """Kennard-Yovanovich rarefied He gap + mesa solid contact conductance [W/(m^2*K)]."""
    P_arr = np.maximum(np.asarray(P_torr, dtype=float), 1e-12)
    T_arr = np.asarray(T_g, dtype=float)
    mfp = LAMBDA0_HE * (P0_HE_TORR / P_arr) * (T_arr / T0_HE)
    tac_factor = (2.0 - alpha_T1) / alpha_T1 + (2.0 - alpha_T2) / alpha_T2
    gas_factor = (2.0 * GAMMA_HE / (1.0 + GAMMA_HE)) / PR_HE
    M = tac_factor * gas_factor * mfp
    h_he = K0_HE / (d_gap + M)
    res = h_solid + h_he
    return float(res) if (np.ndim(P_torr) == 0 and np.ndim(T_g) == 0) else res
