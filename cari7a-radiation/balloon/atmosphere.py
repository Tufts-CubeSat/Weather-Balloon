"""U.S. Standard Atmosphere 1976, 0-86 km (geometric).

Only the lower seven layers are needed for balloon work (< ~50 km).
All functions take geometric altitude in metres and accept scalars or numpy arrays.
"""
from __future__ import annotations

import numpy as np

R_AIR = 287.05287  # J/(kg K)
G0 = 9.80665  # m/s^2
R_EARTH_GP = 6356766.0  # m, effective earth radius for geopotential conversion
M_PER_FT = 0.3048

# Layer bases: geopotential altitude (m), lapse rate (K/m)
_H_BASE = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_LAPSE = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
_T_BASE = np.empty(7)
_P_BASE = np.empty(7)
_T_BASE[0], _P_BASE[0] = 288.15, 101325.0
for _i in range(1, 7):
    _dh = _H_BASE[_i] - _H_BASE[_i - 1]
    _L = _LAPSE[_i - 1]
    _T_BASE[_i] = _T_BASE[_i - 1] + _L * _dh
    if _L == 0.0:
        _P_BASE[_i] = _P_BASE[_i - 1] * np.exp(-G0 * _dh / (R_AIR * _T_BASE[_i - 1]))
    else:
        _P_BASE[_i] = _P_BASE[_i - 1] * (_T_BASE[_i] / _T_BASE[_i - 1]) ** (-G0 / (R_AIR * _L))


def geopotential(z_m):
    z = np.asarray(z_m, dtype=float)
    return R_EARTH_GP * z / (R_EARTH_GP + z)


def temperature(z_m):
    """Air temperature (K)."""
    T, _ = _tp(z_m)
    return T


def pressure(z_m):
    """Air pressure (Pa)."""
    _, P = _tp(z_m)
    return P


def density(z_m):
    """Air density (kg/m^3)."""
    T, P = _tp(z_m)
    return P / (R_AIR * T)


def _tp(z_m):
    h = np.clip(geopotential(z_m), 0.0, 84852.0)
    idx = np.clip(np.searchsorted(_H_BASE, h, side="right") - 1, 0, 6)
    L = _LAPSE[idx]
    Tb = _T_BASE[idx]
    Pb = _P_BASE[idx]
    dh = h - _H_BASE[idx]
    T = Tb + L * dh
    with np.errstate(divide="ignore", invalid="ignore"):
        P_grad = Pb * (T / Tb) ** (-G0 / (R_AIR * np.where(L == 0, 1.0, L)))
    P_iso = Pb * np.exp(-G0 * dh / (R_AIR * Tb))
    P = np.where(L == 0.0, P_iso, P_grad)
    if np.ndim(z_m) == 0:
        return float(T), float(P)
    return T, P


def depth_g_cm2(z_m):
    """Atmospheric depth above altitude (g/cm^2), = P/g0 converted."""
    return pressure(z_m) / G0 / 10.0


def altitude_for_pressure(p_pa: float) -> float:
    """Inverse of pressure() by bisection (m)."""
    lo, hi = 0.0, 84000.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pressure(mid) > p_pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
