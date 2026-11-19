"""Integrate CARI-7A rates (computed on an altitude grid) over a flight profile."""
from __future__ import annotations

import numpy as np
import pandas as pd


def interp_rate(alt_ft: np.ndarray, grid: pd.DataFrame, col: str = "rate") -> np.ndarray:
    """Log-linear interpolation in altitude (rates vary roughly exponentially with depth)."""
    x = grid["alt_ft"].to_numpy(dtype=float)
    y = np.clip(grid[col].to_numpy(dtype=float), 1e-30, None)
    return np.exp(np.interp(np.asarray(alt_ft, dtype=float), x, np.log(y)))


def integrate(profile: pd.DataFrame, rates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add per-step rate and cumulative columns to a copy of the profile.

    Dose keys (absorbed_wb, si_*) are in µGy/h  -> cumulative µGy.
    Flux keys (flux_P#) are particles/cm^2/s   -> cumulative fluence particles/cm^2.
    CARI's SIGMA is a model uncertainty and is correlated along the flight, so the
    cumulative sigma is integrated linearly (not in quadrature).
    """
    t_h = profile["t_min"].to_numpy() / 60.0
    alt = profile["alt_ft"].to_numpy()
    cols = {}
    for key, grid in rates.items():
        r = interp_rate(alt, grid, "rate")
        s = interp_rate(alt, grid, "sigma")
        scale = 3600.0 if key.startswith("flux_") else 1.0  # per-second flux -> per-hour
        cols[f"{key}_rate"] = r
        cols[f"{key}_sigma"] = s
        cols[f"{key}_cum"] = _cumtrapz(r * scale, t_h)
        cols[f"{key}_cum_sigma"] = _cumtrapz(s * scale, t_h)
    return pd.concat([profile.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.zeros_like(y, dtype=float)
    c[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return c


def totals(integrated: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{key: (total, sigma)} from the last row of an integrated profile."""
    last = integrated.iloc[-1]
    res = {}
    for col in integrated.columns:
        if col.endswith("_cum") and not col.endswith("_sigma_cum"):
            key = col[:-4]
            res[key] = (float(last[col]), float(last[f"{key}_cum_sigma"]))
    return res
