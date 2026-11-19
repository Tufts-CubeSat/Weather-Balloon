"""Turn balloon trajectories into a common flight-profile DataFrame."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .atmosphere import M_PER_FT, depth_g_cm2


def to_profile(t_s, z_m, phase, launch_utc: datetime, lat: float, lon: float) -> pd.DataFrame:
    t_s = np.asarray(t_s, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    df = pd.DataFrame({
        "t_min": t_s / 60.0,
        "alt_m": z_m,
        "alt_ft": z_m / M_PER_FT,
        "depth_g_cm2": depth_g_cm2(z_m),
        "phase": phase,
        "lat": lat,
        "lon": lon,
    })
    df["utc"] = [launch_utc + timedelta(seconds=float(s)) for s in t_s]
    return df


def summarize(profile: pd.DataFrame) -> dict:
    hi = profile["alt_ft"] >= 60000
    dt = np.diff(profile["t_min"].to_numpy(), prepend=profile["t_min"].iloc[0])
    return {
        "duration_h": profile["t_min"].iloc[-1] / 60.0,
        "max_alt_ft": float(profile["alt_ft"].max()),
        "hours_above_60kft": float(dt[hi.to_numpy()].sum() / 60.0),
    }
