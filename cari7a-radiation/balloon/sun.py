"""Solar elevation (NOAA General Solar Position algorithm) and sunset finding.

Accurate to a fraction of a degree, which is plenty for estimating how long a
solar balloon stays heated.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

R_EARTH_M = 6371000.0


def solar_elevation_deg(when_utc: datetime, lat_deg: float, lon_deg: float) -> float:
    """Geometric solar elevation (deg) without refraction. lon east-positive."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    when_utc = when_utc.astimezone(timezone.utc)
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    days_in_year = 366 if (when_utc.year % 4 == 0 and (when_utc.year % 100 != 0 or when_utc.year % 400 == 0)) else 365
    g = 2 * math.pi / days_in_year * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon_deg  # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    return 90.0 - math.degrees(math.acos(cos_zen))


def horizon_dip_deg(alt_m: float) -> float:
    """How far below the astronomical horizon the visible horizon is, seen from alt_m."""
    if alt_m <= 0:
        return 0.0
    return math.degrees(math.acos(R_EARTH_M / (R_EARTH_M + alt_m)))


def next_crossing(start_utc: datetime, lat: float, lon: float, threshold_deg: float,
                  rising: bool, max_hours: float = 48.0, step_min: float = 1.0) -> datetime | None:
    """First time after start when the sun crosses threshold (rising or setting)."""
    t = start_utc
    prev = solar_elevation_deg(t, lat, lon)
    end = start_utc + timedelta(hours=max_hours)
    while t < end:
        t2 = t + timedelta(minutes=step_min)
        e = solar_elevation_deg(t2, lat, lon)
        if rising and prev < threshold_deg <= e:
            return t2
        if not rising and prev >= threshold_deg > e:
            return t2
        t, prev = t2, e
    return None


def heated_until(start_utc: datetime, lat: float, lon: float, alt_m: float,
                 min_elevation_deg: float = 0.0) -> datetime | None:
    """When the sun drops below `min_elevation_deg` relative to the *visible* horizon at alt_m."""
    thr = min_elevation_deg - horizon_dip_deg(alt_m)
    return next_crossing(start_utc, lat, lon, thr, rising=False)
