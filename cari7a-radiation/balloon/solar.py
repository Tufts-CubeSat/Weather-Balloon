"""Solar (sun-heated, black envelope) balloon: ascent, float until sunset, descent.

The envelope is open to ambient pressure. Sunlight heats the air inside it by dT above
ambient, so each cubic metre of envelope lifts
    rho_air(z) * dT / (T_air(z) + dT)  kg.
The balloon floats where that lift times the envelope volume equals the total mass
(envelope + payload). When the sun gets low the heating fades and the balloon sinks.

dT is the main uncertainty. Measured stratospheric solar balloons report roughly
10-40 K, so it is a user input.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from . import atmosphere as atm
from . import sun

CD_ENVELOPE = 0.5


@dataclass
class SolarBalloon:
    envelope_volume_m3: float = 250.0
    envelope_mass_kg: float = 0.8
    payload_mass_kg: float = 1.0
    delta_t_k: float = 30.0
    launch_alt_m: float = 150.0
    lat: float = 44.26
    lon: float = -72.575
    launch_utc: datetime = datetime(2026, 11, 21, 13, 0, tzinfo=timezone.utc)
    min_sun_elevation_deg: float = 0.0  # heating ends when sun is this high above the visible horizon
    descent_rate_mps: float = 4.0
    float_altitude_override_m: float | None = None
    float_hours_override: float | None = None

    @property
    def total_mass(self) -> float:
        return self.envelope_mass_kg + self.payload_mass_kg

    def lift_kg(self, z_m: float) -> float:
        Ta = atm.temperature(z_m)
        return float(atm.density(z_m) * self.envelope_volume_m3 * self.delta_t_k / (Ta + self.delta_t_k))

    def equilibrium_altitude_m(self) -> float:
        if self.lift_kg(self.launch_alt_m) <= self.total_mass:
            raise ValueError(
                f"Not enough lift to leave the ground: {self.lift_kg(self.launch_alt_m):.2f} kg lift vs "
                f"{self.total_mass:.2f} kg mass. Increase the volume or dT, or reduce the mass.")
        lo, hi = self.launch_alt_m, 50000.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.lift_kg(mid) > self.total_mass:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def float_altitude_m(self) -> float:
        return self.float_altitude_override_m or self.equilibrium_altitude_m()

    @staticmethod
    def required_volume_m3(target_alt_m: float, total_mass_kg: float, delta_t_k: float) -> float:
        Ta = atm.temperature(target_alt_m)
        return float(total_mass_kg / (atm.density(target_alt_m) * delta_t_k / (Ta + delta_t_k)))

    def _ascent_rate(self, z: float) -> float:
        fl = self.lift_kg(z) - self.total_mass
        r = (3.0 * self.envelope_volume_m3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        A = np.pi * r * r
        return float(np.sqrt(max(fl, 0.0) * 2.0 * atm.G0 / (atm.density(z) * CD_ENVELOPE * A)))

    def trajectory(self, dt_s: float = 10.0):
        """Return (t_s, z_m, phase, info) for launch -> float -> descent."""
        zf = self.float_altitude_m()
        el0 = sun.solar_elevation_deg(self.launch_utc, self.lat, self.lon)
        info = {"float_alt_m": zf, "sun_elevation_at_launch": el0}
        if self.float_altitude_override_m is None:
            info["equilibrium_alt_m"] = zf

        t, z = 0.0, self.launch_alt_m
        ts, zs, ph = [t], [z], ["ascent"]
        target = zf - 30.0
        while z < target:
            if self.float_altitude_override_m is None:
                v = max(self._ascent_rate(z), 1.0)  # floor avoids asymptotic creep into equilibrium
            else:
                v = 3.0  # overridden altitude: assume a typical solar-balloon climb rate
            z = min(z + v * dt_s, zf)
            t += dt_s
            ts.append(t); zs.append(z); ph.append("ascent")
        z = zf
        reach = self.launch_utc + timedelta(seconds=t)
        info["reach_float_utc"] = reach

        if self.float_hours_override:
            float_s = self.float_hours_override * 3600.0
            info["float_end_source"] = "user override"
        else:
            end = sun.heated_until(reach, self.lat, self.lon, zf, self.min_sun_elevation_deg)
            if end is None:
                float_s = 24 * 3600.0
                info["float_end_source"] = "no sunset within 48 h (polar day), capped at 24 h"
            else:
                float_s = max((end - reach).total_seconds(), 0.0)
                info["float_end_source"] = "sunset at float altitude"
        info["float_hours"] = float_s / 3600.0
        n = int(np.ceil(float_s / 60.0))
        for _ in range(n):  # 1-minute steps at float
            t += 60.0
            ts.append(t); zs.append(z); ph.append("float")
        while z > self.launch_alt_m:
            z = max(z - self.descent_rate_mps * dt_s, self.launch_alt_m)
            t += dt_s
            ts.append(t); zs.append(z); ph.append("descent")
        info["landing_utc"] = self.launch_utc + timedelta(seconds=t)
        return np.array(ts), np.array(zs), np.array(ph), info
