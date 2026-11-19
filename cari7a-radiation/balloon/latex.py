"""Helium (or hydrogen) latex sounding balloon: ascent, burst and parachute descent.

Physics (the same approach as the classic HabHub burst calculator):
  * Gas mass is fixed and stays at ambient temperature and pressure (zero superpressure),
    so volume scales as V(z) = V0 * (P0/P(z)) * (T(z)/T0).
  * The net free lift (kg) is constant with altitude to first order, because both the
    displaced air mass and the gas mass are fixed.
  * Ascent speed: free lift balances drag, v = sqrt(2 g FL / (rho_air Cd A(z))).
  * Burst happens when the balloon diameter reaches the manufacturer burst diameter.
  * Descent under a parachute: v = sqrt(2 m g / (rho_air Cd_p A_p)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import atmosphere as atm

# Nominal burst diameters (m) for Totex TA-series balloons (the table used by the HabHub
# burst calculator). Kaymont/Hwoyee balloons of the same mass are similar.
BURST_DIAMETER_M = {
    100: 1.96, 200: 3.00, 300: 3.78, 350: 4.12, 450: 4.72, 500: 4.99, 600: 6.02,
    700: 6.53, 800: 7.00, 1000: 7.86, 1200: 8.63, 1500: 9.44, 2000: 10.54, 3000: 13.00,
}
BALLOON_CD = 0.25  # drag coefficient of a latex sphere in ascent (HabHub uses 0.25-0.3)

GAS_MOLAR_MASS = {"helium": 0.0040026, "hydrogen": 0.002016}  # kg/mol
R_UNIVERSAL = 8.314462618


@dataclass
class LatexBalloon:
    balloon_mass_g: float = 1200.0
    payload_mass_kg: float = 1.5  # everything below the balloon: payload + parachute + line
    neck_lift_kg: float | None = None  # gross lift minus balloon mass; None -> use target ascent
    target_ascent_mps: float = 5.0
    gas: str = "helium"
    burst_diameter_m: float | None = None  # override table value
    parachute_diameter_m: float = 1.2
    parachute_cd: float = 1.3
    burst_altitude_override_m: float | None = None
    launch_alt_m: float = 150.0

    # ---- derived quantities -------------------------------------------------
    @property
    def burst_d(self) -> float:
        if self.burst_diameter_m:
            return self.burst_diameter_m
        if self.balloon_mass_g in BURST_DIAMETER_M:
            return BURST_DIAMETER_M[self.balloon_mass_g]
        masses = np.array(sorted(BURST_DIAMETER_M))
        diams = np.array([BURST_DIAMETER_M[m] for m in masses])
        return float(np.interp(self.balloon_mass_g, masses, diams))

    def _gas_density_ratio(self) -> float:
        """rho_gas / rho_air at equal T and P."""
        return GAS_MOLAR_MASS[self.gas] / 0.0289644

    def _volume_for_free_lift(self, free_lift_kg: float, z0: float) -> float:
        gross = free_lift_kg + self.payload_mass_kg + self.balloon_mass_g / 1000.0
        rho_a = atm.density(z0)
        return gross / (rho_a * (1.0 - self._gas_density_ratio()))

    def _ascent_rate(self, free_lift_kg: float, V: float, z: float) -> float:
        r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
        A = np.pi * r * r
        return float(np.sqrt(2.0 * free_lift_kg * atm.G0 / (atm.density(z) * BALLOON_CD * A)))

    def free_lift_kg(self) -> float:
        z0 = self.launch_alt_m
        if self.neck_lift_kg is not None:
            return self.neck_lift_kg - self.payload_mass_kg
        # solve for the free lift that gives the target ascent rate at launch
        lo, hi = 1e-4, 50.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            V = self._volume_for_free_lift(mid, z0)
            if self._ascent_rate(mid, V, z0) < self.target_ascent_mps:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def launch_volume_m3(self) -> float:
        return self._volume_for_free_lift(self.free_lift_kg(), self.launch_alt_m)

    def burst_altitude_m(self) -> float:
        if self.burst_altitude_override_m:
            return self.burst_altitude_override_m
        z0 = self.launch_alt_m
        V0 = self.launch_volume_m3()
        Vb = np.pi / 6.0 * self.burst_d**3
        P0, T0 = atm.pressure(z0), atm.temperature(z0)
        lo, hi = z0, 50000.0
        if V0 >= Vb:
            return z0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            V = V0 * (P0 / atm.pressure(mid)) * (atm.temperature(mid) / T0)
            if V < Vb:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ---- trajectory ----------------------------------------------------------
    def trajectory(self, dt_s: float = 10.0):
        """Return (t_s, z_m, phase) arrays for launch -> burst -> landing."""
        FL = self.free_lift_kg()
        if FL <= 0:
            raise ValueError("Neck lift is less than the payload mass: the balloon will not rise.")
        z0 = self.launch_alt_m
        V0 = self.launch_volume_m3()
        P0, T0 = atm.pressure(z0), atm.temperature(z0)
        zb = self.burst_altitude_m()

        t, z = 0.0, z0
        ts, zs, ph = [t], [z], ["ascent"]
        while z < zb:
            V = V0 * (P0 / atm.pressure(z)) * (atm.temperature(z) / T0)
            v = self._ascent_rate(FL, V, z)
            z = min(z + v * dt_s, zb)
            t += dt_s
            ts.append(t); zs.append(z); ph.append("ascent")
        ph[-1] = "burst"
        A_p = np.pi * (self.parachute_diameter_m / 2.0) ** 2
        m = self.payload_mass_kg
        while z > z0:
            v = np.sqrt(2.0 * m * atm.G0 / (atm.density(z) * self.parachute_cd * A_p))
            z = max(z - v * dt_s, z0)
            t += dt_s
            ts.append(t); zs.append(z); ph.append("descent")
        return np.array(ts), np.array(zs), np.array(ph)
