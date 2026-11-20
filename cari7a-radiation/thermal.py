"""Lumped-capacitance thermal model of the payload box and the film package.

Thermal parts (each "node" is one lumped temperature):
  * electronics + batteries (+ the inner half of the foam): heated by the DC power the devices draw
    from the battery, until the battery's energy runs out. Nearly all of that power ends up as heat
    inside the box (only the radiated RF leaves, which is small).
  * box interior air / inner wall: massless node between the electronics and the foam. Electronics
    lose heat to it by natural convection inside the box (h ~ 2.5 W/m2K at sea level, falling as
    sqrt(pressure) because the box is vented) plus internal infrared radiation. So electronics run
    warmer than the box air, and more so at altitude. This internal resistance is often as large as
    the foam wall's, and it lengthens the cooling time constant.
  * foam wall: R = t / (k A) from the interior air to the box's outer surface.
  * film package: film + container. Either outside the box with its own outer surface, or
    inside the box, coupled to the interior air.
  * outer surfaces have no heat capacity: each step they are solved from an energy balance of
    conduction from inside + absorbed sun + absorbed albedo + absorbed IR from earth and sky,
    against convection and emitted IR.

Environment along the flight profile:
  * air T and density: US Standard Atmosphere 1976, with the surface temperature shifted to the
    user's value. The shift tapers linearly to zero at the 11 km tropopause.
  * direct sun: 1361 W/m^2 attenuated by the air above (Meinel air-mass model, scaled by the
    pressure ratio), present only when the sun is above the visible horizon from that altitude.
  * albedo: ground albedo x horizontal direct irradiance, seen by the lower half of the body.
  * IR from below: blends from the ground's blackbody flux (on the ground) to 240 W/m^2 (top-of-
    atmosphere outgoing longwave radiation) by the pressure ratio. IR from above: clear-sky air
    emissivity ~0.75 at the surface, scaled down by the pressure ratio.
  * convection: Ranz-Marshall sphere correlation, Nu = 2 + 0.6 Re^0.5 Pr^(1/3). The air speed
    past the payload is |vertical speed| with a floor for swinging/wind.
  * projected area of a convex body averaged over orientation = A/4. That is used for the sun
    because the payload spins.

Accuracy: about +/-10-15 C. Use it to compare scenarios and spot risks, not to predict exact
temperatures.

Checks against published data:
  * Isenberg et al. (AHAC 2012) and Meduvsky, Howeth, Katan & Isenberg (AHAC 2013), Embry-Riddle:
    lumped R-C models of balloon gondolas validated on flights. Thermal resistance rises
    significantly at low pressure, while capacitance stays roughly constant. Here that comes from
    the sqrt(p) internal convection and the density-dependent outer convection.
  * SparkFun freezer test: a 1 in foam box cooled from 25 to 0 C in ~35 min at about -18 C. This
    model's time constant for a 1 in wall is ~35 min.
  * Safonova et al. 2015 (arXiv:1506.01538): an under-insulated Raspberry Pi failed ~40 min into a
    flight. Their -22 C cold-chamber test showed foam slows cooling but doesn't stop it.
  * Crowe et al., GSBC 2015 (UNSW): a sun-heated transparent PET container bottomed at -12 C
    with -60 C outside. White foam reflects most of the sun, so it gains less solar heat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from balloon import atmosphere as atm
from balloon import sun as sunmod

SIGMA = 5.670374e-8
S0 = 1361.0
OLR = 240.0


# ----------------------------------------------------------------------------------- specs
@dataclass
class Device:
    name: str
    power_w: float  # power while on
    duty: float  # fraction of time on (0-1)

    @property
    def avg_w(self) -> float:
        return self.power_w * self.duty


DEFAULT_DEVICES = [  # DC input power drawn from the battery (not RF output). Measure yours!
    Device("SPOT GPS tracker", 0.05, 1.0),
    Device("Radio transmitter (DC in while transmitting)", 2.5, 0.2),
    Device("Avionics board (radio + camera control)", 1.5, 1.0),
    Device("Camera", 1.0, 0.5),
]

BATTERIES = {  # name: (specific heat J/kgK, lowest usable temperature C, usable energy Wh/kg)
    "Lithium primary AA (e.g. Energizer L91)": (900.0, -40.0, 230.0),
    "Li-ion / LiPo rechargeable": (1000.0, -20.0, 170.0),
    "Alkaline": (1000.0, -18.0, 90.0),
}


@dataclass
class BoxSpec:
    length_cm: float = 30.0
    width_cm: float = 25.0
    height_cm: float = 20.0
    wall_cm: float = 1.27  # 1/2 in
    foam_k: float = 0.035  # W/mK (EPS ~0.033-0.038, XPS ~0.029)
    foam_density: float = 20.0  # kg/m^3
    absorptivity: float = 0.3  # white foam in sunlight
    emissivity: float = 0.9
    devices: list[Device] = field(default_factory=lambda: list(DEFAULT_DEVICES))
    electronics_mass_kg: float = 0.3
    battery_mass_kg: float = 0.2
    battery_type: str = "Lithium primary AA (e.g. Energizer L91)"
    battery_wh: float = 0.0  # usable energy; 0 = estimate from battery mass x type
    component_area_m2: float = 0.0  # exposed surface of boards/batteries; 0 = estimate from mass
    component_emissivity: float = 0.85

    @property
    def battery_energy_wh(self) -> float:
        if self.battery_wh > 0:
            return self.battery_wh
        return self.battery_mass_kg * BATTERIES[self.battery_type][2]

    @property
    def component_area(self) -> float:
        if self.component_area_m2 > 0:
            return self.component_area_m2
        # boards, camera bodies and battery packs: roughly 0.2 m2 of exposed surface per kg
        return min(max(0.2 * (self.electronics_mass_kg + self.battery_mass_kg), 0.01), 0.8 * self.inner_area)

    def internal_conductance(self, T_k: float, p_ratio: float) -> float:
        """W/K from the electronics to the box interior air/inner wall (convection + IR)."""
        h_nat = 2.5 * math.sqrt(max(p_ratio, 1e-4))
        eps = 1.0 / (1.0 / self.component_emissivity + 1.0 / 0.9 - 1.0)
        h_rad = 4.0 * eps * SIGMA * T_k ** 3
        return self.component_area * (h_nat + h_rad)

    @property
    def outer_area(self) -> float:
        l, w, h = self.length_cm / 100, self.width_cm / 100, self.height_cm / 100
        return 2 * (l * w + l * h + w * h)

    @property
    def inner_area(self) -> float:
        t = 2 * self.wall_cm / 100
        l, w, h = (max(d / 100 - t, 0.01) for d in (self.length_cm, self.width_cm, self.height_cm))
        return 2 * (l * w + l * h + w * h)

    @property
    def foam_mass_kg(self) -> float:
        return 0.5 * (self.outer_area + self.inner_area) * self.wall_cm / 100 * self.foam_density

    @property
    def avg_power_w(self) -> float:
        return sum(d.avg_w for d in self.devices)

    @property
    def wall_resistance(self) -> float:
        """K/W through the foam (mean of inner/outer area)."""
        a = 0.5 * (self.outer_area + self.inner_area)
        return (self.wall_cm / 100) / (self.foam_k * a)

    @property
    def heat_capacity(self) -> float:
        cp_b = BATTERIES[self.battery_type][0]
        return self.electronics_mass_kg * 800.0 + self.battery_mass_kg * cp_b + 0.5 * self.foam_mass_kg * 1300.0

    @property
    def mass_kg(self) -> float:
        return self.foam_mass_kg + self.electronics_mass_kg + self.battery_mass_kg


CONTAINER_MATERIALS = {  # density g/cm3, cp J/kgK, radiation-shield equivalent
    "Polypropylene (plastic can)": (0.90, 1900.0, "hdpe"),
    "Aluminum / tinplate (metal can)": (2.70, 900.0, "al"),
}

FILM_PACKAGES = {
    "100 ft 35mm daylight spool, black plastic can": dict(
        diameter_cm=10.0, height_cm=4.5, film_mass_kg=0.22, container_material="Polypropylene (plastic can)",
        container_wall_mm=1.5, absorptivity=0.95, emissivity=0.90),
    "400 ft 35mm core load, bare metal can": dict(
        diameter_cm=18.5, height_cm=4.5, film_mass_kg=0.85, container_material="Aluminum / tinplate (metal can)",
        container_wall_mm=0.5, absorptivity=0.25, emissivity=0.10),
    "400 ft 35mm core load, black-painted metal can": dict(
        diameter_cm=18.5, height_cm=4.5, film_mass_kg=0.85, container_material="Aluminum / tinplate (metal can)",
        container_wall_mm=0.5, absorptivity=0.95, emissivity=0.90),
    "35mm cassette in plastic canister": dict(
        diameter_cm=3.2, height_cm=5.2, film_mass_kg=0.02, container_material="Polypropylene (plastic can)",
        container_wall_mm=1.0, absorptivity=0.5, emissivity=0.9),
}


@dataclass
class FilmPackage:
    location: str = "outside"  # "outside" or "inside"
    diameter_cm: float = 10.0
    height_cm: float = 4.5
    film_mass_kg: float = 0.22
    container_material: str = "Polypropylene (plastic can)"
    container_wall_mm: float = 1.5
    absorptivity: float = 0.95
    emissivity: float = 0.90
    insulation_cm: float = 0.0  # foam / bubble wrap around the can
    insulation_k: float = 0.035
    insulation_density: float = 20.0

    @property
    def area(self) -> float:
        r = (self.diameter_cm / 2 + self.insulation_cm) / 100
        h = (self.height_cm + 2 * self.insulation_cm) / 100
        return 2 * math.pi * r * r + 2 * math.pi * r * h

    @property
    def can_area(self) -> float:
        r, h = self.diameter_cm / 200, self.height_cm / 100
        return 2 * math.pi * r * r + 2 * math.pi * r * h

    @property
    def container_mass_kg(self) -> float:
        rho = CONTAINER_MATERIALS[self.container_material][0] * 1000
        return self.can_area * self.container_wall_mm / 1000 * rho

    @property
    def insulation_mass_kg(self) -> float:
        return 0.5 * (self.area + self.can_area) * self.insulation_cm / 100 * self.insulation_density

    @property
    def mass_kg(self) -> float:
        return self.film_mass_kg + self.container_mass_kg + self.insulation_mass_kg

    @property
    def heat_capacity(self) -> float:
        cp_c = CONTAINER_MATERIALS[self.container_material][1]
        return self.film_mass_kg * 1300.0 + self.container_mass_kg * cp_c + self.insulation_mass_kg * 1300.0

    @property
    def insulation_resistance(self) -> float:
        if self.insulation_cm <= 0:
            return 1e-3  # can wall + film contact: effectively well coupled
        a = 0.5 * (self.area + self.can_area)
        return (self.insulation_cm / 100) / (self.insulation_k * a)

    def shielding_g_cm2(self) -> dict[str, float]:
        """Areal density of the can wall + insulation, for CARI's VEHICLE.DAT."""
        rho, _, kind = CONTAINER_MATERIALS[self.container_material]
        wall = rho * self.container_wall_mm / 10.0
        ins = self.insulation_density / 1000 * self.insulation_cm
        out = {"al": 0.0, "hdpe": ins}
        out[kind] += wall
        return out


@dataclass
class Environment:
    surface_temp_c: float = 2.0
    ground_rh: float = 0.7
    ground_albedo: float = 0.3
    start_temp_c: float = 20.0  # payload temperature when it leaves the building / car
    prelaunch_min: float = 45.0
    postlanding_min: float = 120.0
    ground_wind_mps: float = 3.0
    min_airspeed_mps: float = 1.0  # swinging / shear when balloon moves with the wind


@dataclass
class FilmLimits:
    """Editable thresholds. Defaults follow Kodak handling guidance (see THERMAL_NOTES)."""
    cold_handling_c: float = -18.0  # 0 F: Kodak's cold-weather camera practices start here
    severe_cold_c: float = -40.0
    warm_storage_c: float = 13.0  # Kodak: store unprocessed film below 13 C (55 F)
    heat_c: float = 32.0
    transported_in_flight: bool = False  # True if the film is advanced/wound during the flight


THERMAL_NOTES = """\
How to read the film temperature results

- Cold does not damage the latent image or add fog. Kodak's storage advice is to keep
  unprocessed film cold. The cold risks are mechanical and electrostatic:
    * Triacetate film base gets brittle and curls in deep cold, so it can crack or tear if it is
      wound, bent or advanced.
    * Very dry cold air plus film moving over rollers causes static discharge marks
      (branched or dotted exposures). That only matters if the film is advanced during the
      flight.
    * Kodak's cold-weather filming advice starts around 0 F (-18 C): camera lubricants and
      batteries suffer, and film must be handled slowly.
- Condensation / frost: if film colder than the dew point meets moist air (descent through
  the lower atmosphere, recovery, or opening the can indoors), water condenses on it. That can
  spot or stick the emulsion. Keep the can sealed (tape the lid) and let it warm to room
  temperature before opening, typically 1-3 hours for a can.
- Heat: Kodak recommends storing unprocessed film below 13 C (55 F). Exposed or unexposed
  stock left warm for long (days) slowly gains fog and shifts in speed and color balance.
  A few hours above ~32 C (a sunny car or field) is the realistic heat risk, and it happens
  on the ground, not aloft.
Thresholds are editable defaults, not manufacturer specifications.
"""


# ----------------------------------------------------------------------------- physics
def _air_props(T: float) -> tuple[float, float]:
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    k = 2.64638e-3 * T ** 1.5 / (T + 245.4 * 10 ** (-12.0 / T))
    return mu, k


def h_conv(T_air: float, rho: float, v: float, D: float) -> float:
    mu, k = _air_props(T_air)
    Re = rho * v * D / mu
    Nu = 2.0 + 0.6 * math.sqrt(Re) * 0.71 ** (1 / 3)
    return Nu * k / D


def direct_sun(elev_deg: float, alt_m: float, p_ratio: float) -> float:
    if elev_deg <= -sunmod.horizon_dip_deg(alt_m):
        return 0.0
    s = math.sin(math.radians(max(elev_deg, 0.5)))
    am = p_ratio / s
    return S0 * 0.7 ** (am ** 0.678)


def dew_point_c(T_c: float, rh: float) -> float:
    a, b = 17.62, 243.12
    g = math.log(max(rh, 1e-3)) + a * T_c / (b + T_c)
    return b * g / (a - g)


def _solve_surface(Ti, R, A, alpha, eps, q_solar, e_up, e_sky, h, Ta, Ts0):
    """Massless surface node: conduction from inside + absorbed = convection + emission."""
    q_abs = alpha * q_solar + eps * (e_up + e_sky) * A / 2
    Ts = Ts0
    for _ in range(30):
        g = (Ti - Ts) / R + q_abs + h * A * (Ta - Ts) - eps * SIGMA * A * Ts ** 4
        dg = -1.0 / R - h * A - 4 * eps * SIGMA * A * Ts ** 3
        step = g / dg
        Ts -= step
        if abs(step) < 1e-4:
            break
    return Ts


def simulate(profile: pd.DataFrame, box: BoxSpec, film: FilmPackage, env: Environment,
             dt_s: float = 10.0) -> pd.DataFrame:
    """Run the thermal model along a flight profile (with ground holds added before/after)."""
    t_prof = profile["t_min"].to_numpy() * 60.0
    z_prof = profile["alt_m"].to_numpy()
    z0, z_end = z_prof[0], z_prof[-1]
    pre, post = env.prelaunch_min * 60.0, env.postlanding_min * 60.0
    launch_utc = profile["utc"].iloc[0]
    total = pre + t_prof[-1] + post
    times = np.arange(-pre, t_prof[-1] + post + dt_s / 2, dt_s)

    T_launch_std = float(atm.temperature(z0))
    offset = (env.surface_temp_c + 273.15) - T_launch_std
    P0 = float(atm.pressure(0.0))
    T_ground_k = env.surface_temp_c + 273.15
    e_ground = SIGMA * T_ground_k ** 4
    dew_ground = dew_point_c(env.surface_temp_c, env.ground_rh)

    # solar elevation on a coarse grid (every 2 min) then interpolate
    grid = np.arange(times[0], times[-1] + 121, 120.0)
    from datetime import timedelta
    elev_grid = np.array([sunmod.solar_elevation_deg(launch_utc + timedelta(seconds=float(s)),
                                                     float(profile["lat"].iloc[0]), float(profile["lon"].iloc[0]))
                          for s in grid])

    def z_at(t):
        if t < 0:
            return z0
        if t > t_prof[-1]:
            return z_end
        return float(np.interp(t, t_prof, z_prof))

    D_box = math.sqrt(box.outer_area / math.pi)
    D_film = math.sqrt(film.area / math.pi)
    Ti = Tf = env.start_temp_c + 273.15
    Ts_box = Ts_film = Ti
    Cb, Cf = box.heat_capacity, film.heat_capacity
    P_avg = box.avg_power_w
    energy_j = box.battery_energy_wh * 3600.0
    Rw = box.wall_resistance

    rows = []
    z_prev = z_at(times[0])
    for t in times:
        z = z_at(t)
        v = abs(z - z_prev) / dt_s
        z_prev = z
        on_ground = t < 0 or t > t_prof[-1]
        Ta = float(atm.temperature(z)) + offset * max(0.0, 1.0 - z / 11000.0)
        P_z = float(atm.pressure(z))
        rho = P_z / (atm.R_AIR * Ta)
        pr = P_z / P0
        elev = float(np.interp(t, grid, elev_grid))
        I = direct_sun(elev, z, pr)
        horiz = I * max(math.sin(math.radians(elev)), 0.0)
        e_up = pr * e_ground + (1 - pr) * OLR if not on_ground else e_ground
        e_sky = 0.75 * pr * SIGMA * Ta ** 4
        speed = max(v, env.ground_wind_mps if on_ground else env.min_airspeed_mps)
        P = P_avg if energy_j > 0 else 0.0
        energy_j -= P * dt_s

        # interior network: electronics --G_in-- air node --Rw-- outer surface;
        # film (if inside) --G_f-- air node. Reduced to a Thevenin source for the surface solve.
        G_in = box.internal_conductance(Ti, pr)
        if film.location == "inside":
            h_f = 2.5 * math.sqrt(pr) + 4.0 * 0.8 * SIGMA * Tf ** 3
            G_f = 1.0 / (1.0 / (h_f * film.area) + film.insulation_resistance)
        else:
            G_f = 0.0
        T_eq = (G_in * Ti + G_f * Tf) / (G_in + G_f)
        R_eq = 1.0 / (G_in + G_f) + Rw

        Ab = box.outer_area
        qsol_b = I * Ab / 4 + env.ground_albedo * horiz * Ab / 4 * (0.5 if on_ground else 1.0)
        hb = h_conv(Ta, rho, speed, D_box)
        Ts_box = _solve_surface(T_eq, R_eq, Ab, box.absorptivity, box.emissivity, qsol_b, e_up, e_sky, hb, Ta, Ts_box)
        q_wall = (T_eq - Ts_box) / R_eq
        T_air_in = Ts_box + q_wall * Rw
        q_in = G_in * (Ti - T_air_in)

        if film.location == "outside":
            Af = film.area
            qsol_f = I * Af / 4 + env.ground_albedo * horiz * Af / 4 * (0.5 if on_ground else 1.0)
            hf = h_conv(Ta, rho, speed, D_film)
            Ts_film = _solve_surface(Tf, film.insulation_resistance, Af, film.absorptivity, film.emissivity,
                                     qsol_f, e_up, e_sky, hf, Ta, Ts_film)
            q_film = (Tf - Ts_film) / film.insulation_resistance
        else:
            q_film = G_f * (Tf - T_air_in)
            Ts_film = Tf

        rows.append((t, z, Ta, Ti, T_air_in, Ts_box, Tf, Ts_film, I, elev, P, max(energy_j, 0.0) / 3600.0))
        Ti += dt_s * (P - q_in) / Cb
        Tf += dt_s * (-q_film) / Cf

    df = pd.DataFrame(rows, columns=["t_s", "alt_m", "T_air", "T_box", "T_box_air", "T_box_surface", "T_film",
                                     "T_film_surface", "sun_wm2", "sun_elev", "power_w", "battery_wh_left"])
    for c in ["T_air", "T_box", "T_box_air", "T_box_surface", "T_film", "T_film_surface"]:
        df[c] = df[c] - 273.15
    df["t_min"] = df["t_s"] / 60.0
    df["alt_ft"] = df["alt_m"] / atm.M_PER_FT
    df["phase"] = np.where(df["t_s"] < 0, "pre-launch", np.where(df["t_s"] > t_prof[-1], "post-landing", "flight"))
    df["dew_point_c"] = dew_ground  # boundary-layer dew point (used below ~3 km)
    return df


# ------------------------------------------------------------------------ film assessment
def assess_film_thermal(sim: pd.DataFrame, box: BoxSpec, limits: FilmLimits) -> dict:
    dt_h = np.diff(sim["t_s"].to_numpy(), prepend=sim["t_s"].iloc[0]) / 3600.0
    tf = sim["T_film"].to_numpy()
    flight = (sim["phase"] == "flight").to_numpy()

    def hours(mask):
        return float(dt_h[mask].sum())

    low_moist = sim["alt_m"].to_numpy() < 3000
    descent_or_after = sim["t_s"].to_numpy() > sim.loc[sim["alt_m"].idxmax(), "t_s"]
    cond_mask = low_moist & descent_or_after & (sim["T_film_surface"].to_numpy() < sim["dew_point_c"].to_numpy())
    batt_min = BATTERIES[box.battery_type][1]
    res = {
        "film_min_c": float(tf.min()),
        "film_max_c": float(tf.max()),
        "film_min_flight_c": float(tf[flight].min()) if flight.any() else float("nan"),
        "box_min_c": float(sim["T_box"].min()),
        "box_max_c": float(sim["T_box"].max()),
        "air_min_c": float(sim["T_air"].min()),
        "h_below_cold": hours(tf < limits.cold_handling_c),
        "h_below_severe": hours(tf < limits.severe_cold_c),
        "h_above_storage": hours(tf > limits.warm_storage_c),
        "h_above_heat": hours(tf > limits.heat_c),
        "h_condensation": hours(cond_mask),
        "h_battery_too_cold": hours(sim["T_box"].to_numpy() < batt_min),
        "battery_min_rating_c": batt_min,
        "box_air_min_c": float(sim["T_box_air"].min()),
        "battery_empty_h": (float(sim.loc[sim["power_w"] <= 0, "t_s"].iloc[0]) / 3600.0
                            if (sim["power_w"] <= 0).any() and box.avg_power_w > 0 else None),
        "dew_point_c": float(sim["dew_point_c"].iloc[0]),
    }
    risks = []

    def add(level, title, detail):
        risks.append({"level": level, "risk": title, "detail": detail})

    if res["film_min_c"] < limits.severe_cold_c:
        add("caution" if not limits.transported_in_flight else "high",
            "Severe cold / brittle base",
            f"Film reaches {res['film_min_c']:.0f} °C ({res['h_below_severe']:.1f} h below {limits.severe_cold_c:.0f} °C). "
            "Don't flex, wind or open it while cold. Let it warm slowly in the sealed can.")
    elif res["film_min_c"] < limits.cold_handling_c:
        add("caution" if limits.transported_in_flight else "low", "Cold handling",
            f"Film reaches {res['film_min_c']:.0f} °C ({res['h_below_cold']:.1f} h below {limits.cold_handling_c:.0f} °C). "
            "Harmless to the latent image. Handle gently until warm.")
    else:
        add("ok", "Cold", f"Film stays above {limits.cold_handling_c:.0f} °C.")
    if limits.transported_in_flight and res["h_below_cold"] > 0:
        add("high", "Static marks / tearing while advancing",
            "The film is advanced while below the cold-handling limit in very dry air. Static discharge "
            "marks are likely to show far more than any radiation effect.")
    if res["h_condensation"] > 0:
        add("caution", "Condensation / frost on descent or recovery",
            f"Film surface is below the ground dew point ({res['dew_point_c']:.0f} °C) for {res['h_condensation']:.1f} h "
            "in moist low-altitude air. Seal the can airtight, tape the lid, and add a desiccant packet. "
            "Don't open it until it's at room temperature.")
    else:
        add("ok", "Condensation", "Film warms above the dew point before or soon after landing.")
    if res["h_above_heat"] > 0:
        add("caution", "Heat", f"{res['h_above_heat']:.1f} h above {limits.heat_c:.0f} °C (peak {res['film_max_c']:.0f} °C), "
            "most likely during a ground hold in the sun.")
    elif res["h_above_storage"] > 0:
        add("low", "Warm storage", f"{res['h_above_storage']:.1f} h above {limits.warm_storage_c:.0f} °C. Negligible over hours. "
            "Keep the film cool before and after the flight.")
    else:
        add("ok", "Heat", "Film never gets warm.")
    if res["h_battery_too_cold"] > 0:
        add("high", "Batteries too cold (inside box)",
            f"Box interior goes below the {batt_min:.0f} °C rating of {box.battery_type} for {res['h_battery_too_cold']:.1f} h. "
            "Expect tracker/radio dropouts. Use lithium primaries, add insulation, or add a heater.")
    if res["battery_empty_h"] is not None:
        add("high", "Battery runs out",
            f"At {box.avg_power_w:.2f} W the {box.battery_energy_wh:.0f} Wh battery is empty "
            f"{res['battery_empty_h']:.1f} h after launch (ground holds included). The electronics stop heating "
            "the box after that. Cold also cuts usable capacity, especially for Li-ion and alkaline.")
    return {**res, "risks": risks}
