from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from balloon.latex import LatexBalloon
from balloon.profile import to_profile
from balloon.solar import SolarBalloon
from thermal import (BoxSpec, Device, Environment, FilmLimits, FilmPackage, assess_film_thermal,
                     dew_point_c, h_conv, simulate)

NIGHT = datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc)  # ~1 am local in Vermont, no sun
DAY = datetime(2026, 11, 21, 13, 0, tzinfo=timezone.utc)


def ground_profile(hours=8.0, when=NIGHT):
    t = np.arange(0, hours * 3600 + 1, 60.0)
    return to_profile(t, np.full_like(t, 150.0), ["x"] * len(t), when, 44.26, -72.575)


def test_unpowered_box_relaxes_toward_ambient():
    env = Environment(surface_temp_c=0.0, start_temp_c=20.0, prelaunch_min=0, postlanding_min=0)
    box = BoxSpec(devices=[])
    sim = simulate(ground_profile(4), box, FilmPackage(location="inside"), env)  # 1-5 am local: no sun
    end = sim.iloc[-1]
    # radiative cooling to a clear night sky pulls things slightly below air temperature
    assert -8.0 < end["T_box"] < 1.0
    assert abs(end["T_film"] - end["T_box"]) < 2.0


def test_power_warms_box_by_power_times_resistance():
    env = Environment(surface_temp_c=0.0, start_temp_c=0.0, prelaunch_min=0, postlanding_min=0)
    box0 = BoxSpec(devices=[])
    box5 = BoxSpec(devices=[Device("heater", 5.0, 1.0)])
    film = FilmPackage()
    a = simulate(ground_profile(4), box0, film, env).iloc[-1]
    b = simulate(ground_profile(4), box5, film, env).iloc[-1]
    rise = (b["T_box"] - b["T_box_surface"])
    assert rise == pytest.approx(5.0 * box5.wall_resistance, rel=0.05)
    assert b["T_box"] > a["T_box"] + 3


def test_insulation_slows_film_cooling_in_flight():
    prof = _helium()
    env = Environment()
    bare = simulate(prof, BoxSpec(), FilmPackage(insulation_cm=0.0), env)
    wrapped = simulate(prof, BoxSpec(), FilmPackage(insulation_cm=3.0), env)
    assert wrapped["T_film"].min() > bare["T_film"].min() + 3


def test_low_emissivity_can_stays_warmer_at_night_float():
    s = SolarBalloon(launch_utc=DAY, float_altitude_override_m=24000, float_hours_override=3)
    t, z, ph, _ = s.trajectory()
    # shift launch to night so float is dark
    prof = to_profile(t, z, ph, NIGHT, 44.26, -72.575)
    env = Environment()
    black = simulate(prof, BoxSpec(), FilmPackage(absorptivity=0.95, emissivity=0.9), env)
    shiny = simulate(prof, BoxSpec(), FilmPackage(absorptivity=0.2, emissivity=0.05), env)
    assert shiny["T_film"].min() > black["T_film"].min()


def test_assessment_flags_and_counts():
    sim = simulate(_helium(), BoxSpec(), FilmPackage(), Environment())
    a = assess_film_thermal(sim, BoxSpec(), FilmLimits(transported_in_flight=True))
    assert a["film_min_c"] < -15
    levels = {r["risk"]: r["level"] for r in a["risks"]}
    assert any("Static" in k for k in levels)


def test_helpers():
    assert dew_point_c(20.0, 0.5) == pytest.approx(9.3, abs=0.3)
    assert h_conv(288.0, 1.225, 5.0, 0.3) > h_conv(220.0, 0.05, 5.0, 0.3)


def _helium():
    b = LatexBalloon(balloon_mass_g=1200, payload_mass_kg=1.5)
    t, z, ph = b.trajectory()
    return to_profile(t, z, ph, DAY, 44.26, -72.575)
