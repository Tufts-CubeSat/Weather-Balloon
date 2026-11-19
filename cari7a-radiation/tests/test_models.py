from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from balloon import atmosphere as atm
from balloon import sun
from balloon.latex import LatexBalloon
from balloon.solar import SolarBalloon
from cari.locfile import MAX_LINE, Query, build_loc, format_line
from cari.parsers import parse_ans, parse_deg_dat, parse_sum
from cari.runner import CariConfig, CariRunner, plan_queries
from dose import integrate, interp_rate, totals
from film import FilmSpec, assess, thresholds_mgy

CARI_DIR = CariConfig.load().cari_dir


# ---- atmosphere: USSA-1976 tabulated values -------------------------------------
@pytest.mark.parametrize("z, p, t", [(0, 101325, 288.15), (11000, 22632, 216.65),
                                     (20000, 5474.9, 216.65), (32000, 868.02, 228.65)])
def test_ussa76(z, p, t):
    # tables are given at geopotential altitude; convert to geometric
    zg = z * atm.R_EARTH_GP / (atm.R_EARTH_GP - z)
    assert atm.pressure(zg) == pytest.approx(p, rel=2e-3)
    assert atm.temperature(zg) == pytest.approx(t, abs=0.1)


def test_depth_sea_level():
    assert atm.depth_g_cm2(0) == pytest.approx(1033.2, rel=1e-3)


# ---- balloons ------------------------------------------------------------------------------
def test_latex_burst_typical():
    b = LatexBalloon(balloon_mass_g=1200, payload_mass_kg=1.5, target_ascent_mps=5.0)
    zb = b.burst_altitude_m()
    assert 29000 < zb < 35000  # HabHub calculator gives ~32-33 km for this setup
    t, z, ph = b.trajectory()
    assert z.max() == pytest.approx(zb)
    assert ph[0] == "ascent" and ph[-1] == "descent"
    assert 1.5 < t[-1] / 3600 < 3.5


def test_latex_more_lift_lower_burst():
    a = LatexBalloon(neck_lift_kg=2.5).burst_altitude_m()
    b = LatexBalloon(neck_lift_kg=4.0).burst_altitude_m()
    assert b < a


def test_solar_float_consistent():
    s = SolarBalloon(envelope_volume_m3=350, envelope_mass_kg=0.8, payload_mass_kg=1.0, delta_t_k=30)
    zf = s.equilibrium_altitude_m()
    assert s.lift_kg(zf) == pytest.approx(s.total_mass, rel=1e-3)
    v = SolarBalloon.required_volume_m3(zf, s.total_mass, 30)
    assert v == pytest.approx(350, rel=1e-3)


def test_solar_insufficient_lift():
    with pytest.raises(ValueError):
        SolarBalloon(envelope_volume_m3=1.0).equilibrium_altitude_m()


def test_sun_noon_elevation():
    # equinox, solar noon at Greenwich, equator -> near zenith
    el = sun.solar_elevation_deg(datetime(2026, 3, 20, 12, 7, tzinfo=timezone.utc), 0.0, 0.0)
    assert el > 88


def test_sunset_vermont_november():
    start = datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc)
    ss = sun.heated_until(start, 44.26, -72.575, 0.0)
    # local sunset ~16:15 EST = 21:15 UTC
    assert abs((ss - datetime(2026, 11, 21, 21, 15, tzinfo=timezone.utc)).total_seconds()) < 20 * 60
    assert sun.heated_until(start, 44.26, -72.575, 24000) > ss  # later sunset aloft


# ---- LOC writing / planning --------------------------------------------------------
def test_loc_lines_fit():
    for q in plan_queries(130000, "fine", [80123.4]):
        assert len(format_line(-44.123456, -172.5751234, q, 2026, 11)) <= MAX_LINE
    txt = build_loc(44.26, -72.575, [Query(80000, 5, 0)], 2026, 11)
    assert "START-" in txt and "STOP--" in txt and "D5,P0" in txt


def test_plan_covers_max_alt():
    qs = plan_queries(101234, "fast", [101234])
    alts = {q.alt_ft for q in qs if q.dose_code == 5}
    assert max(alts) >= 101234 and 101234 in alts


def test_set_ini_keeps_format():
    ini = "MENUS     = YES           !comment here\r\nVEHICLE   = YES           !x\r\nEND       = END\r\n"
    out = CariRunner._set_ini(ini, {"MENUS": "NO!", "VEHICLE": "NO!"})
    lines = out.splitlines()
    assert lines[0] == "MENUS     = NO!           !comment here"
    assert lines[1][12:24] == "NO!         "


# ---- parsers against the shipped CARI examples ---------------------------------------
@pytest.mark.skipif(not (CARI_DIR / "Examples" / "EXAMPLES.ANS").exists(), reason="CARI examples not available")
def test_parse_examples_ans():
    df = parse_ans(CARI_DIR / "Examples" / "EXAMPLES.ANS")
    assert len(df) > 50
    assert df.iloc[0]["rate"] == pytest.approx(4.7168, rel=1e-4)
    assert set(df["particle"]) >= {"TOTAL", "NEUTRONS"}


@pytest.mark.skipif(not (CARI_DIR / "Examples" / "EXAMPLE.DAT").exists(), reason="CARI examples not available")
def test_parse_examples_deg():
    dat = parse_deg_dat(CARI_DIR / "Examples" / "EXAMPLE.DAT")
    s = parse_sum(CARI_DIR / "Examples" / "EXAMPLE.SUM")
    assert len(dat) > 20
    assert dat["total_dose"].iloc[-1] == pytest.approx(s["total"], rel=0.02)


# ---- dose integration ------------------------------------------------------------------------
def test_integrate_constant_rate():
    prof = pd.DataFrame({"t_min": np.arange(0, 601, 1.0), "alt_ft": 80000.0})
    grid = pd.DataFrame({"alt_ft": [0, 80000, 100000], "rate": [0.02, 6.0, 8.0], "sigma": [0, 0.4, 0.5]})
    tot = totals(integrate(prof, {"absorbed_wb": grid}))
    assert tot["absorbed_wb"][0] == pytest.approx(60.0)  # 6 µGy/h * 10 h
    assert tot["absorbed_wb"][1] == pytest.approx(4.0)


def test_interp_loglinear():
    grid = pd.DataFrame({"alt_ft": [0, 10000], "rate": [1.0, 100.0]})
    assert interp_rate([5000], grid)[0] == pytest.approx(10.0)


# ---- film -----------------------------------------------------------------------------------
def test_film_thresholds_scale_with_speed():
    assert thresholds_mgy(500)["likely"] == pytest.approx(1.0)
    assert thresholds_mgy(125)["likely"] == pytest.approx(3.0, rel=1e-6)


def test_film_assess():
    fa = assess(100.0, 6.0, FilmSpec(), {7: 1000.0, 14: 1.0, 37: 1.0})
    assert fa["equiv_ground_storage_days"] == pytest.approx(100 / 0.1 / 24)
    assert fa["fraction_of_low"] == pytest.approx(0.1 / 0.3)
    assert fa["tracks"]["hze_per_cm2"] == 2.0
