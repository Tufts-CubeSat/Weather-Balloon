"""Generate the reference dataset in data/ by running CARI-7A.

    .venv\\Scripts\\python scripts\\build_dataset.py [--lat 44.26 --lon -72.575 --year 2026 --month 11]

Outputs (data/):
  altitude_sweep.csv        absorbed doses + all-species flux, 0-120 kft every 2 kft, at the site
  solar_sensitivity.csv     dose/flux at 60/80/100 kft for solar-min/max/latest dates
  solar_monthly_80kft.csv   monthly dose at 80 kft, 2019 -> latest heliocentric data
  mission_*.csv             time series for the baseline helium and solar missions
  mission_summary.csv       totals + film metrics for every mission
  metadata.json             run settings / provenance
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from balloon.atmosphere import M_PER_FT, depth_g_cm2  # noqa: E402
from balloon.latex import LatexBalloon  # noqa: E402
from balloon.profile import summarize, to_profile  # noqa: E402
from balloon.solar import SolarBalloon  # noqa: E402
from cari.locfile import PARTICLES, Query, format_line  # noqa: E402
from cari.runner import CariRunner, rate_table  # noqa: E402
from dose import integrate, totals  # noqa: E402
from film import FilmSpec, assess  # noqa: E402

DATA = ROOT / "data"
DOSE_COLS = {5: "absorbed_wb", 6: "si_0p5mm", 7: "si_0p3mm"}
SOLAR_DATES = {  # label -> (year, month)
    "solar min 2009/12 (deepest of space age)": (2009, 12),
    "solar min 2019/12": (2019, 12),
    "solar max 2014/04 (cycle 24)": (2014, 4),
    "solar max 2024/10 (cycle 25)": (2024, 10),
}


def wide(table: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    t = table.copy()
    t["col"] = t["key"]
    rate = t.pivot_table(index=index, columns="col", values="rate").add_suffix("")
    sig = t.pivot_table(index=index, columns="col", values="sigma").add_suffix("_sigma")
    out = rate.join(sig).reset_index()
    rename = {f"flux_P{c}": f"flux_{PARTICLES[c].replace(' ', '_').replace('.', '')}" for c in PARTICLES}
    rename.update({f"{k}_sigma": f"{v}_sigma" for k, v in rename.items()})
    return out.rename(columns=rename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=44.26)
    ap.add_argument("--lon", type=float, default=-72.575)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--month", type=int, default=11)
    ap.add_argument("--launch-utc", default="2026-11-21T13:00")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    r = CariRunner()
    lat, lon, yr, mo = args.lat, args.lon, args.year, args.month

    # 1) altitude sweep -------------------------------------------------------------
    alts = list(np.arange(0, 120001, 2000, dtype=float))
    qs = [Query(a, d, 0) for a in alts for d in DOSE_COLS] + [Query(a, 1, p) for a in alts for p in range(38)]
    print(f"altitude sweep: {len(qs)} queries ...", flush=True)
    sweep = r.run_locations(lat, lon, qs, yr, mo, tag="dataset-sweep")
    sw = wide(sweep.table, ["alt_ft"])
    sw.insert(1, "alt_km", sw["alt_ft"] * M_PER_FT / 1000)
    sw.insert(2, "depth_g_cm2", depth_g_cm2(sw["alt_ft"].to_numpy() * M_PER_FT))
    sw.to_csv(DATA / "altitude_sweep.csv", index=False, float_format="%.5g")

    # 2) solar-cycle sensitivity (dates are per-line in a LOC file, so one run) ------------
    hp_latest = r.heliocentric_potential(yr, mo)
    dates = dict(SOLAR_DATES)
    dates[f"launch month {yr}/{mo:02d} (uses {hp_latest['source']})"] = (yr, mo)
    sens_q, sens_meta = [], []
    for label, (y, m) in dates.items():
        for a in (60000.0, 80000.0, 100000.0):
            for d in DOSE_COLS:
                sens_q.append(Query(a, d, 0)); sens_meta.append((label, y, m))
            for p in (0, 1, 7, 13):
                sens_q.append(Query(a, 1, p)); sens_meta.append((label, y, m))
    monthly = [(y, m) for y in range(2019, 2027) for m in range(1, 13)
               if r.heliocentric_potential(y, m).get("fallback") is False]
    mon_q = [Query(80000.0, 5, 0) for _ in monthly]
    loc_lines = []
    for q, (_, y, m) in zip(sens_q, sens_meta):
        loc_lines.append(format_line(lat, lon, q, y, m))
    for q, (y, m) in zip(mon_q, monthly):
        loc_lines.append(format_line(lat, lon, q, y, m))
    print(f"solar sensitivity: {len(loc_lines)} queries ...", flush=True)
    res = _run_raw(r, loc_lines, "dataset-solar")
    n = len(sens_q)
    sens = res.iloc[:n].copy()
    sens["label"] = [s[0] for s in sens_meta]
    sens["key"] = [q.key for q in sens_q]
    sens["alt_ft"] = [q.alt_ft for q in sens_q]
    sens["heliocentric_mv"] = [r.heliocentric_potential(s[1], s[2])["mv"] for s in sens_meta]
    sw2 = wide(sens, ["label", "heliocentric_mv", "alt_ft"])
    sw2.to_csv(DATA / "solar_sensitivity.csv", index=False, float_format="%.5g")
    mon = res.iloc[n:].copy()
    pd.DataFrame({
        "month": [f"{y}-{m:02d}" for y, m in monthly],
        "heliocentric_mv": [r.heliocentric_potential(y, m)["mv"] for y, m in monthly],
        "absorbed_wb_ugy_h_80kft": mon["rate"].to_numpy(),
        "sigma": mon["sigma"].to_numpy(),
    }).to_csv(DATA / "solar_monthly_80kft.csv", index=False, float_format="%.5g")

    # 3) missions (integrate the launch-month sweep over each profile) ----------------------
    rates = rate_table(sweep.table)
    launch = datetime.fromisoformat(args.launch_utc).replace(tzinfo=timezone.utc)
    spec = FilmSpec()
    missions = {}
    hb = LatexBalloon(balloon_mass_g=1200, payload_mass_kg=1.5, target_ascent_mps=5.0)
    t, z, ph = hb.trajectory()
    missions["helium_1200g_1.5kg"] = (to_profile(t, z, ph, launch, lat, lon),
                                      f"1200 g latex, 1.5 kg, 5 m/s ascent; burst {hb.burst_altitude_m() / M_PER_FT:,.0f} ft")
    sb = SolarBalloon(lat=lat, lon=lon, launch_utc=launch, float_altitude_override_m=80000 * M_PER_FT)
    t, z, ph, info = sb.trajectory()
    missions["solar_80kft_until_sunset"] = (to_profile(t, z, ph, launch, lat, lon),
                                            f"80 kft float until sunset at float ({info['float_hours']:.1f} h) for "
                                            f"launch {launch:%Y-%m-%d %H:%M} UTC")
    for h in (10, 13, 16):
        sb = SolarBalloon(lat=lat, lon=lon, launch_utc=launch, float_altitude_override_m=80000 * M_PER_FT,
                          float_hours_override=h)
        t, z, ph, _ = sb.trajectory()
        missions[f"solar_80kft_{h}h"] = (to_profile(t, z, ph, launch, lat, lon), f"80 kft float for {h} h (override)")
    rows = []
    for name, (prof, desc) in missions.items():
        integ = integrate(prof, rates)
        cols = ["t_min", "utc", "alt_ft", "depth_g_cm2", "phase"] + [
            c for c in integ.columns if c.startswith(("absorbed_wb", "si_"))]
        integ[cols].to_csv(DATA / f"mission_{name}.csv", index=False, float_format="%.5g")
        tot = totals(integ)
        flu = {int(k[6:]): v[0] for k, v in tot.items() if k.startswith("flux_P")}
        fa = assess(*tot["absorbed_wb"], spec, flu)
        ps = summarize(prof)
        rows.append({
            "mission": name, "description": desc, "max_alt_ft": round(ps["max_alt_ft"]),
            "duration_h": ps["duration_h"], "hours_above_60kft": ps["hours_above_60kft"],
            "absorbed_wb_ugy": tot["absorbed_wb"][0], "absorbed_wb_sigma_ugy": tot["absorbed_wb"][1],
            "si_0p3mm_ugy": tot["si_0p3mm"][0], "si_0p5mm_ugy": tot["si_0p5mm"][0],
            "equiv_ground_storage_days": fa["equiv_ground_storage_days"],
            "fraction_of_measurable_fog": fa["fraction_of_low"],
            "hze_hits_per_frame": fa["tracks_per_frame"]["hze"],
            "proton_hits_per_frame": fa["tracks_per_frame"]["protons"],
            "neutron_fluence_per_cm2": flu.get(1),
        })
    pd.DataFrame(rows).to_csv(DATA / "mission_summary.csv", index=False, float_format="%.5g")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cari_version": "CARI-7A 4.2.0 (19 Oct 2021)",
        "settings": {"GCR_MODEL": 4, "NVCR": 0, "SUPERPOS": "NO!", "VEHICLE": "NO! (unshielded)",
                     "time resolution": "monthly average (DD=00, H0)"},
        "site": {"lat": lat, "lon": lon, "vertical_cutoff_gv": float(sweep.table["vcr_gv"].iloc[0])},
        "launch_month": f"{yr}/{mo:02d}", "heliocentric_potential_used": hp_latest,
        "film": vars(spec),
        "run_dirs (local, git-ignored)": [Path(sweep.run_dir).name, Path(res.attrs.get("run_dir", "")).name],
    }
    (DATA / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print("done ->", DATA)


def _run_raw(r: CariRunner, lines: list[str], tag: str) -> pd.DataFrame:
    """Run pre-formatted LOC lines (mixed dates) and return the parsed ANS rows."""
    import shutil
    import uuid
    from cari.parsers import parse_ans
    from cari.runner import Shielding
    wd = r.ensure_workdir()
    name = "R" + uuid.uuid4().hex[:7].upper()
    text = "\r\n".join(["dataset", "START-------------------------------", *lines,
                        "STOP--------------------------------"]) + "\r\n"
    with r._lock():
        (wd / f"{name}.LOC").write_text(text, newline="")
        r._write_controls(wd, f"{name}.LOC", Shielding())
        r._exec(wd, len(lines))
        run_dir = r._new_run_dir(tag)
        shutil.copy2(wd / f"{name}.LOC", run_dir / "input.LOC")
        shutil.copy2(wd / f"{name}.ANS", run_dir / "output.ANS")
        (wd / f"{name}.LOC").unlink(); (wd / f"{name}.ANS").unlink()
    df = parse_ans(run_dir / "output.ANS")
    if len(df) != len(lines):
        raise RuntimeError(f"expected {len(lines)} rows, got {len(df)} ({run_dir})")
    df.attrs["run_dir"] = run_dir
    return df


if __name__ == "__main__":
    main()
