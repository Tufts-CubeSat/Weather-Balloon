"""Balloon radiation planner: balloon physics -> flight profile -> CARI-7A -> film dose.

Run:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from balloon.atmosphere import M_PER_FT
from balloon.latex import BURST_DIAMETER_M, LatexBalloon
from balloon.profile import summarize, to_profile
from balloon.solar import SolarBalloon
from cari.locfile import PARTICLES, particle_z
from cari.runner import DETAIL_LEVELS, CariRunner, Shielding, plan_queries, rate_table
from dose import integrate, totals
from film import FLAT_SHEET_FACTOR, FORMATS, STOCKS, FilmSpec, assess
from thermal import (BATTERIES, CONTAINER_MATERIALS, DEFAULT_DEVICES, FILM_PACKAGES, THERMAL_NOTES, BoxSpec,
                     Device, Environment, FilmLimits, FilmPackage, assess_film_thermal, simulate)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PHASE_COLORS = {"ascent": "#2a78d6", "burst": "#d64545", "float": "#e0a100", "descent": "#3a9b5c", "x": "#888"}

st.set_page_config(page_title="Balloon Radiation Planner", layout="wide")
st.title("Balloon Radiation Planner")
st.caption("Flight profile from balloon physics → galactic cosmic radiation from FAA CARI-7A → dose to the film payload.")


@st.cache_resource
def get_runner() -> CariRunner:
    return CariRunner()


@st.cache_data(max_entries=30, show_spinner=False)
def run_thermal(_profile, _box, _film, _env, cache_key: str) -> pd.DataFrame:
    return simulate(_profile, _box, _film, _env)


def thermal_for(prof: pd.DataFrame) -> pd.DataFrame:
    ck = repr((len(prof), float(prof["alt_m"].sum()), str(prof["utc"].iloc[0]), prof["lat"].iloc[0],
               prof["lon"].iloc[0], box, film_pkg, env))
    return run_thermal(prof, box, film_pkg, env, ck)


runner = get_runner()
problems = runner.check()
if problems:
    st.error("\n\n".join(problems))

# ----------------------------------------------------------------------------- sidebar
FOAMS = {  # name: (k W/mK, density kg/m3, solar absorptivity, IR emissivity)
    "White EPS (styrofoam)": (0.036, 20.0, 0.30, 0.90),
    "Pink/blue XPS (foam board)": (0.029, 30.0, 0.50, 0.90),
}
FINISHES = {  # name: (absorptivity, emissivity) or None = use the foam's own values
    "Bare foam": None,
    "White paint / tape": (0.25, 0.90),
    "Aluminum foil / tape": (0.15, 0.05),
    "Black": (0.95, 0.90),
}

with st.sidebar:
    with st.expander("Launch & weather", expanded=True):
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (°N)", -90.0, 90.0, 44.26, 0.01, format="%.4f")
        lon = c2.number_input("Longitude (°E, W negative)", -180.0, 180.0, -72.575, 0.01, format="%.4f")
        launch_alt_ft = st.number_input("Launch site elevation (ft)", 0, 15000, 500, 50)
        c1, c2 = st.columns(2)
        launch_date = c1.date_input("Launch date", date(2026, 11, 21))
        launch_time = c2.time_input("Launch time (UTC)", time(13, 0))
        launch_utc = datetime.combine(launch_date, launch_time, tzinfo=timezone.utc)
        c1, c2 = st.columns(2)
        surf_c = c1.number_input("Ground air temp (°C)", -40.0, 45.0, 2.0, 1.0,
                                 help="Shifts the standard atmosphere near the ground; no effect above the tropopause.")
        rh = c2.number_input("Ground humidity (%)", 5, 100, 70, 5) / 100
        albedo = st.slider("Ground albedo", 0.05, 0.9, 0.3, 0.05,
                           help="Reflected sunlight. About 0.15 for forest, 0.3 average, 0.8 for fresh snow.")

    with st.expander("Payload: box & electronics", expanded=False):
        c1, c2, c3 = st.columns(3)
        bl = c1.number_input("L (cm)", 5.0, 100.0, 30.0, 1.0)
        bw = c2.number_input("W (cm)", 5.0, 100.0, 25.0, 1.0)
        bh = c3.number_input("H (cm)", 5.0, 100.0, 20.0, 1.0)
        wall_in = st.number_input("Wall thickness (in)", 0.125, 4.0, 0.5, 0.125)
        foam = st.selectbox("Foam", list(FOAMS))
        finish = st.selectbox("Exterior finish", list(FINISHES))
        fk, fdens, fa_, fe_ = FOAMS[foam]
        ba, be = FINISHES[finish] or (fa_, fe_)
        st.caption("Electronics: enter the **DC power drawn from the battery** (e.g. measured with a USB power "
                   "meter), not RF output. A 1 W transmitter draws ~2.5–3 W. Average = power × duty. "
                   "Add a row for a resistive heater if you fly one.")
        dev_df = st.data_editor(
            pd.DataFrame([vars(d) for d in DEFAULT_DEVICES]).rename(
                columns={"name": "device", "power_w": "power (W)", "duty": "duty (0-1)"}),
            num_rows="dynamic", use_container_width=True, hide_index=True, key="devices")
        devices = [Device(str(r["device"]), float(r["power (W)"] or 0), float(r["duty (0-1)"] or 0))
                   for _, r in dev_df.dropna(subset=["device"]).iterrows()]
        c1, c2 = st.columns(2)
        elec_mass = c1.number_input("Electronics mass (kg)", 0.0, 5.0, 0.30, 0.05)
        batt_mass = c2.number_input("Battery mass (kg)", 0.0, 5.0, 0.20, 0.05)
        batt_type = st.selectbox("Battery type", list(BATTERIES))
        c1, c2 = st.columns(2)
        batt_wh = c1.number_input("Battery energy (Wh, 0 = estimate)", 0.0, 1000.0, 0.0, 5.0,
                                  help="Usable energy of the whole pack. 0 estimates it from mass × type "
                                       "(L91 ≈ 230 Wh/kg, Li-ion ≈ 170, alkaline ≈ 90). When it runs out the "
                                       "electronics stop heating the box.")
        comp_cm2 = c2.number_input("Electronics surface (cm², 0 = estimate)", 0.0, 20000.0, 0.0, 100.0,
                                   help="Exposed surface of boards, camera and battery packs that sheds heat into "
                                        "the box air. 0 estimates ~2000 cm² per kg. Smaller = electronics run "
                                        "warmer than the box air.")
        box = BoxSpec(length_cm=bl, width_cm=bw, height_cm=bh, wall_cm=wall_in * 2.54, foam_k=fk,
                      foam_density=fdens, absorptivity=ba, emissivity=be, devices=devices,
                      electronics_mass_kg=elec_mass, battery_mass_kg=batt_mass, battery_type=batt_type,
                      battery_wh=batt_wh, component_area_m2=comp_cm2 / 1e4)
        runtime = (f" (≈ {box.battery_energy_wh / box.avg_power_w:.1f} h at this draw)" if box.avg_power_w > 0 else "")
        st.caption(f"Average power **{box.avg_power_w:.2f} W** · battery **{box.battery_energy_wh:.0f} Wh**{runtime} · "
                   f"box mass **{box.mass_kg:.2f} kg** (foam {box.foam_mass_kg * 1000:.0f} g)")

    with st.expander("Payload: film package", expanded=False):
        preset = st.selectbox("Package", list(FILM_PACKAGES))
        p = FILM_PACKAGES[preset]
        location = st.radio("Location", ["Outside the box", "Inside the box"], horizontal=True)
        k_ = f"_{preset}"
        c1, c2 = st.columns(2)
        f_d = c1.number_input("Can Ø (cm)", 1.0, 50.0, p["diameter_cm"], 0.5, key="fd" + k_)
        f_h = c2.number_input("Can height (cm)", 1.0, 30.0, p["height_cm"], 0.5, key="fh" + k_)
        c1, c2 = st.columns(2)
        f_m = c1.number_input("Film mass (kg)", 0.005, 5.0, p["film_mass_kg"], 0.01, key="fm" + k_)
        f_w = c2.number_input("Can wall (mm)", 0.1, 10.0, p["container_wall_mm"], 0.1, key="fw" + k_)
        f_mat = st.selectbox("Can material", list(CONTAINER_MATERIALS),
                             index=list(CONTAINER_MATERIALS).index(p["container_material"]), key="fmat" + k_)
        c1, c2 = st.columns(2)
        f_a = c1.number_input("Solar absorptivity", 0.02, 1.0, p["absorptivity"], 0.05, key="fa" + k_)
        f_e = c2.number_input("IR emissivity", 0.02, 1.0, p["emissivity"], 0.05, key="fe" + k_)
        f_ins = st.number_input("Insulation wrap (cm of foam)", 0.0, 10.0, 0.0, 0.5,
                                help="Foam or bubble wrap around the can. Slows cooling, adds mass and a little shielding.")
        film_pkg = FilmPackage(location="outside" if location.startswith("Outside") else "inside",
                               diameter_cm=f_d, height_cm=f_h, film_mass_kg=f_m, container_material=f_mat,
                               container_wall_mm=f_w, absorptivity=f_a, emissivity=f_e, insulation_cm=f_ins)
        st.caption(f"Package mass **{film_pkg.mass_kg:.2f} kg** (can {film_pkg.container_mass_kg * 1000:.0f} g)")

    with st.expander("Balloon", expanded=True):
        rig_mass = st.number_input("Parachute + line + rigging (kg)", 0.0, 3.0, 0.25, 0.05)
        computed_payload = box.mass_kg + film_pkg.mass_kg + rig_mass
        use_computed = st.checkbox(f"Use computed payload mass ({computed_payload:.2f} kg)", value=True,
                                   help="Box + electronics + batteries + film package + rigging.")
        kind = st.radio("Type", ["Helium latex", "Solar (black envelope)"], horizontal=True)
        if kind == "Helium latex":
            sizes = sorted(BURST_DIAMETER_M)
            mass = st.selectbox("Balloon mass (g)", sizes, index=sizes.index(1200),
                                format_func=lambda m: f"{m} g (burst Ø {BURST_DIAMETER_M[m]} m)")
            payload = computed_payload if use_computed else st.number_input(
                "Payload + parachute + line (kg)", 0.1, 6.0, 1.5, 0.1)
            gas = st.selectbox("Lift gas", ["helium", "hydrogen"])
            fill_mode = st.radio("Fill by", ["Target ascent rate", "Neck lift"], horizontal=True)
            if fill_mode == "Target ascent rate":
                ascent = st.slider("Target ascent rate (m/s)", 2.0, 8.0, 5.0, 0.1)
                neck = None
            else:
                neck = st.number_input("Neck lift (kg)", 0.2, 10.0, 2.6, 0.05)
                ascent = 5.0
            c1, c2 = st.columns(2)
            chute_d = c1.number_input("Parachute Ø (m)", 0.3, 4.0, 1.2, 0.1)
            chute_cd = c2.number_input("Parachute Cd", 0.3, 2.0, 1.3, 0.05)
            burst_override = st.number_input("Override burst altitude (ft, 0 = compute)", 0, 140000, 0, 1000)
            balloon = LatexBalloon(balloon_mass_g=mass, payload_mass_kg=payload, neck_lift_kg=neck,
                                   target_ascent_mps=ascent, gas=gas, parachute_diameter_m=chute_d,
                                   parachute_cd=chute_cd, launch_alt_m=launch_alt_ft * M_PER_FT,
                                   burst_altitude_override_m=burst_override * M_PER_FT or None)
        else:
            vol = st.number_input("Envelope volume (m³)", 5.0, 5000.0, 350.0, 5.0)
            env_mass = st.number_input("Envelope mass (kg)", 0.05, 20.0, 0.8, 0.05)
            payload = computed_payload if use_computed else st.number_input(
                "Payload mass (kg)", 0.05, 10.0, 1.0, 0.05)
            dT = st.slider("Solar heating ΔT inside envelope (K)", 5.0, 60.0, 30.0, 1.0,
                           help="Biggest uncertainty. Stratospheric black balloons typically reach roughly 10-40 K.")
            min_el = st.slider("Heating ends when sun is below (° above visible horizon)", -2.0, 15.0, 0.0, 0.5)
            descent = st.number_input("Descent rate after sunset (m/s)", 0.5, 15.0, 4.0, 0.5)
            o_alt = st.number_input("Override float altitude (ft, 0 = compute)", 0, 130000, 0, 1000)
            o_hours = st.number_input("Override float duration (h, 0 = until sunset)", 0.0, 48.0, 0.0, 0.5)
            balloon = SolarBalloon(envelope_volume_m3=vol, envelope_mass_kg=env_mass, payload_mass_kg=payload,
                                   delta_t_k=dT, launch_alt_m=launch_alt_ft * M_PER_FT, lat=lat, lon=lon,
                                   launch_utc=launch_utc, min_sun_elevation_deg=min_el, descent_rate_mps=descent,
                                   float_altitude_override_m=o_alt * M_PER_FT or None,
                                   float_hours_override=o_hours or None)
            st.caption("Sizing helper")
            tgt = st.number_input("Target float altitude (ft)", 20000, 120000, 80000, 1000)
            need = SolarBalloon.required_volume_m3(tgt * M_PER_FT, env_mass + payload, dT)
            st.write(f"Needs **{need:,.0f} m³** (sphere Ø {(6 * need / np.pi) ** (1 / 3):.1f} m) for "
                     f"{env_mass + payload:.2f} kg at ΔT = {dT:.0f} K. Envelope mass grows with size.")

    with st.expander("Radiation shielding (film)", expanded=False):
        derive = st.checkbox("Derive from the film package", value=True,
                             help="Uses the can wall and insulation, plus the box wall if the film is inside the box.")
        if derive:
            sh = film_pkg.shielding_g_cm2()
            if film_pkg.location == "inside":
                sh["hdpe"] += box.foam_density / 1000 * box.wall_cm
            al, hdpe = round(sh["al"], 4), round(sh["hdpe"], 4)
            st.caption(f"Aluminum-equivalent {al:.3f} g/cm² · plastic/foam {hdpe:.3f} g/cm²")
        else:
            al = st.number_input("Aluminum (g/cm²)", 0.0, 20.0, 0.0, 0.05, help="0.1 g/cm² ≈ 0.37 mm Al.")
            hdpe = st.number_input("Polyethylene / foam (g/cm²)", 0.0, 20.0, 0.0, 0.05)
        shielding = Shielding(al, hdpe)

    with st.expander("Film stock & limits", expanded=False):
        stock = st.selectbox("Stock", list(STOCKS), index=0)
        fmt = st.selectbox("Frame format", list(FORMATS))
        length_ft = st.number_input("Roll length (ft)", 1, 2000, 100, 25)
        bg = st.number_input("Ground background (µGy/h)", 0.01, 1.0, 0.10, 0.01,
                             help="Used only for the 'equivalent shelf time' comparison.")
        film_spec = FilmSpec(stock=stock, length_m=length_ft * 0.3048, frame_area_cm2=FORMATS[fmt],
                             background_ugy_h=bg)
        c1, c2 = st.columns(2)
        lim_cold = c1.number_input("Cold-handling limit (°C)", -80.0, 10.0, -18.0, 1.0)
        lim_severe = c2.number_input("Severe cold (°C)", -90.0, 0.0, -40.0, 1.0)
        c1, c2 = st.columns(2)
        lim_store = c1.number_input("Storage limit (°C)", 0.0, 40.0, 13.0, 1.0)
        lim_heat = c2.number_input("Heat limit (°C)", 15.0, 80.0, 32.0, 1.0)
        transported = st.checkbox("Film is advanced/wound during the flight (loaded in a camera)", value=False)
        limits = FilmLimits(cold_handling_c=lim_cold, severe_cold_c=lim_severe, warm_storage_c=lim_store,
                            heat_c=lim_heat, transported_in_flight=transported)

    with st.expander("Ground operations", expanded=False):
        start_c = st.number_input("Payload temp when taken outside (°C)", -30.0, 50.0, 20.0, 1.0)
        c1, c2 = st.columns(2)
        pre_min = c1.number_input("Pre-launch hold (min)", 0, 600, 45, 5)
        post_min = c2.number_input("Wait before recovery (min)", 0, 2880, 120, 15)
        wind = st.number_input("Ground wind (m/s)", 0.0, 20.0, 3.0, 0.5)
        env = Environment(surface_temp_c=surf_c, ground_rh=rh, ground_albedo=albedo, start_temp_c=start_c,
                          prelaunch_min=pre_min, postlanding_min=post_min, ground_wind_mps=wind)

    st.header("CARI-7A")
    detail = st.radio("Particle detail", list(DETAIL_LEVELS), index=1, horizontal=True,
                      help="fast: dose + total ions/neutrons/protons/alphas every 5 kft · "
                           "standard: all 37 species every 5 kft · fine: all species every 2 kft")
    run_clicked = st.button("▶ Run CARI-7A", type="primary", use_container_width=True, disabled=bool(problems))

# ----------------------------------------------------------------------------- profile
key = "helium" if kind == "Helium latex" else "solar"
try:
    if key == "helium":
        t_s, z_m, ph = balloon.trajectory()
        info = {"burst_alt_ft": balloon.burst_altitude_m() / M_PER_FT,
                "free_lift_kg": balloon.free_lift_kg(), "launch_volume_m3": balloon.launch_volume_m3()}
    else:
        t_s, z_m, ph, info = balloon.trajectory()
    profile = to_profile(t_s, z_m, ph, launch_utc, lat, lon)
except ValueError as e:
    st.error(str(e))
    st.stop()

psum = summarize(profile)
max_alt = psum["max_alt_ft"]
extra_alts = [max_alt]
n_q = len(plan_queries(max_alt, detail, extra_alts))

if run_clicked:
    with st.spinner(f"Running CARI-7A on {n_q} location queries (~{n_q * 0.14 / 60:.1f} min)…"):
        try:
            res = runner.run_locations(lat, lon, plan_queries(max_alt, detail, extra_alts),
                                       launch_date.year, launch_date.month, shielding=shielding, tag=key)
        except Exception as e:  # surface CARI failures in the UI
            st.error(f"CARI-7A run failed: {e}")
            st.stop()
    st.session_state[f"run_{key}"] = {
        "res": res, "profile": profile, "info": info, "film_spec": film_spec,
        "inputs": {"lat": lat, "lon": lon, "launch_utc": launch_utc, "detail": detail,
                   "shielding": vars(shielding), "balloon": repr(balloon)},
    }

run = st.session_state.get(f"run_{key}")
stale = run is not None and (len(run["profile"]) != len(profile)
                             or not np.allclose(run["profile"]["alt_ft"].to_numpy(), profile["alt_ft"].to_numpy()))

tabs = st.tabs(["Profile", "Dose", "Particles", "Film", "Thermal", "Compare", "Dataset"])

# ---- Profile
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Max altitude", f"{max_alt / 1000:.1f} kft", f"{max_alt * M_PER_FT / 1000:.1f} km", delta_color="off")
    c[1].metric("Flight duration", f"{psum['duration_h']:.2f} h")
    c[2].metric("Time above 60 kft", f"{psum['hours_above_60kft']:.2f} h")
    if key == "helium":
        c[3].metric("Free lift", f"{info['free_lift_kg']:.2f} kg", f"{info['launch_volume_m3']:.1f} m³ at launch",
                    delta_color="off")
    else:
        c[3].metric("Float duration", f"{info['float_hours']:.1f} h", info["float_end_source"], delta_color="off")
        if info["sun_elevation_at_launch"] < 5:
            st.warning(f"Sun elevation at launch is {info['sun_elevation_at_launch']:.1f}°. "
                       "A solar balloon needs sunlight to lift off.")
        st.info(f"Reaches float at {info['reach_float_utc']:%H:%M} UTC · heating ends / descent starts "
                f"after {info['float_hours']:.1f} h · lands ≈ {info['landing_utc']:%H:%M} UTC.")
    fig = go.Figure()
    for phase, g in profile.groupby("phase", sort=False):
        fig.add_scatter(x=g["t_min"] / 60, y=g["alt_ft"], mode="markers", name=phase,
                        marker=dict(size=3, color=PHASE_COLORS.get(phase, "#888")))
    fig.update_layout(xaxis_title="Hours since launch", yaxis_title="Altitude (ft)", height=420,
                      margin=dict(t=20, b=40), legend_title="Phase")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{n_q} CARI queries needed for this profile at '{detail}' detail "
               f"(~{n_q * 0.14 / 60:.1f} min). Horizontal drift is ignored: the geomagnetic cutoff "
               "changes little over typical drift distances.")

if run is None:
    for t in tabs[1:4]:
        with t:
            st.info("Set up the flight in the sidebar and press **▶ Run CARI-7A**.")
else:
    if stale:
        st.warning("The inputs changed since the last CARI run. Results below are for the previous profile. "
                   "Press Run again to update.")
    res = run["res"]
    rates = rate_table(res.table)
    integ = integrate(run["profile"], rates)
    tot = totals(integ)
    fluences = {int(k[6:]): v[0] for k, v in tot.items() if k.startswith("flux_P")}
    fa = assess(*tot["absorbed_wb"], run["film_spec"], fluences)

    # ---- Dose
    with tabs[1]:
        d, s = tot["absorbed_wb"]
        c = st.columns(4)
        c[0].metric("Total absorbed dose (tissue)", f"{d:.2f} µGy", f"± {s:.2f} (CARI 1σ)", delta_color="off")
        c[1].metric("Dose in 0.3 mm Si", f"{tot['si_0p3mm'][0]:.2f} µGy")
        c[2].metric("Peak rate", f"{integ['absorbed_wb_rate'].max():.2f} µGy/h")
        c[3].metric("Equivalent shelf time", f"{fa['equiv_ground_storage_days']:.1f} days",
                    f"at {run['film_spec'].background_ugy_h} µGy/h", delta_color="off")
        hp = res.meta["heliocentric_potential"]
        st.caption(f"CARI-7A ISO GCR model, heliocentric potential {hp.get('mv')} MV from {hp.get('source')} · "
                   f"vertical cutoff {res.table['vcr_gv'].iloc[0]:.2f} GV · {res.elapsed_s:.0f} s · "
                   f"run files: `{res.run_dir}`")
        x = integ["t_min"] / 60
        fig = go.Figure()
        fig.add_scatter(x=x, y=integ["absorbed_wb_rate"], name="Dose rate (µGy/h)", line=dict(color="#2a78d6"))
        up = integ["absorbed_wb_cum"] + integ["absorbed_wb_cum_sigma"]
        lo = integ["absorbed_wb_cum"] - integ["absorbed_wb_cum_sigma"]
        fig.add_scatter(x=pd.concat([x, x[::-1]]), y=pd.concat([up, lo[::-1]]), fill="toself",
                        fillcolor="rgba(214,69,69,0.15)", line=dict(width=0), name="±1σ", yaxis="y2",
                        hoverinfo="skip")
        fig.add_scatter(x=x, y=integ["absorbed_wb_cum"], name="Cumulative dose (µGy)", yaxis="y2",
                        line=dict(color="#d64545"))
        fig.update_layout(xaxis_title="Hours since launch", yaxis=dict(title="Dose rate (µGy/h)"),
                          yaxis2=dict(title="Cumulative (µGy)", overlaying="y", side="right"), height=440,
                          margin=dict(t=20, b=40), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        g = rates["absorbed_wb"]
        fig2 = go.Figure()
        for k, name in (("absorbed_wb", "Tissue (whole-body avg)"), ("si_0p5mm", "0.5 mm Si"), ("si_0p3mm", "0.3 mm Si")):
            fig2.add_scatter(x=rates[k]["rate"], y=rates[k]["alt_ft"], name=name, mode="lines+markers",
                             marker=dict(size=4))
        fig2.update_layout(xaxis_title="Absorbed dose rate (µGy/h)", yaxis_title="Altitude (ft)", height=380,
                           margin=dict(t=20, b=40))
        st.plotly_chart(fig2, use_container_width=True)
        cols = ["t_min", "utc", "alt_ft", "depth_g_cm2", "phase"] + [c for c in integ.columns if c.startswith(("absorbed", "si_"))]
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download profile + dose CSV", integ.to_csv(index=False), f"{key}_dose.csv", "text/csv")
        c2.download_button("Download CARI input (.LOC)", (res.run_dir / "input.LOC").read_text(), f"{key}.LOC")
        c3.download_button("Download CARI output (.ANS)", (res.run_dir / "output.ANS").read_text(), f"{key}.ANS")

    # ---- Particles
    with tabs[2]:
        flux_keys = sorted([k for k in rates if k.startswith("flux_P")], key=lambda k: int(k[6:]))
        default = [k for k in flux_keys if int(k[6:]) in (1, 2, 3, 7, 13, 5)]
        chosen = st.multiselect("Species", flux_keys, default=default or flux_keys[:5],
                                format_func=lambda k: PARTICLES[int(k[6:])])
        fig = go.Figure()
        for k in chosen:
            fig.add_scatter(x=rates[k]["rate"], y=rates[k]["alt_ft"], name=PARTICLES[int(k[6:])], mode="lines")
        fig.update_layout(xaxis_type="log", xaxis_title="Flux (particles / cm² / s)", yaxis_title="Altitude (ft)",
                          height=460, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)
        rows = [{"code": c, "species": PARTICLES[c], "Z": particle_z(c), "fluence (/cm²)": v,
                 "per frame": v * run["film_spec"].frame_area_cm2 * FLAT_SHEET_FACTOR,
                 "per roll": v * run["film_spec"].area_cm2 * FLAT_SHEET_FACTOR}
                for c, v in sorted(fluences.items())]
        st.subheader("Flight fluence")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3g")
                                    for c in ("fluence (/cm²)", "per frame", "per roll")})
        st.caption("Code 0 'Total ions' excludes neutrons and photons. Fluence is the omnidirectional flux integrated "
                   "over the flight. Hits per frame/roll use fluence × area / 2 (isotropic field crossing a flat sheet).")

    # ---- Film
    with tabs[3]:
        thr = fa["thresholds_mgy"]
        st.subheader(f"{run['film_spec'].stock}: {fa['verdict']}")
        fig = go.Figure()
        fig.add_bar(y=["Flight dose"], x=[fa["dose_ugy"] / 1000], orientation="h", marker_color="#2a78d6",
                    error_x=dict(type="data", array=[fa["dose_sigma_ugy"] / 1000]), name="flight")
        for i, (name, v, col) in enumerate((("measurable", thr["low"], "#e0a100"),
                                            ("likely visible fog", thr["likely"], "#d67a00"),
                                            ("obvious fog", thr["high"], "#d64545"))):
            fig.add_vline(x=v, line_color=col, line_dash="dash")
            fig.add_annotation(x=np.log10(v), y=1.0 + 0.16 * i, yref="paper", text=f"{name} ({v:.2g} mGy)",
                               showarrow=False, font=dict(color=col), xanchor="center")
        fig.update_layout(xaxis_type="log", xaxis_title="Absorbed dose (mGy, log)", height=280,
                          margin=dict(t=90, b=40), showlegend=False,
                          xaxis_range=[np.log10(max(fa["dose_ugy"] / 1000 / 3, 1e-4)), np.log10(thr["high"] * 3)])
        st.plotly_chart(fig, use_container_width=True)
        c = st.columns(4)
        c[0].metric("Fraction of 'measurable' level", f"{fa['fraction_of_low']:.0%}")
        c[1].metric("Equivalent shelf time", f"{fa['equiv_ground_storage_days']:.1f} days")
        c[2].metric("≈ carry-on X-ray passes", f"{fa['equiv_carryon_xray_passes']:.0f}")
        c[3].metric("Fraction of one CT bag scan", f"{fa['fraction_of_ct_bag_scan']:.1%}")
        if "tracks_per_frame" in fa:
            tpf = fa["tracks_per_frame"]
            fmt_n = lambda v: "n/a (needs 'standard' detail)" if np.isnan(v) else f"{v:.3g}"
            st.markdown(
                f"**Particle hits per frame** (fluence × frame area / 2): protons ≈ {fmt_n(tpf['protons'])}, "
                f"alphas ≈ {fmt_n(tpf['alphas'])}, heavy ions Z≥3 ≈ {fmt_n(tpf['hze'])}, "
                f"neutrons ≈ {fmt_n(tpf['neutrons'])}. "
                "No published evidence shows single cosmic-ray tracks are visible in color negative film. "
                "Treat these numbers as a curiosity, not a visibility prediction.")
        st.info("**Recommendation:** fly the test roll and keep a control roll from the same emulsion batch on the "
                "ground, in the same shipping path. Develop both together. At these doses the control is what "
                "makes any difference detectable.")
        with st.expander("How the thresholds were chosen (sources and caveats)"):
            import film as film_mod
            st.code(film_mod.__doc__, language="text")

# ---- Thermal (independent of CARI)
LEVEL_ICON = {"ok": "OK", "low": "Low", "caution": "Caution", "high": "High"}
with tabs[4]:
    sim = thermal_for(profile)
    ta = assess_film_thermal(sim, box, limits)
    st.caption(f"Film package **{'outside' if film_pkg.location == 'outside' else 'inside'} the box** · "
               f"electronics {box.avg_power_w:.2f} W average · ground {env.surface_temp_c:.0f} °C, "
               f"RH {env.ground_rh:.0%} (dew point {ta['dew_point_c']:.0f} °C) · "
               f"{env.prelaunch_min:.0f} min pre-launch hold, {env.postlanding_min:.0f} min before recovery.")
    c = st.columns(4)
    c[0].metric("Film minimum", f"{ta['film_min_c']:.0f} °C", f"air min {ta['air_min_c']:.0f} °C", delta_color="off")
    c[1].metric(f"Film below {limits.cold_handling_c:.0f} °C", f"{ta['h_below_cold']:.1f} h",
                f"{ta['h_below_severe']:.1f} h below {limits.severe_cold_c:.0f} °C", delta_color="off")
    c[2].metric("Film maximum", f"{ta['film_max_c']:.0f} °C",
                f"{ta['h_above_storage']:.1f} h above {limits.warm_storage_c:.0f} °C", delta_color="off")
    c[3].metric("Electronics/batteries", f"{ta['box_min_c']:.0f} to {ta['box_max_c']:.0f} °C",
                f"box air min {ta['box_air_min_c']:.0f} °C · rated to {ta['battery_min_rating_c']:.0f} °C",
                delta_color="off")

    risk_df = pd.DataFrame(ta["risks"])
    risk_df["level"] = risk_df["level"].map(LEVEL_ICON)
    st.dataframe(risk_df, hide_index=True, use_container_width=True,
                 column_config={"detail": st.column_config.TextColumn(width="large")})

    x = sim["t_min"] / 60
    fig = go.Figure()
    fig.add_scatter(x=x, y=sim["T_air"], name="Outside air", line=dict(color="#9aa4b2", dash="dot"))
    fig.add_scatter(x=x, y=sim["T_box"], name="Electronics/batteries", line=dict(color="#2a78d6"))
    fig.add_scatter(x=x, y=sim["T_box_air"], name="Box interior air", line=dict(color="#2a78d6", dash="dash", width=1))
    fig.add_scatter(x=x, y=sim["T_film"], name="Film", line=dict(color="#d64545", width=3))
    fig.add_scatter(x=x, y=sim["alt_ft"] / 1000, name="Altitude (kft)", yaxis="y2",
                    line=dict(color="#3a9b5c", width=1), opacity=0.5)
    for v, name in ((limits.cold_handling_c, "cold-handling limit"), (limits.severe_cold_c, "severe cold"),
                    (ta["dew_point_c"], "ground dew point")):
        fig.add_hline(y=v, line_dash="dash", line_color="#888", annotation_text=name, annotation_position="bottom left")
    fig.add_vrect(x0=x.min(), x1=0, fillcolor="#e0a100", opacity=0.08, line_width=0, annotation_text="pre-launch",
                  annotation_position="top left")
    t_land = profile["t_min"].iloc[-1] / 60
    fig.add_vrect(x0=t_land, x1=x.max(), fillcolor="#3a9b5c", opacity=0.08, line_width=0,
                  annotation_text="awaiting recovery", annotation_position="top left")
    fig.update_layout(xaxis_title="Hours since launch", yaxis_title="Temperature (°C)",
                      yaxis2=dict(title="Altitude (kft)", overlaying="y", side="right", showgrid=False),
                      height=480, margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=x, y=sim["sun_wm2"], name="Direct sun (W/m²)", line=dict(color="#e0a100"))
    fig2.update_layout(xaxis_title="Hours since launch", yaxis_title="Direct sunlight (W/m²)", height=220,
                       margin=dict(t=10, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Model details and energy budget"):
        st.markdown(
            f"- Box: {box.length_cm:.0f}×{box.width_cm:.0f}×{box.height_cm:.0f} cm, {box.wall_cm / 2.54:.2f} in walls, "
            f"k = {box.foam_k} W/m·K → foam wall **{box.wall_resistance:.2f} K/W**. Electronics → box air "
            f"(surface {box.component_area * 1e4:.0f} cm²): **{1 / box.internal_conductance(288, 1):.2f} K/W** at sea level, "
            f"**{1 / box.internal_conductance(240, 0.05):.2f} K/W** at ~20 km.\n"
            f"- Steady state with {box.avg_power_w:.2f} W: box air ≈ {box.avg_power_w * box.wall_resistance:.1f} °C and "
            f"electronics ≈ **{box.avg_power_w * (box.wall_resistance + 1 / box.internal_conductance(240, 0.05)):.1f} °C** "
            "above the outer surface at altitude. The outer surface itself sits near radiative balance "
            "(sun + earth infrared vs. emission), well above the −56 °C air in daylight.\n"
            f"- Heat capacity {box.heat_capacity:.0f} J/K. Battery {box.battery_energy_wh:.0f} Wh"
            + (f", empty after {ta['battery_empty_h']:.1f} h.\n" if ta["battery_empty_h"] is not None else ", lasts the whole flight.\n") +
            f"- Film package: {film_pkg.mass_kg * 1000:.0f} g, heat capacity {film_pkg.heat_capacity:.0f} J/K, "
            f"surface area {film_pkg.area * 1e4:.0f} cm², absorptivity/emissivity {film_pkg.absorptivity}/{film_pkg.emissivity}.\n"
            "- Low-emissivity (bare metal, foil) surfaces lose very little heat by radiation, so they stay warmer at night "
            "and in the stratosphere. A black can warms fastest in sun and cools fastest in shade.")
        st.code(__import__("thermal").__doc__, language="text")
    with st.expander("What temperature does to film (and what to do about it)"):
        st.code(THERMAL_NOTES, language="text")
    st.download_button("Download thermal time series CSV", sim.to_csv(index=False), f"{key}_thermal.csv", "text/csv")

# ---- Compare
with tabs[5]:
    have = {k: st.session_state[f"run_{k}"] for k in ("helium", "solar") if f"run_{k}" in st.session_state}
    if len(have) < 1:
        st.info("Run CARI for each balloon type (switch the type in the sidebar and run again) to compare them here.")
    else:
        rows, fig = [], go.Figure()
        for k, r in have.items():
            integ_k = integrate(r["profile"], rate_table(r["res"].table))
            tk = totals(integ_k)
            ps = summarize(r["profile"])
            fa_k = assess(*tk["absorbed_wb"], r["film_spec"])
            th_k = assess_film_thermal(thermal_for(r["profile"]), box, limits)
            rows.append({"balloon": k, "max alt (ft)": round(ps["max_alt_ft"]), "duration (h)": round(ps["duration_h"], 2),
                         "h above 60 kft": round(ps["hours_above_60kft"], 2),
                         "absorbed dose (µGy)": round(tk["absorbed_wb"][0], 2), "± 1σ": round(tk["absorbed_wb"][1], 2),
                         "shelf-days equiv.": round(fa_k["equiv_ground_storage_days"], 1),
                         "% of measurable": f"{fa_k['fraction_of_low']:.0%}",
                         "film min (°C)": round(th_k["film_min_c"], 1),
                         f"h film < {limits.cold_handling_c:.0f} °C": round(th_k["h_below_cold"], 1),
                         "box min (°C)": round(th_k["box_min_c"], 1)})
            fig.add_scatter(x=integ_k["t_min"] / 60, y=integ_k["absorbed_wb_cum"], name=k)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Temperatures use the current payload/thermal settings in the sidebar.")
        fig.update_layout(xaxis_title="Hours since launch", yaxis_title="Cumulative absorbed dose (µGy)", height=400,
                          margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ---- Dataset
with tabs[6]:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        st.info("No dataset yet. Generate it with `python scripts/build_dataset.py`.")
    else:
        f = st.selectbox("File", files, format_func=lambda p: p.name)
        df = pd.read_csv(f)
        st.dataframe(df, use_container_width=True, height=360)
        if (DATA_DIR / "README.md").exists():
            with st.expander("Dataset notes"):
                st.markdown((DATA_DIR / "README.md").read_text(encoding="utf-8"))
