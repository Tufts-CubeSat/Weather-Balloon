"""Rough film-effect estimates for Kodak Vision3 500T (5219/7219, ECN-2).

READ THIS FIRST: nobody publishes a "mGy to visible fog" number for color negative
film. The thresholds below are order-of-magnitude values inferred from:

  [1] Kodak Motion, "Storage Information": EI 500 film shows ~3x the background-radiation
      change of EI 125 film; fast films stored > 6 months "may be affected".
      https://www.kodak.com/en/motion/page/storage-information
  [2] Kodak TIB 5201 "Baggage X-ray Scanning Effects on Film" (2001): CT checked-bag scanners
      fog film on a single pass.
  [3] NIOSH HHE 2003-0206-3067 (2005), Table 6: L3 CT bag scanner ~156 mrem (~1.5 mGy) per
      pass; carry-on scanners <~1 µSv per pass (4 mrem after 36 passes).
      https://www.cdc.gov/niosh/hhe/reports/pdfs/2003-0206-3067.pdf
  [4] Kodak support: motion-picture origination film should be hand-inspected; fog can
      appear "even after a single X-ray scan".
  [5] UNSCEAR 2000 Annex B: world-average outdoor/indoor ground background ~0.09-0.12 µGy/h
      (terrestrial gamma + sea-level cosmic).
  [6] Slater, Kinard & Firsov, NASA/TP-2000-210193: Vision 500T and ECN films flown 39-132 days
      on Mir (estimated ~10+ mGy) showed obvious grain increase.

Derived thresholds for ISO/EI 500 color negative film (absorbed dose):
  low    0.3 mGy  -> densitometer-measurable fog / grain change (~6 months of ground background)
  likely 1.0 mGy  -> fog visible in shadows / grain increase (below one CT bag-scan pass)
  high   3.0 mGy  -> obvious fog & grain
Sensitivity is scaled with speed using [1] (4x speed -> ~3x effect => exponent ~0.79).

Heavy-ion tracks: no source shows single cosmic-ray tracks visible in color negative film.
Counts are reported for curiosity, not as a visibility prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLDS_MGY_ISO500 = {"low": 0.3, "likely": 1.0, "high": 3.0}
SPEED_EXPONENT = np.log(3) / np.log(4)  # [1]
GROUND_BACKGROUND_UGY_H = 0.10  # [5]
CARRYON_XRAY_UGY_PER_PASS = 1.0  # [3] measured upper-ish value
CT_BAG_SCAN_UGY_PER_PASS = 1500.0  # [3] L3 CT
FLAT_SHEET_FACTOR = 0.5  # crossings of a flat sheet per unit area per unit isotropic fluence
STOCKS = {
    "Vision3 500T (5219/7219)": 500,
    "Vision3 250D (5207/7207)": 250,
    "Vision3 200T (5213/7213)": 200,
    "Vision3 50D (5203/7203)": 50,
}
FORMATS = {  # frame areas in cm^2
    "35 mm, 4-perf motion picture (24.9 x 18.7 mm)": 2.489 * 1.866,
    "35 mm still, 8-perf (36 x 24 mm)": 3.6 * 2.4,
    "16 mm (10.3 x 7.5 mm)": 1.026 * 0.749,
}


@dataclass
class FilmSpec:
    stock: str = "Vision3 500T (5219/7219)"
    width_mm: float = 35.0
    length_m: float = 30.5  # 100 ft daylight spool
    frame_area_cm2: float = FORMATS["35 mm, 4-perf motion picture (24.9 x 18.7 mm)"]
    background_ugy_h: float = GROUND_BACKGROUND_UGY_H

    @property
    def iso(self) -> int:
        return STOCKS[self.stock]

    @property
    def area_cm2(self) -> float:
        return self.width_mm / 10.0 * self.length_m * 100.0


def thresholds_mgy(iso: int) -> dict[str, float]:
    factor = (500.0 / iso) ** SPEED_EXPONENT  # slower film needs more dose
    return {k: v * factor for k, v in THRESHOLDS_MGY_ISO500.items()}


def assess(dose_ugy: float, dose_sigma_ugy: float, spec: FilmSpec,
           fluences: dict[int, float] | None = None) -> dict:
    """Summarize the film impact of a flight dose (µGy) and optional fluences (code -> /cm^2)."""
    thr = thresholds_mgy(spec.iso)
    dose_mgy = dose_ugy / 1000.0
    if dose_mgy < thr["low"]:
        verdict = "Below the measurable-change estimate: expect no visible effect."
    elif dose_mgy < thr["likely"]:
        verdict = "Possibly measurable with a densitometer or careful scan comparison against a control roll."
    elif dose_mgy < thr["high"]:
        verdict = "Visible fog / grain increase is likely."
    else:
        verdict = "Obvious fog and grain are expected."
    res = {
        "dose_ugy": dose_ugy,
        "dose_sigma_ugy": dose_sigma_ugy,
        "thresholds_mgy": thr,
        "fraction_of_low": dose_mgy / thr["low"],
        "fraction_of_likely": dose_mgy / thr["likely"],
        "verdict": verdict,
        "equiv_ground_storage_days": dose_ugy / spec.background_ugy_h / 24.0,
        "equiv_carryon_xray_passes": dose_ugy / CARRYON_XRAY_UGY_PER_PASS,
        "fraction_of_ct_bag_scan": dose_ugy / CT_BAG_SCAN_UGY_PER_PASS,
    }
    if fluences:
        hze = sum(v for c, v in fluences.items() if 14 <= c <= 37)
        res["tracks"] = {
            "protons_per_cm2": fluences.get(7, np.nan),
            "alphas_per_cm2": fluences.get(13, np.nan),
            "hze_per_cm2": hze if any(14 <= c <= 37 for c in fluences) else np.nan,
            "neutrons_per_cm2": fluences.get(1, np.nan),
        }
        # isotropic fluence Phi crosses a flat sheet of area A Phi*A/2 times
        res["tracks_per_frame"] = {k.replace("_per_cm2", ""): v * spec.frame_area_cm2 * FLAT_SHEET_FACTOR
                                   for k, v in res["tracks"].items()}
        res["tracks_per_roll"] = {k.replace("_per_cm2", ""): v * spec.area_cm2 * FLAT_SHEET_FACTOR
                                  for k, v in res["tracks"].items()}
    return res
