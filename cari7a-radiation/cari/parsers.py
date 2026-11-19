"""Parsers for CARI-7A output files."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NUM_START = re.compile(r"^\s*-?\d")

def _read(path_or_text: str | Path) -> str:
    """Accept a Path (read it) or the file's text."""
    if isinstance(path_or_text, Path):
        return path_or_text.read_text(errors="replace")
    return str(path_or_text)


ANS_COLUMNS = ["lat", "lon", "altitude", "alt_unit", "date", "hour", "vcr_gv",
               "particle", "rate", "sigma", "unit", "quantity"]


def parse_ans(path_or_text: str | Path) -> pd.DataFrame:
    """Parse a .ANS (location results) file into a DataFrame, one row per data line.

    Comment echoes, headers and the 'COMMENT OR IMPROPER FORMAT' junk lines are skipped.
    Rows keep the order of the data lines in the .LOC file.
    """
    text = _read(path_or_text)
    rows = []
    for line in text.splitlines():
        if not _NUM_START.match(line):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 12:
            continue
        rows.append({
            "lat": float(parts[0]), "lon": float(parts[1]), "altitude": float(parts[2]),
            "alt_unit": parts[3], "date": parts[4], "hour": int(parts[5]),
            "vcr_gv": float(parts[6]), "particle": parts[7], "rate": float(parts[8]),
            "sigma": float(parts[9]), "unit": parts[10], "quantity": ",".join(parts[11:]).strip(),
        })
    return pd.DataFrame(rows, columns=ANS_COLUMNS)


def parse_deg_dat(path_or_text: str | Path) -> pd.DataFrame:
    """Parse the long-form .DAT output of a waypoint (.DEG) flight run."""
    text = _read(path_or_text)
    rows = []
    for line in text.splitlines():
        parts = line.split()
        # v4.2 appends year/month/day/hour columns after the 6 documented ones
        if len(parts) >= 6 and _NUM_START.match(line):
            try:
                lat, lon, depth, step, rate, total = map(float, parts[:6])
            except ValueError:
                continue
            rows.append({"lat": lat, "lon": lon, "depth_g_cm2": depth, "step": int(step),
                         "dose_rate": rate, "total_dose": total})
    return pd.DataFrame(rows)


def parse_sum(path_or_text: str | Path) -> dict:
    """Parse a .SUM file: returns the total dose and its description."""
    text = _read(path_or_text)
    m = re.search(r"([-\d.]+E[+-]\d+)\s+TOTAL\s+(.*)", text)
    if not m:
        raise ValueError("No TOTAL line found in SUM output")
    return {"total": float(m.group(1)), "description": m.group(2).strip()}
