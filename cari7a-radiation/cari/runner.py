"""Run CARI-7A headless from Python.

CARI-7A is a Fortran console program. It reads CARI.INI and DEFAULT.INP from its working
directory and writes outputs next to its inputs, so we never touch the user's install.
Instead the needed program files are copied once into a private working directory
(`.cari_work/` by default). Each run rewrites the control files there, under a file lock.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .locfile import Query, build_loc, PARTICLES
from .parsers import parse_ans, parse_deg_dat, parse_sum

PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARI_DIR = r"C:\Users\goldm\Downloads\cari_7Aexe\CARI_7A_DVD"
_COPY_DIRS = ["AIRPORTS", "ATMOSPHERE", "CUTOFFS", "FORBUSH", "GCR_MODELS", "KP_INDEX",
              "SHOWERS", "SOLARMOD", "SUNSPOTS"]
_COPY_FILES = ["CARI.INI", "DEFAULT.INP", "FROMUSER.DAT", "VEHICLE.DAT", "HELP.TXT"]
SECONDS_PER_LINE = 0.35  # measured ~0.13 s/line on a laptop; generous for timeouts


@dataclass
class CariConfig:
    cari_dir: Path = Path(DEFAULT_CARI_DIR)
    exe: str = "CARI-7A.exe"
    work_dir: Path = PKG_ROOT / ".cari_work"
    runs_dir: Path = PKG_ROOT / "runs"

    @classmethod
    def load(cls, path: Path | None = None) -> "CariConfig":
        path = path or PKG_ROOT / "config.toml"
        cfg = cls()
        if path.exists():
            data = tomllib.loads(path.read_text())
            if "cari_dir" in data:
                cfg.cari_dir = Path(data["cari_dir"])
            if "exe" in data:
                cfg.exe = data["exe"]
            if "work_dir" in data:
                cfg.work_dir = (PKG_ROOT / data["work_dir"]).resolve()
            if "runs_dir" in data:
                cfg.runs_dir = (PKG_ROOT / data["runs_dir"]).resolve()
        if env := os.environ.get("CARI_DIR"):
            cfg.cari_dir = Path(env)
        return cfg


@dataclass
class Shielding:
    """Payload enclosure shielding, passed to CARI via VEHICLE.DAT (g/cm^2)."""
    aluminum_g_cm2: float = 0.0
    hdpe_g_cm2: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.aluminum_g_cm2 > 0 or self.hdpe_g_cm2 > 0


@dataclass
class LocRunResult:
    table: pd.DataFrame  # one row per query: key, alt_ft, rate, sigma, unit, particle, vcr_gv
    run_dir: Path
    elapsed_s: float
    stdout_tail: str
    meta: dict = field(default_factory=dict)


class CariRunner:
    def __init__(self, config: CariConfig | None = None):
        self.cfg = config or CariConfig.load()

    # ---- setup --------------------------------------------------------------
    def check(self) -> list[str]:
        """Return a list of problems (empty if OK)."""
        problems = []
        src = self.cfg.cari_dir
        if not (src / self.cfg.exe).exists():
            problems.append(f"CARI-7A executable not found at {src / self.cfg.exe}. "
                            "Set cari_dir in config.toml or the CARI_DIR environment variable.")
        return problems

    def ensure_workdir(self) -> Path:
        wd = self.cfg.work_dir
        exe_src = self.cfg.cari_dir / self.cfg.exe
        exe_dst = wd / self.cfg.exe
        if exe_dst.exists() and all((wd / d).is_dir() for d in _COPY_DIRS):
            return wd
        if not exe_src.exists():
            raise FileNotFoundError("; ".join(self.check()))
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "DIAGNOSE").mkdir(exist_ok=True)
        for d in _COPY_DIRS:
            if not (wd / d).is_dir():
                shutil.copytree(self.cfg.cari_dir / d, wd / d)
        for f in _COPY_FILES:
            if (self.cfg.cari_dir / f).exists():
                shutil.copy2(self.cfg.cari_dir / f, wd / f)
        shutil.copy2(exe_src, exe_dst)
        return wd

    def _lock(self) -> FileLock:
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.cfg.work_dir / ".run.lock"))

    # ---- control files ---------------------------------------------------------
    @staticmethod
    def _set_ini(text: str, settings: dict[str, str]) -> str:
        """Set values in CARI.INI keeping its fixed A10,A2,A12 format."""
        out = []
        seen = set()
        for line in text.splitlines():
            key = line[:10].strip()
            if key in settings and line[10:12] == "= ":
                line = line[:12] + f"{settings[key]:<12}" + line[24:]
                seen.add(key)
            out.append(line)
        missing = set(settings) - seen
        if missing:
            raise RuntimeError(f"CARI.INI is missing keys: {sorted(missing)}")
        return "\r\n".join(out) + "\r\n"

    def _write_controls(self, wd: Path, target: str, shielding: Shielding,
                        date: str = "0000/00/00", particle: int = 0, dose: int = 5) -> None:
        ini_src = self.cfg.cari_dir / "CARI.INI"
        ini_text = ini_src.read_text(errors="replace") if ini_src.exists() else (wd / "CARI.INI").read_text()
        ini = self._set_ini(ini_text, {
            "DIAGNOSE": "NO!", "MENUS": "NO!", "SUPERPOS": "NO!", "GCR_MODEL": "4",
            "NVCR": "0", "CARI7RPT": "NO!", "VEHICLE": "YES" if shielding.enabled else "NO!",
        })
        (wd / "CARI.INI").write_text(ini, newline="")
        (wd / "DEFAULT.INP").write_text(
            f"{date}\r\n 0 \r\n {particle} \r\n {dose} \r\n{target}\r\n"
            "ONLY THE FIRST 5 LINES ARE READ (written by cari7a-radiation)\r\n", newline="")
        (wd / "VEHICLE.DAT").write_text(
            f" {shielding.aluminum_g_cm2:.4f}\r\n {shielding.hdpe_g_cm2:.4f}\r\n 0.\r\nEND OF DATA\r\n",
            newline="")
        (wd / "FROMUSER.DAT").write_text(
            " -1, 'Kp index'\r\n -1.0000, 'Vertical Cutoff Rigidity'\r\n"
            " -1.0000, 'Forbush scale factor'\r\n", newline="")

    def _exec(self, wd: Path, n_lines: int) -> tuple[str, float]:
        t0 = time.time()
        proc = subprocess.run(
            [str(wd / self.cfg.exe)], cwd=wd, input="\n" * 20, capture_output=True, text=True,
            timeout=120 + SECONDS_PER_LINE * n_lines * 3,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or "forrtl" in out:
            raise RuntimeError(f"CARI-7A failed (exit {proc.returncode}):\n{out[-2000:]}")
        return out, time.time() - t0

    def _new_run_dir(self, tag: str) -> Path:
        d = self.cfg.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- location runs -----------------------------------------------------------
    def run_locations(self, lat: float, lon: float, queries: list[Query], year: int, month: int,
                      day: int = 0, shielding: Shielding | None = None, tag: str = "loc") -> LocRunResult:
        shielding = shielding or Shielding()
        wd = self.ensure_workdir()
        name = "R" + uuid.uuid4().hex[:7].upper()
        loc_text = build_loc(lat, lon, queries, year, month, day)
        with self._lock():
            for old in wd.glob("R???????.*"):
                old.unlink(missing_ok=True)
            (wd / f"{name}.LOC").write_text(loc_text, newline="")
            self._write_controls(wd, f"{name}.LOC", shielding)
            out, elapsed = self._exec(wd, len(queries))
            ans_path = wd / f"{name}.ANS"
            if not ans_path.exists():
                raise RuntimeError(f"CARI-7A produced no .ANS output.\n{out[-2000:]}")
            ans_text = ans_path.read_text(errors="replace")
            run_dir = self._new_run_dir(tag)
            shutil.copy2(wd / f"{name}.LOC", run_dir / "input.LOC")
            shutil.copy2(ans_path, run_dir / "output.ANS")
            for f in ("CARI.INI", "DEFAULT.INP", "VEHICLE.DAT"):
                shutil.copy2(wd / f, run_dir / f)
            (wd / f"{name}.LOC").unlink(missing_ok=True)
            ans_path.unlink(missing_ok=True)

        ans = parse_ans(ans_text)
        if len(ans) != len(queries):
            raise RuntimeError(f"Expected {len(queries)} result rows, parsed {len(ans)}; see {run_dir}")
        table = pd.DataFrame({
            "key": [q.key for q in queries],
            "alt_ft": [q.alt_ft for q in queries],
            "dose_code": [q.dose_code for q in queries],
            "particle_code": [q.particle for q in queries],
        })
        table = pd.concat([table, ans[["rate", "sigma", "unit", "particle", "vcr_gv", "quantity"]]], axis=1)
        table.to_csv(run_dir / "results.csv", index=False)
        meta = {"lat": lat, "lon": lon, "year": year, "month": month, "day": day,
                "shielding": vars(shielding), "n_queries": len(queries),
                "heliocentric_potential": self.heliocentric_potential(year, month)}
        return LocRunResult(table, run_dir, elapsed, out[-1500:], meta)

    # ---- waypoint (DEG) runs: used to cross-check the LOC integration -------------
    def run_deg(self, waypoints: list[tuple[float, float, float, float]], year: int, month: int,
                particle: int = 0, dose: int = 5, shielding: Shielding | None = None,
                name_tag: str = "deg") -> dict:
        """waypoints: (lat, lon, alt_ft, t_min). Returns {'dat': df, 'sum': {...}, 'run_dir': Path}."""
        shielding = shielding or Shielding()
        wd = self.ensure_workdir()
        name = "R" + uuid.uuid4().hex[:7].upper()
        lines = [f"{month:02d}/{year:04d}, {name_tag[:12]}", "DEG MIN N/S DEG MIN E/W FEET TIME(MIN)"]
        for lat, lon, alt_ft, t in waypoints:
            la, lo = abs(lat), abs(lon)
            lines.append(f"{int(la)}, {(la - int(la)) * 60:.2f}, {'N' if lat >= 0 else 'S'}, "
                         f"{int(lo)}, {(lo - int(lo)) * 60:.2f}, {'E' if lon >= 0 else 'W'}, "
                         f"{int(round(alt_ft))}, {int(round(t))}")
        with self._lock():
            for old in wd.glob("R???????.*"):
                old.unlink(missing_ok=True)
            (wd / f"{name}.DEG").write_text("\r\n".join(lines) + "\r\n", newline="")
            self._write_controls(wd, f"{name}.DEG", shielding, particle=particle, dose=dose)
            out, _ = self._exec(wd, int(waypoints[-1][3]) + 10)
            run_dir = self._new_run_dir(name_tag)
            for ext in ("DEG", "DAT", "SUM"):
                p = wd / f"{name}.{ext}"
                if p.exists():
                    shutil.copy2(p, run_dir / f"flight.{ext}")
                    p.unlink()
        return {"dat": parse_deg_dat(run_dir / "flight.DAT"), "sum": parse_sum(run_dir / "flight.SUM"),
                "run_dir": run_dir}

    # ---- solar modulation info ---------------------------------------------------
    def heliocentric_potential(self, year: int, month: int) -> dict:
        """Heliocentric potential (MV) CARI will use, and whether it is a fallback."""
        src = self.cfg.cari_dir / "SOLARMOD"
        table: dict[tuple[int, int], float] = {}
        for fname in ("MV-DATES.L99", "MORDATES.2K"):  # MORDATES takes precedence
            p = src / fname
            if not p.exists():
                continue
            for line in p.read_text(errors="replace").splitlines():
                m = re.match(r"\s*(\d{1,2})/(\d{4}),\s*(-?\d+)", line)
                if m:
                    mo, yr = int(m.group(1)), int(m.group(2))
                    # skip annual-average rows (month 00) and placeholder rows (e.g. 00/4500, 12/2112)
                    if 1 <= mo <= 12 and 1958 <= yr <= datetime.now().year + 1:
                        table[(yr, mo)] = float(m.group(3))
        if not table:
            return {"mv": None, "source": "unknown"}
        if (year, month) in table:
            return {"mv": table[(year, month)], "source": f"{year}/{month:02d}", "fallback": False}
        latest = max(table)
        return {"mv": table[latest], "source": f"{latest[0]}/{latest[1]:02d} (latest available)",
                "fallback": True}


# ---- query planning ------------------------------------------------------------------
DETAIL_LEVELS = {
    "fast": {"flux_particles": [0, 1, 7, 13], "flux_step_ft": 5000},
    "standard": {"flux_particles": list(range(0, 38)), "flux_step_ft": 5000},
    "fine": {"flux_particles": list(range(0, 38)), "flux_step_ft": 2000},
}
DOSE_STEP_FT = 2000


def altitude_grid(max_alt_ft: float, step_ft: float, extra: list[float] | None = None) -> list[float]:
    top = step_ft * np.ceil(max_alt_ft / step_ft)
    grid = set(np.arange(0.0, top + step_ft / 2, step_ft).round(0).tolist())
    for e in extra or []:
        grid.add(round(float(e), 0))
    return sorted(grid)


def plan_queries(max_alt_ft: float, detail: str = "standard", extra_alts: list[float] | None = None) -> list[Query]:
    lvl = DETAIL_LEVELS[detail]
    qs = []
    for a in altitude_grid(max_alt_ft, DOSE_STEP_FT, extra_alts):
        for d in (5, 6, 7):
            qs.append(Query(a, d, 0))
    for a in altitude_grid(max_alt_ft, lvl["flux_step_ft"], extra_alts):
        for p in lvl["flux_particles"]:
            qs.append(Query(a, 1, p))
    return qs


def rate_table(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a run table into {key: DataFrame(alt_ft, rate, sigma)} sorted by altitude."""
    out = {}
    for key, g in table.groupby("key"):
        out[key] = g[["alt_ft", "rate", "sigma"]].sort_values("alt_ft").reset_index(drop=True)
    return out


def particle_label(key: str) -> str:
    if key.startswith("flux_P"):
        return PARTICLES[int(key[6:])]
    return key
