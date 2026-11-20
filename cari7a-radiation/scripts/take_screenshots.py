"""Regenerate the README screenshots of the Balloon Radiation Planner.

Drives a running copy of the Streamlit app (``app.py``) with headless Edge or
Chrome through Playwright, clicks through the sidebar and tabs, and writes PNGs
to ``docs/screenshots/``. It does not import anything from the app.

Dependencies (keep them out of the app's venv and requirements.txt):

    py -m venv %TEMP%\\shots-venv
    %TEMP%\\shots-venv\\Scripts\\python -m pip install playwright pillow

Playwright uses the system Edge (``--channel msedge``, default) or Chrome
(``--channel chrome``), so ``playwright install`` is not needed. Pillow is
optional; when present, images larger than ``--max-kb`` are re-encoded with a
256-colour palette to keep them small.

Usage:

    # 1. start the app (in another terminal), e.g.
    .venv\\Scripts\\streamlit run app.py --server.port 8517
    # 2. take the screenshots
    %TEMP%\\shots-venv\\Scripts\\python scripts\\take_screenshots.py --url http://localhost:8517

Options: ``--out DIR`` (default docs/screenshots next to this repo),
``--channel msedge|chrome``, ``--headed`` to watch, ``--skip-runs`` to take
only the shots that do not need CARI-7A (overview, thermal, dataset).

Two CARI-7A runs are made at "fast" detail (helium, then solar; roughly
30-60 s each). Everything happens in a single browser session because the
Compare tab reads both runs from Streamlit's per-session state.

Files written:
    overview.png   Profile tab with the sidebar (helium latex)
    dose.png       Dose tab, solar balloon
    particles.png  Particles tab, solar balloon
    film.png       Film tab, solar balloon
    thermal.png    Thermal tab, solar balloon
    compare.png    Compare tab with helium and solar runs
    dataset.png    Dataset tab
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PWTimeout, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
W, H = 1400, 900
MAX_FULL_H = 2600  # cap for "full page" captures

SOLAR = "Solar (black envelope)"
HELIUM = "Helium latex"


def wait_idle(page: Page, settle: float = 1.5, timeout: float = 120) -> None:
    """Wait until Streamlit is not running a script and nothing is spinning."""
    deadline = time.time() + timeout
    time.sleep(0.5)
    while time.time() < deadline:
        busy = page.evaluate(
            """() => {
                const s = document.querySelector('[data-testid=stStatusWidget]');
                const running = s && /running/i.test(s.innerText || '');
                const spin = document.querySelector('[data-testid=stSpinner]');
                const skel = document.querySelector('[data-testid=stSkeleton]');
                return Boolean(running || spin || skel);
            }"""
        )
        if not busy:
            break
        time.sleep(0.5)
    time.sleep(settle)


def has_error(page: Page) -> bool:
    return page.locator("[data-testid=stException]").count() > 0


def sidebar_choose(page: Page, label: str) -> None:
    """Click a radio option (by its visible text) in the sidebar."""
    opt = page.locator("[data-testid=stSidebar] label").filter(has_text=label).first
    opt.scroll_into_view_if_needed()
    opt.click()
    wait_idle(page)


def open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()
    wait_idle(page, settle=2.0)
    page.evaluate("() => { const m = document.querySelector('[data-testid=stMain]'); if (m) m.scrollTo(0, 0); }")


def run_cari(page: Page) -> None:
    sidebar_choose(page, "fast")
    btn = page.get_by_role("button", name="Run CARI-7A")
    btn.scroll_into_view_if_needed()
    btn.click()
    try:
        page.locator("[data-testid=stSpinner]").first.wait_for(state="visible", timeout=15_000)
    except PWTimeout:
        pass  # very fast run, or spinner already gone
    page.locator("[data-testid=stSpinner]").first.wait_for(state="detached", timeout=600_000)
    wait_idle(page, settle=2.0)
    if has_error(page) or page.get_by_text("CARI-7A run failed").count():
        print("  warning: the page shows an error after the CARI run", file=sys.stderr)


def shrink(path: Path, max_kb: int) -> None:
    if path.stat().st_size <= max_kb * 1024:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    img = Image.open(path).convert("RGB")
    img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def shoot(page: Page, path: Path, full: bool, max_kb: int) -> None:
    """Screenshot the viewport, or the whole main column when ``full``."""
    for attempt in range(3):
        if not has_error(page):
            break
        print(f"  error on page, waiting for a hot reload (attempt {attempt + 1})", file=sys.stderr)
        time.sleep(10)
        wait_idle(page)
    # show the top of the sidebar (site, date, balloon inputs) in every shot
    page.evaluate(
        """() => { for (const sel of ['[data-testid=stSidebarContent]', '[data-testid=stSidebarUserContent]',
                                      '[data-testid=stSidebar]'])
                     { const e = document.querySelector(sel); if (e) e.scrollTo(0, 0); } }"""
    )
    if full:
        h = page.evaluate(
            """() => { const m = document.querySelector('[data-testid=stMain]')
                         || document.querySelector('section.main');
                       return m ? m.scrollHeight : document.body.scrollHeight; }"""
        )
        page.set_viewport_size({"width": W, "height": max(H, min(int(h) + 20, MAX_FULL_H))})
        time.sleep(2.0)  # let plotly re-layout
    # hide the top-right toolbar ("Deploy", "File change. Rerun") in the image only
    page.screenshot(path=str(path), style="[data-testid=stHeader] { visibility: hidden !important; }")
    if full:
        page.set_viewport_size({"width": W, "height": H})
        time.sleep(1.0)
    shrink(path, max_kb)
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
          f"({path.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8517")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    ap.add_argument("--channel", default="msedge", choices=["msedge", "chrome"])
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--skip-runs", action="store_true", help="skip CARI runs (no dose/particles/film/compare)")
    ap.add_argument("--max-kb", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    out = lambda name: a.out / f"{name}.png"  # noqa: E731

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=a.channel, headless=not a.headed)
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1,
                                color_scheme="light")
        page.goto(a.url, wait_until="domcontentloaded", timeout=60_000)
        page.get_by_role("tab", name="Profile", exact=True).wait_for(timeout=180_000)
        wait_idle(page, settle=3.0)

        print("helium latex")
        sidebar_choose(page, HELIUM)
        open_tab(page, "Profile")
        shoot(page, out("overview"), full=False, max_kb=a.max_kb)
        if not a.skip_runs:
            print("  running CARI-7A (fast)...")
            run_cari(page)

        print("solar")
        sidebar_choose(page, SOLAR)
        if not a.skip_runs:
            print("  running CARI-7A (fast)...")
            run_cari(page)
            for tab in ("Dose", "Particles", "Film"):
                open_tab(page, tab)
                shoot(page, out(tab.lower()), full=True, max_kb=a.max_kb)
        open_tab(page, "Thermal")
        shoot(page, out("thermal"), full=True, max_kb=a.max_kb)
        if not a.skip_runs:
            open_tab(page, "Compare")
            shoot(page, out("compare"), full=True, max_kb=a.max_kb)
        open_tab(page, "Dataset")
        shoot(page, out("dataset"), full=False, max_kb=a.max_kb)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
