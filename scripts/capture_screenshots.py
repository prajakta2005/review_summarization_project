"""
Captures full-page screenshots of every dashboard page for the report.

    pip install -r requirements-dev.txt
    python scripts/capture_screenshots.py            # uses installed Chrome/Edge if available
    python -m playwright install chromium             # only if neither is installed

Starts Streamlit headless on port 8599, selects the full Kaggle dataset on
Home, then visits every page through the sidebar (so the data choice is
kept), and saves docs/screenshots/<NN>_<name>.png. Re-run after any UI
change; the gallery in docs/SYSTEM_OUTPUT.md points at these files.
"""

import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

PORT = 8599
BASE = f"http://localhost:{PORT}"
OUT_DIR = "docs/screenshots"
WIDTH = 1440


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{BASE}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Streamlit didn't start on port %d" % PORT)


def launch_browser(p):
    for channel in ("chrome", "msedge", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:
            continue
    raise RuntimeError("No browser found: run `python -m playwright install chromium`.")


def wait_until_idle(page, settle: float = 1.5, timeout: float = 180):
    """Streamlit shows a status widget / spinners while the script runs."""
    deadline = time.time() + timeout
    quiet_since = None
    while time.time() < deadline:
        busy = page.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"]').count() > 0
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since >= settle:
            return
        time.sleep(0.25)


def shoot(page, name: str):
    """Full-page shot: Streamlit scrolls inside its main container, so grow
    the viewport to the content height first."""
    height = page.evaluate(
        "() => { const el = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
        " return el ? el.scrollHeight + 160 : document.body.scrollHeight; }")
    page.set_viewport_size({"width": WIDTH, "height": max(900, min(int(height), 12000))})
    time.sleep(0.8)  # let charts re-layout
    path = os.path.join(OUT_DIR, f"{name}.png")
    page.screenshot(path=path, full_page=True)
    page.set_viewport_size({"width": WIDTH, "height": 900})
    print("saved", path, flush=True)


def go_to(page, url_path: str):
    page.locator(f'[data-testid="stSidebarNav"] a[href$="/{url_path}"]').first.click()
    time.sleep(0.5)
    wait_until_idle(page)


def click_tab(page, label: str):
    page.get_by_role("tab", name=label).first.click()
    wait_until_idle(page, settle=1.0)


def click_button(page, label: str):
    page.get_by_role("button", name=label).first.click()
    wait_until_idle(page, settle=1.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    server = start_server()
    try:
        with sync_playwright() as p:
            browser = launch_browser(p)
            page = browser.new_page(viewport={"width": WIDTH, "height": 900})
            page.goto(BASE)
            wait_until_idle(page)
            page.get_by_text("Kaggle fake-reviews dataset", exact=False).first.click()
            wait_until_idle(page)
            shoot(page, "01_home")

            go_to(page, "Live_Demo")
            click_button(page, "Genuine example")
            shoot(page, "02_live_demo_genuine")
            click_button(page, "Fake example")
            shoot(page, "03_live_demo_fake")

            go_to(page, "Explainability")
            click_tab(page, "Single Review")
            shoot(page, "04_explainability_waterfall")

            go_to(page, "Summarization")
            shoot(page, "05_summarization_side_by_side")

            go_to(page, "Review_Explorer")
            shoot(page, "06_review_explorer")

            for path, name in [("Model_Comparison", "07_model_comparison"), ("Performance", "08_performance"),
                               ("Ablation_Study", "09_ablation"), ("Experimental_Results", "10_experimental_results"),
                               ("Dataset", "11_dataset"), ("Novelty", "12_novelty")]:
                if page.locator(f'[data-testid="stSidebarNav"] a[href$="/{path}"]').count() == 0:
                    print(f"skipping {name}: page not found")
                    continue
                go_to(page, path)
                shoot(page, name)
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
