"""
Exports the report figures (F1-F10) as PNGs from results/*.json.

    python experiments/export_figures.py

Writes docs/figures/<name>.png at 2x scale. Uses Plotly + kaleido>=1.0
(which drives an installed Chrome/Edge; run `kaleido_get_chrome` if none is found).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from results_summary import load_all_results
from report_figures import report_figures

FIGURES_DIR = "docs/figures"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figs = report_figures(load_all_results())
    for name, (caption, fig) in figs.items():
        fig.update_layout(template="plotly_white", width=1100, height=max(fig.layout.height or 500, 450))
        path = os.path.join(FIGURES_DIR, f"{name}.png")
        fig.write_image(path, scale=2)
        print(f"wrote {path}  ({caption})")
    print(f"{len(figs)} figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
