"""
Reproduces every result in the report, in dependency order.

    python experiments/run_all.py                    # full dataset (~25 min)
    python experiments/run_all.py --quick            # smaller/faster settings (results are NOT report-grade)
    python experiments/run_all.py --skip cross_validation,ablation

DistilBERT is trained on Colab (notebooks/bert_finetune_colab.ipynb); this
script merges its predictions if results/predictions/distilbert.csv exists.
Ends by exporting the report tables (docs/RESULTS.md) and figures (docs/figures/).
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, script, extra args for --quick)
STEPS = [
    ("dataset_profile", "run_dataset_profile.py", []),
    ("model_comparison", "run_model_comparison.py", ["--max-rows", "10000"]),
    ("cross_validation", "run_cross_validation.py", ["--quick"]),
    ("significance", "run_significance.py", []),
    ("explainability_eval", "run_explainability_eval.py", ["--n", "100"]),
    ("ablation", "run_ablation.py", ["--quick"]),       # after explainability: uses its SHAP ranking
    ("summary_comparison", "run_summary_comparison.py", ["--quick"]),
    ("export_tables", "export_tables.py", []),
    ("export_figures", "export_figures.py", []),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip", default="", help="comma-separated step names")
    args = parser.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    if not os.path.exists(os.path.join("results", "predictions", "distilbert.csv")):
        print("NOTE: no DistilBERT predictions found; the model comparison will not include BERT.\n"
              "      Run notebooks/bert_finetune_colab.ipynb on Colab to add it.\n")
    if args.quick:
        print("NOTE: --quick overwrites results/ with smaller-sample numbers. Re-run without --quick for the report.\n")

    timings = []
    for name, script, quick_args in STEPS:
        if name in skip:
            print(f"--- skipping {name}")
            continue
        cmd = [sys.executable, os.path.join(HERE, script)] + (quick_args if args.quick else [])
        print(f"--- {name}: {' '.join(cmd[1:])}", flush=True)
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        timings.append((name, time.perf_counter() - start))

    print("\nDone. Time per step:")
    for name, seconds in timings:
        print(f"  {name:22s} {seconds / 60:5.1f} min")


if __name__ == "__main__":
    main()
