"""
run_all.py — single entry point for the EWC reproduction project.

Usage:
    python run_all.py --figure A
    python run_all.py --figure B
    python run_all.py --figure C
    python run_all.py --figure all
"""

import argparse
import subprocess
import sys
import os

SCRIPTS = {
    "A": "main-graph A.py",
    "B": "main-graph_B.py",
    "C": "main-GraphC.py",
}

OUTPUTS = {
    "A": "figure_2A_online_perfect.png",
    "B": "figure_B_lambda_12000.png",
    "C": "figure_C_improved.png",
}


def run_figure(fig: str):
    script = SCRIPTS[fig]
    expected_output = OUTPUTS[fig]

    print(f"\n{'='*60}")
    print(f"  Running Figure {fig}  →  {script}")
    print(f"  Expected output: {expected_output}")
    print(f"{'='*60}\n")

    if not os.path.exists(script):
        print(f"[ERROR] Script not found: {script}")
        sys.exit(1)

    result = subprocess.run([sys.executable, script], check=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Figure {fig} exited with code {result.returncode}")
        sys.exit(result.returncode)

    if os.path.exists(expected_output):
        print(f"\n[OK] Output saved: {expected_output}")
    else:
        print(f"\n[WARNING] Expected output not found: {expected_output}")
        print("         The script may have saved under a different name.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run EWC reproduction scripts for Figures A, B, or C."
    )
    parser.add_argument(
        "--figure",
        choices=["A", "B", "C", "all"],
        required=True,
        help="Which figure to reproduce (A, B, C, or all).",
    )
    args = parser.parse_args()

    figures = ["A", "B", "C"] if args.figure == "all" else [args.figure]

    for fig in figures:
        run_figure(fig)

    print("Done.")


if __name__ == "__main__":
    main()
