#!/usr/bin/env python3
"""
Main entry point — generate all standard figures and validate results.

Usage:
    python3 src/generate_figures.py
"""

import os
import sys
import subprocess

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from figure_panels import generate_figure1, generate_figure2, generate_figure3


# ============================================================================
# VALIDATION
# ============================================================================

def validate_results():
    """Check generated figures exist, then run the comprehensive validator."""
    figures_dir = "figures/"
    figures = ["figure 1.png", "figure 2.png", "figure 3.png"]
    all_valid = True

    print("\nPerforming basic validation...")
    for fig in figures:
        fig_path = os.path.join(figures_dir, fig)
        if not os.path.exists(fig_path):
            print(f"  ✗ {fig_path} is missing")
            all_valid = False
        elif os.path.getsize(fig_path) == 0:
            print(f"  ✗ {fig_path} is empty (zero size)")
            all_valid = False
        else:
            print(f"  ✓ {fig_path} exists and has non-zero size")

    if not all_valid:
        print("Basic validation failed: some figures are missing or empty.")
        return False

    print("Basic validation successful.\n")
    print("Running comprehensive validation...")
    try:
        result = subprocess.run(
            [sys.executable, 'validation/validate_results.py'],
            capture_output=True, text=True, check=False
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Validation script returned error code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
        if "All figures match their references" in result.stdout:
            print("Comprehensive validation successful!")
            return True
        if "Reference figure" in result.stdout and "does not exist" in result.stdout:
            print("Reference figures created. Run again to validate.")
            return True
        print("Comprehensive validation failed.")
        return False
    except Exception as e:
        print(f"Error running validation script: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Starting lifetime simulation...\n")
    generate_figure1()
    generate_figure2()
    generate_figure3()
    print("\nAll figures generated successfully.")
    validate_results()


if __name__ == "__main__":
    main()
