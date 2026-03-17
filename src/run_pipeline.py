#!/usr/bin/env python3
"""
run_pipeline.py

End-to-End Pipeline:
1. Runs the Dit fitting model on experimental data in `data/`, exporting 
   parameters to `results/dit_fit_summary.xlsx` and figures to `figures/dit_fits/`.
2. Runs the lifetime simulation using those newly fitted parameters, 
   exporting figures to `figures/lifetimes/`.

Usage:
    python3 run_pipeline.py
"""

import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_step(command, step_name):
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"RUNNING: {' '.join(command)}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {step_name}")
        sys.exit(result.returncode)

def main():
    print("Starting End-to-End Analysis Pipeline...\n")
    
    # 1. Run Dit modelling
    run_step(
        [sys.executable, os.path.join(script_dir, "model_dit.py"), "--no-show"],
        "Fitting Interface Trap Density (Dit) from Data"
    )
    
    # 2. Run Lifetime comparison using fitted parameters
    run_step(
        [sys.executable, os.path.join(script_dir, "simulate_uv.py")],
        "Simulating Minority Carrier Lifetime vs UV Exposure"
    )
    
    # 3. Generate master summary table
    run_step(
        [sys.executable, os.path.join(script_dir, "generate_summary.py")],
        "Generating Master Model Comparison Summary"
    )
    
    # 4. Simulate impact of doping
    run_step(
        [sys.executable, os.path.join(script_dir, "simulate_impact_doping.py")],
        "Simulating Impact of Surface Doping on Delta J0"
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE.")
    print("Outputs generated:")
    print("  - Dit Fits:     figures/dit_fits/")
    print("  - Lifetimes:    figures/lifetimes/")
    print("  - Fit Data/CSV: results/")
    print("  - Master Table: results/master_model_comparison_summary.xlsx")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
