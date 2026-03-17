#!/usr/bin/env python3
"""
Validation script for the lifetime simulation figures.

This script compares newly generated figures with reference figures to ensure
that the refactoring hasn't changed the scientific results.
"""

import os
import sys
import numpy as np
import PIL
# print(f"Using PIL version: {PIL.__version__}")
import PIL.Image as Image
import matplotlib.pyplot as plt
import argparse

# --- path setup ---
# Ensure we can import from src/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Original parameters from Solver26.ipynb as a reference starting point
SOLVER26_PARAMS = {
    'NC': 2.86e19,
    'NV': 3.11e19,
    'Eg': 1.1246,
    'T': 298.15,            # Derived from kT=0.02569 eV
    'SIGMA0_N': 1e-17,      # Constant cross-section in S26
    'SIGMA0_P': 1e-17,      # Constant cross-section in S26
    'A_N': 0.0,             # Set width to 0 for constant value
    'A_P': 0.0,             # Set width to 0 for constant value
    'TAU_N0_BULK': 25e-3,   # tau_SRH = 25 ms
    'TAU_P0_BULK': 25e-3,   # tau_SRH = 25 ms
    'Ndop_bulk': 2.4e15,
    'W': 0.014,
    'GAUSS_E0': 0.56,
    'GAUSS_SIGMA': 0.18
}

def apply_solver26_overrides():
    """
    Override constants in src.config with values from Solver26.ipynb.
    Does not modify the original config.py file, only the runtime module state.
    """
    try:
        import config
        print("\nApplying overrides from Solver26.ipynb...")
        for key, value in SOLVER26_PARAMS.items():
            if hasattr(config, key):
                old_val = getattr(config, key)
                setattr(config, key, value)
                print(f"  {key}: {old_val} -> {value}")
            else:
                setattr(config, key, value)
                print(f"  {key}: [NEW] -> {value}")
        
        # Recalculate derived constants if necessary
        config.kT = config.kB * config.T
        config.Eg = config.Ec - config.Ev
        # Note: vth_n/p depend on T, but they are often imported individually.
        # We override T, but modules might have already imported old vth values.
        # Re-importing or explicitly updating them if they exist:
        if hasattr(config, 'vth_n'):
            config.vth_n = 2.046e7 * (config.T / 300.0) ** 0.5
        if hasattr(config, 'vth_p'):
            config.vth_p = 1.688e7 * (config.T / 300.0) ** 0.5
            
    except ImportError:
        print("Warning: Could not import src.config for overrides.")

def validate_figure(figure_name, reference_prefix="reference_"):
    """
    Validate a figure by comparing it with a reference figure.
    """
    validation_dir = os.path.join(project_root, "validation/")
    figures_dir = os.path.join(project_root, "figures/")
    reference_dir = os.path.join(project_root, "reference/")
    
    # Validation figures are generated in validation/ folder
    figure_path = os.path.join(validation_dir, figure_name)
    reference_name = reference_prefix + figure_name
    reference_path = os.path.join(reference_dir, reference_name)
    
    # Check if the figure exists
    if not os.path.exists(figure_path):
        # Fallback to figures_dir if not in validation_dir (for non-override runs)
        figure_path = os.path.join(figures_dir, figure_name)
        if not os.path.exists(figure_path):
            print(f"VALIDATION FAILED: {figure_path} does not exist.")
            return False
    
    # If reference doesn't exist, create it
    if not os.path.exists(reference_path):
        print(f"Reference figure {reference_path} does not exist. Creating it...")
        img = Image.open(figure_path)
        img.save(reference_path)
        print(f"Created reference figure {reference_path}")
        return True
    
    # Load the original and new figures
    try:
        reference_img = np.array(Image.open(reference_path))
        new_img = np.array(Image.open(figure_path))
    except Exception as e:
        print(f"VALIDATION FAILED: Error loading images: {str(e)}")
        return False
    
    # Check if the images are identical
    are_identical = np.array_equal(reference_img, new_img)
    
    # Print the result
    if are_identical:
        print(f"VALIDATION PASSED: {figure_name} is identical to reference!")
        return True
    else:
        print(f"VALIDATION FAILED: {figure_name} is different from reference.")
        
        comparison_filename = os.path.join(validation_dir, f"comparison_{figure_name}")
        
        if reference_img.shape == new_img.shape:
            # Calculate the difference if shapes match
            diff = np.abs(reference_img.astype(np.int32) - new_img.astype(np.int32))
            max_diff = np.max(diff)
            print(f"Maximum pixel difference: {max_diff}")
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1); plt.imshow(reference_img); plt.title('Reference'); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(new_img); plt.title('New (Solver26 Params)'); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(diff, cmap='hot'); plt.title('Difference Map'); plt.axis('off'); plt.colorbar()
        else:
            print(f"Images have different shapes: {reference_img.shape} vs {new_img.shape}")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1); plt.imshow(reference_img); plt.title(f'Reference\n{reference_img.shape}'); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(new_img); plt.title(f'New (Solver26 Params)\n{new_img.shape}'); plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_filename)
        print(f"Comparison visualization saved as '{comparison_filename}'")
        
        return False

def validate_all_figures():
    """
    Validate all figures generated by the simulation.
    """
    figures = ["figure 1.png", "figure 2.png", "figure 3.png"]
    all_valid = True
    
    print("\nValidating figures against references...")
    for fig in figures:
        print(f"\nValidating {fig}...")
        if not validate_figure(fig):
            all_valid = False
    
    if all_valid:
        print("\nAll figures match their references. Validation successful!")
    else:
        print("\nSome figures differ from their references. Validation failed.")
    
    return all_valid

def run_simulation():
    """Import and run the figure generation logic."""
    try:
        from figure_panels import generate_figure1, generate_figure2, generate_figure3
        print("\nStarting simulation with current parameters...")
        generate_figure1(output_dir="validation/")
        generate_figure2(output_dir="validation/")
        generate_figure3(output_dir="validation/")
        print("\nSimulation complete (saved to validation/ folder).")
    except ImportError as e:
        print(f"Error importing simulation logic: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Validate simulation results.")
    parser.add_argument("--no-override", action="store_true", help="Do NOT override parameters with Solver26 defaults.")
    args = parser.parse_args()

    if not args.no_override:
        apply_solver26_overrides()
        run_simulation()
    
    validate_all_figures()

if __name__ == "__main__":
    main()
