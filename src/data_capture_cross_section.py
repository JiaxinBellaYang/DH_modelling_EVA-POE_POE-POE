#!/usr/bin/env python3
"""
Plot dependent capture cross sections (sigma_n, sigma_p) Before and After UV.

Output: figures/capture_cross_section/sigma_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N, SIGMA0_P,
    Ev, Ec, ENERGY_POINTS, lw
)

from physics import (
    create_energy_array,
)

from scatter_data import (
    data_saint_cast, data_werner, data_aberle, data_glunz #data_keith
    
)

def plot_capture_cross_sections():
    print("Calculating and plotting capture cross sections...")
    
    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Constant capture cross sections
    sigma_n = np.full_like(E_array, SIGMA0_N)
    sigma_p = np.full_like(E_array, SIGMA0_P)
    
    # Plot setup
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot Electron and Hole Cross Sections (constant)
    ax.semilogy(E_array, sigma_n, color='blue', linestyle='-', linewidth=lw,
                label=r'$\sigma_n$')
    ax.semilogy(E_array, sigma_p, color='red', linestyle='-', linewidth=lw,
                label=r'$\sigma_p$')
    
    # --- Scatter Plots from Literature ---
    # Saint-Cast (Red Square)
    if data_saint_cast:
        sx, sy = zip(*data_saint_cast)
        ax.semilogy(sx, sy, 's', color='red', label='Saint-Cast et al.')
    
    # Werner (Red Circle)
    if data_werner:
        sx, sy = zip(*data_werner)
        ax.semilogy(sx, sy, 'o', color='red', label='Werner et al.')
    
    # Aberle (Blue Triangle Up)
    if data_aberle:
        sx, sy = zip(*data_aberle)
        ax.semilogy(sx, sy, '^', color='dodgerblue', label='Aberle et al.')
    
    # Glunz (Blue Star)
    if data_glunz:
        sx, sy = zip(*data_glunz)
        ax.semilogy(sx, sy, '*', color='dodgerblue', label='Glunz et al.')

    # Keith (Red Diamond)
    #if data_keith:
        #sx, sy = zip(*data_keith)
        # Using red diamond for Keith
        #ax.semilogy(sx, sy, 'd', color='red', markersize=5, label='Keith et al.')

    
    
    # Formatting
    ax.set_xlabel('Energy ($E_t - E_{v}$) (eV)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Capture Cross Section (cm$^2$)', fontsize=20, fontweight='bold')
    # ax.set_title('Capture Cross Sections (Gaussian)', fontsize=20, fontweight='bold')
    
    ax.set_xlim([0, 1.2])
    # Set ylim to show the full range
    ax.set_ylim([1e-21, 1e-11])
    
    ax.legend(frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 15}, loc='upper left')
    # ax.grid(True, which="major", linestyle='--', alpha=0.4)
    
    # Add band edge markers
    ax.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax.axvline(x=1.12, color='k', linestyle='--', linewidth=2)
    ax.text(0.02, 2e-20, r'$E_v$', fontsize=16, fontweight='bold', verticalalignment='bottom')
    ax.text(1.10, 2e-20, r'$E_c$', fontsize=16, fontweight='bold', verticalalignment='bottom', horizontalalignment='right')

    ax.tick_params(axis='both', which='major', direction='in', length=10,
                   labelsize=16, width=2, pad=8, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=5,
                   labelsize=16, width=1.5, top=True, right=True)
    
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_weight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    fig.tight_layout()

    # Save
    out_dir = os.path.join(os.path.dirname(script_dir), "figures", "capture_cross_section")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sigma_comparison_1.png")
    
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    plot_capture_cross_sections()
