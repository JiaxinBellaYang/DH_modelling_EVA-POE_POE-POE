#!/usr/bin/env python3
"""
Simulate Delta J0 vs Surface Doping using 3-peak Gaussian parameters.

Output: figures/impact_of_doping/figure_delta_j0_vs_doping.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import e as elementary_charge

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N_BEFORE, A_N_BEFORE, E0_N_BEFORE, SIGMA0_P_BEFORE, A_P_BEFORE, E0_P_BEFORE,
    SIGMA0_N_AFTER, A_N_AFTER, E0_N_AFTER, SIGMA0_P_AFTER, A_P_AFTER, E0_P_AFTER,
    Ev, Ec, T,
    dop_type_bulk, dop_type_emitter,
    Ndop_bulk, W,
    ENERGY_POINTS, lw,
)

from physics import (
    surfaceLifetime, Dig_func, ni_func,
    create_energy_array, calculate_gaussian_sigma,
)

from helpers import (
    equilibrium_concentrations, build_dit_profile,
)

def simulate_impact_of_doping():
    print("Starting calculations for Delta J0 vs Surface Doping...")

    results_file = os.path.join(os.path.dirname(script_dir), "results", "3peak", "uv_fit_summary_3peak.xlsx")
    if not os.path.exists(results_file):
        print(f"Error: Could not find 3-peak fit summary at {results_file}")
        sys.exit(1)
        
    df = pd.read_excel(results_file)
    
    # Extract Before and After UV parameters
    try:
        row_before = df[df["Dataset"].astype(str).str.contains("Before", case=False)].iloc[0]
        row_after = df[df["Dataset"].astype(str).str.contains("After", case=False)].iloc[0]
    except IndexError:
        print("Error: Could not find Before/After rows in summary.")
        sys.exit(1)
        
    def row_to_params(r):
        return {
            # "Qfix": float(r["Qfix (cm^-2)"]), # Ignored - using fixed Qfix
            "multiple_peaks": True,
            "Dit0_v": r["Dit0_v"], "Ev_trap": r["Ev_trap"],
            "Dit0_c": r["Dit0_c"], "Ec_trap": r["Ec_trap"],
            "Dit0_g1": r["Dit0_g1"], "E0_1": r["E0_1"], "sigma_1": r["sigma_1"],
            "Dit0_g2": r["Dit0_g2"], "E0_2": r["E0_2"], "sigma_2": r["sigma_2"],
            "Dit0_g3": r["Dit0_g3"], "E0_3": r["E0_3"], "sigma_3": r["sigma_3"],
        }
        
    params_ini = row_to_params(row_before)
    params_fin = row_to_params(row_after)

    # --- Fixed Qfix Values (Same as simulate_uv.py) ---
    params_ini["Qfix"] = -5e12
    params_fin["Qfix"] = -8.5e12

    print(f"Qfix Initial (Before UV): {params_ini['Qfix']:.2e} cm^-2")
    print(f"Qfix Final (After UV): {params_fin['Qfix']:.2e} cm^-2")

    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Calculate separate sigma arrays
    sigma_n_before = calculate_gaussian_sigma(E_array, SIGMA0_N_BEFORE, A_N_BEFORE, E0_N_BEFORE)
    sigma_p_before = calculate_gaussian_sigma(E_array, SIGMA0_P_BEFORE, A_P_BEFORE, E0_P_BEFORE)

    sigma_n_after = calculate_gaussian_sigma(E_array, SIGMA0_N_AFTER, A_N_AFTER, E0_N_AFTER)
    sigma_p_after = calculate_gaussian_sigma(E_array, SIGMA0_P_AFTER, A_P_AFTER, E0_P_AFTER)

    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    Dit_tot_ini = build_dit_profile(E_array, params_ini, Dig_func, Ev, Ec)
    Dit_tot_fin = build_dit_profile(E_array, params_fin, Dig_func, Ev, Ec)

    qfix_ini = params_ini["Qfix"] * elementary_charge
    qfix_fin = params_fin["Qfix"] * elementary_charge

    # Surface doping array
    # Increased resolution (from 100 to 100 points) to smooth out the curve
    Ndop_emitter_arr = np.logspace(15, 20, 100)
    dn = 1e15 # Standard injection for J0 evaluation

    j0_ini_fA = []
    j0_fin_fA = []
    delta_j0_fA = []

    n, p = n0 + dn, p0 + dn

    # J0 factor based on generic definition for the unpassivated side
    # J0 = q * ni^2 * W / (N_bulk + dn) * (1/tau_surf)
    # Convert A/cm2 to fA/cm2 using 1e15 factor
    j0_factor = elementary_charge * (ni_b**2) * W / (Ndop_bulk + dn) * 1e15

    for Ndop_em in Ndop_emitter_arr:
        # Use "Before UV" sigma params for initial
        tau_surf_ini = surfaceLifetime(
            n0, p0, n, p, dn, qfix_ini, T,
            Ndop_em, Ndop_bulk, dop_type_emitter, dop_type_bulk,
            dn, Dit_tot_ini, sigma_n_before, sigma_p_before)
            
        # Use "After UV" sigma params for final
        tau_surf_fin = surfaceLifetime(
            n0, p0, n, p, dn, qfix_fin, T,
            Ndop_em, Ndop_bulk, dop_type_emitter, dop_type_bulk,
            dn, Dit_tot_fin, sigma_n_after, sigma_p_after)

        j_ini = j0_factor / tau_surf_ini if tau_surf_ini > 0 else np.nan
        j_fin = j0_factor / tau_surf_fin if tau_surf_fin > 0 else np.nan
        
        j0_ini_fA.append(j_ini)
        j0_fin_fA.append(j_fin)
        delta_j0_fA.append(j_fin - j_ini)

    # --- Plot styling ---
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
    plt.rc('legend', fontsize=12)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['mathtext.fontset'] = "dejavuserif"

    fig, ax = plt.subplots(figsize=(9, 6))

    # Using absolute value for log-log plot to handle potential negative delta J0
    # though usually J0 increases with Dit.
    ax.loglog(Ndop_emitter_arr, np.abs(delta_j0_fA), color="purple", linewidth=lw, 
              label=r"$|\Delta J_0|$ (Final - Initial)")

    # Diagnostic: Plot J0_before and J0_after
    ax.loglog(Ndop_emitter_arr, j0_ini_fA, 'k--', alpha=0.5, label=r"$J_0$ Before UV")
    ax.loglog(Ndop_emitter_arr, j0_fin_fA, 'r--', alpha=0.5, label=r"$J_0$ After UV")

    ax.set_xlabel('Surface Doping (cm$^{-3}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$|\Delta J_0|$ & $J_0$ (fA/cm$^2$)', fontsize=16, fontweight='bold')
    ax.set_title(r"Impact of Surface Doping on $|\Delta J_0|$ (UV Exposure)", fontsize=16, fontweight='bold')
    
    ax.grid(True, which="major", linestyle='--', alpha=0.5)
    ax.legend(frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 14})
    
    ax.tick_params(axis='both', which='major', direction='in', length=8,
                   labelsize=14, width=1.2, pad=8)
    ax.tick_params(axis='both', which='minor', direction='in', length=8,
                   labelsize=14, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_weight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    fig.tight_layout()

    # Save to impact of doping folder
    figures_dir = os.path.join(os.path.dirname(script_dir), 'figures', 'impact_of_doping')
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, 'figure_delta_j0_vs_doping_1.png')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.2)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    simulate_impact_of_doping()
