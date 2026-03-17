#!/usr/bin/env python3
"""
Simulate Diffused Surface J0 contribution vs Surface Doping.
Assuming a P-type diffused emitter.

This script calculates the specific surface contribution to J0 (J0s) 
by treating the surface as a P-type region with doping N_surface 
and calculating the surface recombination velocity S at that doping.

Model: J0s = q * ni_eff^2 * S / N_surface

Output: figures/impact_of_doping/figure_diffused_j0_vs_doping.png
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
    Ev, Ec, T, W,
    ENERGY_POINTS, lw,
)

from physics import (
    surfaceLifetime, Dig_func, ni_func,
    create_energy_array, calculate_gaussian_sigma,
)

from helpers import (
    equilibrium_concentrations, build_dit_profile,
)

def simulate_diffused_j0():
    print("Starting calculations for Diffused J0 vs Surface Doping...")

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
            "multiple_peaks": True,
            "Dit0_v": r["Dit0_v"], "Ev_trap": r["Ev_trap"],
            "Dit0_c": r["Dit0_c"], "Ec_trap": r["Ec_trap"],
            "Dit0_g1": r["Dit0_g1"], "E0_1": r["E0_1"], "sigma_1": r["sigma_1"],
            "Dit0_g2": r["Dit0_g2"], "E0_2": r["E0_2"], "sigma_2": r["sigma_2"],
            "Dit0_g3": r["Dit0_g3"], "E0_3": r["E0_3"], "sigma_3": r["sigma_3"],
        }
        
    params_ini = row_to_params(row_before)
    params_fin = row_to_params(row_after)

    # --- Fixed Qfix Values (Same as simulate_impact_doping.py) ---
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

    Dit_tot_ini = build_dit_profile(E_array, params_ini, Dig_func, Ev, Ec)
    Dit_tot_fin = build_dit_profile(E_array, params_fin, Dig_func, Ev, Ec)

    qfix_ini = params_ini["Qfix"] * elementary_charge
    qfix_fin = params_fin["Qfix"] * elementary_charge

    # Surface doping array (P-type emitter surface concentration)
    # Focusing on 1e18 to 1e20 for typical diffusions, but let's show wider range
    Ndop_surf_arr = np.logspace(16, 20.5, 100)
    dn = 1e14 # Injection level for evaluation (low injection relative to emitter)

    j0_ini_fA = []
    j0_fin_fA = []
    delta_j0_fA = []

    # Configuration for P-type Emitter surface
    # We treat the "bulk" argument of surfaceLifetime as the P-type emitter material
    dop_type_emitter_layer = 0.0 # P-type

    for N_surf in Ndop_surf_arr:
        # Calculate ni_eff for this doping (Band gap narrowing)
        ni_eff = ni_func(T, N_surf, dop_type_emitter_layer)
        
        # Equilibrium concentrations in the P-type layer
        n0, p0 = equilibrium_concentrations(N_surf, dop_type_emitter_layer, ni_eff)
        n, p = n0 + dn, p0 + dn

        # --- Initial State (Before UV) ---
        # Note: We pass N_surf as doping for both "bulk" and "emitter" slots or 
        # set Ndop_emitter=0 and pass N_surf as bulk to force the solver to use N_surf as reference.
        # Physics.py logic: Ndop_surface = |Ndop_emitter_n/p - Ndop_bulk_n/p|
        # To simulate a uniform P-type slab of doping N_surf:
        # Ndop_bulk_arg = N_surf, dop_type_bulk_arg = 0 (P).
        # Ndop_emitter = 0.
        
        tau_surf_ini = surfaceLifetime(
            n0, p0, n, p, dn, qfix_ini, T,
            0.0, N_surf, 0.0, dop_type_emitter_layer, # Ndop_emitter=0, Ndop_bulk=N_surf, type=P
            dn, Dit_tot_ini, sigma_n_before, sigma_p_before)
            
        # S = W / (2 * tau)
        # However, W here is arbitrary as long as we convert back to S.
        # surfaceLifetime returns tau = W / (2*S) -> S = W / (2*tau)
        S_ini = W / (2 * tau_surf_ini) if tau_surf_ini > 0 else 0
        
        # J0s = q * ni^2 * S / N_dop
        # Using (N_surf + dn) for generality, though dn is small
        j0s_val_ini = elementary_charge * (ni_eff**2) * S_ini / (N_surf + dn)
        
        # --- Final State (After UV) ---
        tau_surf_fin = surfaceLifetime(
            n0, p0, n, p, dn, qfix_fin, T,
            0.0, N_surf, 0.0, dop_type_emitter_layer,
            dn, Dit_tot_fin, sigma_n_after, sigma_p_after)

        S_fin = W / (2 * tau_surf_fin) if tau_surf_fin > 0 else 0
        j0s_val_fin = elementary_charge * (ni_eff**2) * S_fin / (N_surf + dn)
        
        # Convert to fA/cm2
        j0_ini_fA.append(j0s_val_ini * 1e15)
        j0_fin_fA.append(j0s_val_fin * 1e15)
        delta_j0_fA.append((j0s_val_fin - j0s_val_ini) * 1e15)

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

    ax.loglog(Ndop_surf_arr, np.abs(delta_j0_fA), color="purple", linewidth=lw, 
              label=r"$|\Delta J_{0,surf}|$")

    ax.loglog(Ndop_surf_arr, j0_ini_fA, 'k--', alpha=0.5, label=r"$J_{0,surf}$ Before UV")
    ax.loglog(Ndop_surf_arr, j0_fin_fA, 'r--', alpha=0.5, label=r"$J_{0,surf}$ After UV")

    ax.set_xlabel('Surface Doping $N_{surf}$ (cm$^{-3}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$J_{0,surf}$ (fA/cm$^2$)', fontsize=16, fontweight='bold')
    ax.set_title(r"Surface Contribution to $J_0$ (P-type Emitter)", fontsize=16, fontweight='bold')
    
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

    # Save
    figures_dir = os.path.join(os.path.dirname(script_dir), 'figures', 'impact_of_doping')
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, 'figure_diffused_j0_vs_doping.png')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.2)
    print(f"\n  Saved {out}")
    
    # Save CSV data for reference
    out_csv = os.path.join(figures_dir, 'diffused_j0_data.csv')
    df_out = pd.DataFrame({
        "N_surf_cm3": Ndop_surf_arr,
        "J0_initial_fA": j0_ini_fA,
        "J0_final_fA": j0_fin_fA,
        "Delta_J0_fA": delta_j0_fA
    })
    df_out.to_csv(out_csv, index=False)
    print(f"  Saved Data to {out_csv}")


if __name__ == "__main__":
    simulate_diffused_j0()
