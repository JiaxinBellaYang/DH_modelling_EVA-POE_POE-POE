#!/usr/bin/env python3
"""
Simulate and plot lifetime curves comparing before and after UV exposure.

Combines surface, intrinsic, and bulk SRH recombination.
Overlays experimental data when available.

Output: figures/figure_uv_comparison_2.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as elementary_charge
from matplotlib.ticker import ScalarFormatter

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N_BEFORE, A_N_BEFORE, E0_N_BEFORE, SIGMA0_P_BEFORE, A_P_BEFORE, E0_P_BEFORE,
    SIGMA0_N_AFTER, A_N_AFTER, E0_N_AFTER, SIGMA0_P_AFTER, A_P_AFTER, E0_P_AFTER,
    TAU_N0_BULK, TAU_P0_BULK,
    Ev, Ec, T,
    dop_type_bulk, dop_type_emitter,
    Ndop_emitter, Ndop_bulk, W,
    ENERGY_POINTS, lw,
)

from physics import (
    surfaceLifetime, intrinsicLifetime, Dig_func, ni_func,
    create_energy_array, calculate_gaussian_sigma,
    calculate_srh_lifetime_injection_dependent,
)

from helpers import (
    equilibrium_concentrations, effective_lifetime,
    load_experimental_data, build_dit_profile, setup_plot_style,
)


from scipy.optimize import minimize_scalar

# ============================================================================
# MAIN
# ============================================================================


def simulate_and_plot_uv_comparison(model_name):
    print(f"\nStarting calculations for UV Comparison Figure ({model_name})...")

    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, "data")
    
    figures_dir = os.path.join(project_root, "figures", "lifetimes", model_name)
    os.makedirs(figures_dir, exist_ok=True)
    
    results_dir = os.path.join(project_root, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)


    # --- Parameter sets (defaults) ---
    params_before = {
        "label": "Before UV (Simulated)",
        "type": "before",
        "Dit0_v": 1, "Ev_trap": 1.500e-02,
        "Dit0_c": 1, "Ec_trap": 1.000e-02,
        "Dit0_g": 6.000e+11, "E0_g": 0.6, "sigma_g": 5.000e-01,
        "Qfix": -5e12,
        "color": "blue",
    }
    params_after = {
        "label": "After UV (Simulated)",
        "type": "after",
        "Dit0_v": 1, "Ev_trap": 2.624e-02,
        "Dit0_c": 1, "Ec_trap": 5.000e-02,
        "Dit0_g": 6.00e+12, "E0_g": 0.5, "sigma_g": 4.000e-01,
        "Qfix": -8.5e12,
        "color": "red",
    }

    # --- Override with actual fit results if available ---
    summary_path = os.path.join(project_root, "results", model_name, "dit_fit_summary.xlsx")
    if os.path.exists(summary_path):
        print(f"  Found {summary_path}. Overriding simulated defaults with fit data...")
        import pandas as pd
        df = pd.read_excel(summary_path)
        
        # Helper to map dataframe row to parameter dictionary
        def override_params(p_dict, row):
            
            p_dict["Dit0_v"] = float(row["Dit0_v"])
            p_dict["Ev_trap"] = float(row["Ev_trap"])
            p_dict["Dit0_c"] = float(row["Dit0_c"])
            p_dict["Ec_trap"] = float(row["Ec_trap"])
            
            p_dict["label"] = p_dict["label"].replace("(Simulated)", "(Fit)")
            
            if model_name == "0peak":
                p_dict["zero_peaks"] = True
                p_dict["Dit0_g"] = 0.0
            elif model_name == "flatpeak":
                p_dict["flat_peak"] = True
                p_dict["Dit0_const"] = float(row["Dit0_const"])
                p_dict["Dit0_g"] = float(row["Dit0_const"]) # for logging
            elif model_name == "1peak":
                p_dict["Dit0_g"] = float(row["Dit0_g"])
                p_dict["E0_g"] = float(row["E0"])
                p_dict["sigma_g"] = float(row["sigma"])
            elif model_name == "2peak":
                p_dict["multiple_peaks"] = True
                p_dict["Dit0_g1"] = float(row["Dit0_g1"])
                p_dict["E0_1"] = float(row["E0_1"])
                p_dict["sigma_1"] = float(row["sigma_1"])
                p_dict["Dit0_g2"] = float(row["Dit0_g2"])
                p_dict["E0_2"] = float(row["E0_2"])
                p_dict["sigma_2"] = float(row["sigma_2"])
                p_dict["Dit0_g"] = float(row["Dit0_g1"]) + float(row["Dit0_g2"])
            else: # Assuming 3peak or more general multiple peaks
                p_dict["multiple_peaks"] = True
                p_dict["Dit0_g1"] = float(row["Dit0_g1"])
                p_dict["E0_1"] = float(row["E0_1"])
                p_dict["sigma_1"] = float(row["sigma_1"])
                p_dict["Dit0_g2"] = float(row["Dit0_g2"])
                p_dict["E0_2"] = float(row["E0_2"])
                p_dict["sigma_2"] = float(row["sigma_2"])
                p_dict["Dit0_g3"] = float(row["Dit0_g3"])
                p_dict["E0_3"] = float(row["E0_3"])
                p_dict["sigma_3"] = float(row["sigma_3"])
                p_dict["Dit0_g"] = float(row["Dit0_g1"]) + float(row["Dit0_g2"]) + float(row["Dit0_g3"])

            p_dict["label"] = p_dict["label"].replace("(Simulated)", "(Fit)")
        
        # Try to find "Initial" dataset for before
        row_ini = df[df["Dataset"] == "Initial"]
        if not row_ini.empty:
            override_params(params_before, row_ini.iloc[0])
            
        # Try to find "UV20h" dataset for after
        row_uv = df[df["Dataset"] == "UV20h"]
        if not row_uv.empty:
            override_params(params_after, row_uv.iloc[0])
    
    parameter_sets = [params_before, params_after]

    # --- Experimental data ---
    try:
        dn_before, tau_before = load_experimental_data(
            os.path.join(data_folder, "W3_ini.xlsx"))
        dn_after, tau_after = load_experimental_data(
            os.path.join(data_folder, "W3_UV20h.xlsx"))
        has_exp = True
    except FileNotFoundError:
        print("  Experimental data files not found — skipping.")
        has_exp = False

    # --- Compute common arrays ---
    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    
    # Calculate separate sigma arrays for before/after
    sigma_n_before = calculate_gaussian_sigma(E_array, SIGMA0_N_BEFORE, A_N_BEFORE, E0_N_BEFORE)
    sigma_p_before = calculate_gaussian_sigma(E_array, SIGMA0_P_BEFORE, A_P_BEFORE, E0_P_BEFORE)

    sigma_n_after = calculate_gaussian_sigma(E_array, SIGMA0_N_AFTER, A_N_AFTER, E0_N_AFTER)
    sigma_p_after = calculate_gaussian_sigma(E_array, SIGMA0_P_AFTER, A_P_AFTER, E0_P_AFTER)

    dn_array = np.logspace(14, 17, 100)
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    # --- Optimize Qfix against experimental data ---
    if has_exp:
        print("  Using fixed Qfix values (skipping optimization)...")
        print(f"  Qfix Before: {params_before['Qfix']:.2e}")
        print(f"  Qfix After:  {params_after['Qfix']:.2e}")
        # Optimization block removed to use user-defined Qfix
        # ...
        
        # Fit both parameter sets
        # params_before["Qfix"], params_before["SSR"], params_before["R2"] = fit_qfix(params_before, dn_before, tau_before)
        # print(f"    --> {params_before['label']}: Best Qfix = {params_before['Qfix']:.2e} cm^-2 (R2={params_before['R2']:.3f})")
        
        # params_after["Qfix"], params_after["SSR"], params_after["R2"] = fit_qfix(params_after, dn_after, tau_after)
        # print(f"    --> {params_after['label']}: Best Qfix = {params_after['Qfix']:.2e} cm^-2 (R2={params_after['R2']:.3f})")

    setup_plot_style()
    
    # --- Initialize TWO figures ---
    fig_tau, ax_tau = plt.subplots(figsize=(10, 7))
    fig_j0, ax_j0 = plt.subplots(figsize=(10, 7))

    for params in parameter_sets:
        print(f"  Calculating: {params['label']}")
        Dit_tot = build_dit_profile(E_array, params, Dig_func, Ev, Ec)
        qfix_C = params["Qfix"] * elementary_charge

        # Select the correct sigma arrays
        if params.get('type') == 'after':
            sigma_n = sigma_n_after
            sigma_p = sigma_p_after
        else:
            sigma_n = sigma_n_before
            sigma_p = sigma_p_before

        tau_eff_ms = []
        inv_tau_diff = []
        warned_phi = False
        
        for dn in dn_array:
            n, p = n0 + dn, p0 + dn
            tau_surf, diag = surfaceLifetime(
                n0, p0, n, p, dn, qfix_C, T,
                Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk,
                dn, Dit_tot, sigma_n, sigma_p, return_diagnostics=True)
            tau_intr = intrinsicLifetime(n0, p0, n, p, dn)
            tau_bulk = calculate_srh_lifetime_injection_dependent(
                dn, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)

            ns_surf = diag["ns"]
            if not warned_phi:
                if qfix_C < 0 and ns_surf > n:
                    print(f"      [WARNING] Electrons accumulated (ns > bulk) despite negative Qfix at dn = {dn:.1e}!")
                    warned_phi = True
                elif qfix_C > 0 and ns_surf < n:
                    print(f"      [WARNING] Electrons depleted (ns < bulk) despite positive Qfix at dn = {dn:.1e}!")
                    warned_phi = True
            
            # Lifetime list for first plot
            tau_eff_ms.append(effective_lifetime(tau_surf, tau_intr, tau_bulk) * 1e3)
            
            # Kane-Swanson for second plot
            val = (1.0 / tau_surf) + (1.0 / tau_bulk) if tau_surf > 0 and tau_bulk > 0 else np.nan
            inv_tau_diff.append(val)

        # Plot onto Figure 1 (Lifetime)
        ax_tau.loglog(dn_array, tau_eff_ms, label=params["label"],
                      color=params["color"], linewidth=lw, zorder=10)
                      
        # Plot onto Figure 2 (J0 Kane-Swanson)
        ax_j0.plot(dn_array, inv_tau_diff, label=params["label"],
                   color=params["color"], linewidth=lw, zorder=10)

    # --- Experimental overlay ---
    if has_exp:
        # Before UV Experimental (Figure 1 - Lifetime)
        ax_tau.loglog(dn_before, tau_before * 1e3,
                      marker='v', linestyle='', color='blue', label="Before UV (Exp)",
                      markersize=12, markerfacecolor='blue', markeredgecolor='navy',
                      markeredgewidth=2, zorder=5)
        # After UV Experimental (Figure 1 - Lifetime)
        ax_tau.loglog(dn_after, tau_after * 1e3,
                      marker='o', linestyle='', color='red', label="After UV (Exp)",
                      markersize=12, markerfacecolor='red', markeredgecolor='darkred',
                      markeredgewidth=2, zorder=5)

        # Before UV Experimental (Figure 2 - Kane-Swanson)
        inv_tau_exp_before = []
        for n_dn, tau in zip(dn_before, tau_before):
            tau_intr = intrinsicLifetime(n0, p0, n0 + n_dn, p0 + n_dn, n_dn)
            inv_tau_exp_before.append(1.0/tau - 1.0/tau_intr if tau_intr > 0 else np.nan)
        ax_j0.plot(dn_before, inv_tau_exp_before,
                   marker='v', linestyle='', color='blue', label="Before UV (Exp)",
                   markersize=12, markerfacecolor='blue', markeredgecolor='navy',
                   markeredgewidth=2, zorder=5)

        # After UV Experimental (Figure 2 - Kane-Swanson)
        inv_tau_exp_after = []
        for n_dn, tau in zip(dn_after, tau_after):
            tau_intr = intrinsicLifetime(n0, p0, n0 + n_dn, p0 + n_dn, n_dn)
            inv_tau_exp_after.append(1.0/tau - 1.0/tau_intr if tau_intr > 0 else np.nan)
        ax_j0.plot(dn_after, inv_tau_exp_after,
                   marker='o', linestyle='', color='red', label="After UV (Exp)",
                   markersize=12, markerfacecolor='red', markeredgecolor='darkred',
                   markeredgewidth=2, zorder=5)

    # --- Formatting Figure 1 (Lifetime) ---
    ax_tau.set_xlim([1e14, 1.1e17])
    ax_tau.set_ylim([0.01, 100])
    ax_tau.set_xticks([1e14, 1e15, 1e16, 1e17])
    ax_tau.set_yticks([1e-1, 1e0, 1e1, 1e2])
    
    font_labels = {'family': 'Arial', 'weight': 'bold', 'size': 20}
    font_legend = {'family': 'Arial', 'weight': 'bold', 'size': 16}
    
    ax_tau.set_xlabel('Minority Carrier Density (cm$^{-3}$)', fontdict=font_labels)
    ax_tau.set_ylabel('Effective Lifetime (ms)', fontdict=font_labels)
    ax_tau.legend(frameon=False, prop=font_legend)
    ax_tau.tick_params(axis='both', which='major', direction='in', length=8,
                       labelsize=16, width=1.2, pad=8)
    ax_tau.tick_params(axis='both', which='minor', direction='in', length=8,
                       labelsize=16, width=1.2)
    for lab in ax_tau.get_xticklabels() + ax_tau.get_yticklabels():
        lab.set_weight('bold')
    for spine in ax_tau.spines.values():
        spine.set_linewidth(3)
        
    fig_tau.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)

    # --- Formatting Figure 2 (Kane-Swanson J0) ---
    ax_j0.set_xlim([0, 1e16]) # Linear scale tailored for standard injections
    ax_j0.set_ylim(bottom=0)
    
    ax_j0.set_xlabel('Excess carrier concentration (cm$^{-3}$)', fontdict=font_labels)
    ax_j0.set_ylabel(r'$1/\tau_{eff} - 1/\tau_{intr}$ (s$^{-1}$)', fontdict=font_labels)
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax_j0.xaxis.set_major_formatter(formatter)
    
    ax_j0.legend(frameon=True, prop=font_legend, facecolor='white', framealpha=1)
    ax_j0.tick_params(axis='both', which='major', direction='in', length=8,
                      labelsize=16, width=1.2, pad=8)
    for lab in ax_j0.get_xticklabels() + ax_j0.get_yticklabels():
        lab.set_weight('bold')
    for spine in ax_j0.spines.values():
        spine.set_linewidth(3)
    ax_j0.grid(True, linestyle='--', alpha=0.5)
    
    fig_j0.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)


    
    out_tau = os.path.join(figures_dir, 'figure_uv_lifetime.png')
    out_j0 = os.path.join(figures_dir, 'figure_uv_j0_kane_swanson.png')
    
    fig_tau.savefig(out_tau, bbox_inches='tight', pad_inches=0.2)
    fig_j0.savefig(out_j0, bbox_inches='tight', pad_inches=0.2)
    
    print(f"\n  Saved {out_tau}")
    print(f"  Saved {out_j0}")

    # --- Save parameters table ---
    out_table = os.path.join(results_dir, f'uv_fit_summary_{model_name}.xlsx')
    
    # Create a cleaner dictionary for the dataframe
    export_data = []
    for p in parameter_sets:
        def _get_val(key, default=np.nan):
            return p.get(key, default)
            
        export_data.append({
            "Dataset": p["label"],
            "Qfix (cm^-2)": p["Qfix"],
            "SSR": _get_val("SSR"),
            "R2": _get_val("R2"),
            "Dit0_v": _get_val("Dit0_v"),
            "Ev_trap": _get_val("Ev_trap_sigma") if "Ev_trap_sigma" in p else _get_val("Ev_trap"),
            "Dit0_c": _get_val("Dit0_c"),
            "Ec_trap": _get_val("Ec_trap_sigma") if "Ec_trap_sigma" in p else _get_val("Ec_trap"),
            # Single baseline values
            "Dit0_const": _get_val("Dit0_const"),
            "Dit0_g": _get_val("Dit0_g"),
            "E0": _get_val("E0_g") if "E0_g" in p else _get_val("E0"),
            "sigma": _get_val("sigma_g") if "sigma_g" in p else _get_val("sigma"),
            # Multi-peak extra values
            "Dit0_g1": _get_val("Dit0_g1"),
            "E0_1": _get_val("E0_1"),
            "sigma_1": _get_val("sigma_1"),
            "Dit0_g2": _get_val("Dit0_g2"),
            "E0_2": _get_val("E0_2"),
            "sigma_2": _get_val("sigma_2"),
            "Dit0_g3": _get_val("Dit0_g3"),
            "E0_3": _get_val("E0_3"),
            "sigma_3": _get_val("sigma_3"),
        })
        
    df_export = pd.DataFrame(export_data)
    df_export.to_excel(out_table, index=False)
    print(f"  Saved parameters table to {out_table}")



if __name__ == "__main__":
    for m in ["0peak", "flatpeak", "1peak", "2peak", "3peak"]:
        simulate_and_plot_uv_comparison(m)
