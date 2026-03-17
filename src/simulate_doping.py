#!/usr/bin/env python3
"""
Simulate lifetime curves for undiffused vs diffused surfaces at two charge levels.

Output: figures/figure_surface_doping_comparison.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e as elementary_charge
from matplotlib.ticker import ScalarFormatter

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N, A_N, E0_N, SIGMA0_P, A_P, E0_P,
    TAU_N0_BULK, TAU_P0_BULK,
    Ev, Ec, T,
    dop_type_bulk, dop_type_emitter,
    Ndop_bulk,
    ENERGY_POINTS, lw,
)

from physics import (
    surfaceLifetime, intrinsicLifetime, Dig_func, ni_func,
    create_energy_array, calculate_gaussian_sigma,
    calculate_srh_lifetime_injection_dependent,
)

from helpers import (
    equilibrium_concentrations, effective_lifetime, build_dit_profile,
)


# ============================================================================
# MAIN
# ============================================================================

def simulate_and_plot_surface_doping():
    print("Starting calculations for Surface Doping Comparison Figure...")

    base_params = {
        "Dit0_v": 1, "Ev_trap": 2.624e-02,
        "Dit0_c": 1, "Ec_trap": 5.000e-02,
        "Dit0_g": 6.00e+12, "E0_g": 0.5, "sigma_g": 4.000e-01,
    }

    Ndop_undiffused = 0.0
    Ndop_diffused = 1e19

    scenarios = [
        {"plot_idx": 0, "label": "Low Charge (2e13 cm$^{-2}$)",
         "Ndop_emitter": Ndop_undiffused, "Qfix": 2e13, "color": "blue",  "linestyle": "-"},
        {"plot_idx": 0, "label": "High Charge (2e14 cm$^{-2}$)",
         "Ndop_emitter": Ndop_undiffused, "Qfix": 2e14, "color": "red",   "linestyle": "--"},
        {"plot_idx": 1, "label": "Low Charge (2e13 cm$^{-2}$)",
         "Ndop_emitter": Ndop_diffused,   "Qfix": 2e13, "color": "blue",  "linestyle": "-"},
        {"plot_idx": 1, "label": "High Charge (2e14 cm$^{-2}$)",
         "Ndop_emitter": Ndop_diffused,   "Qfix": 2e14, "color": "red",   "linestyle": "--"},
    ]

    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    sigma_n = calculate_gaussian_sigma(E_array, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E_array, SIGMA0_P, A_P, E0_P)
    dn_array = np.logspace(14, 17, 100)
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    Dit_tot = build_dit_profile(E_array, base_params, Dig_func, Ev, Ec)

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    axes[0].set_title(f"Undiffused  (Ndop,surf = {Ndop_undiffused:.0e} cm" + r"$^{\mathregular{-3}}$)",
                      fontsize=16, fontweight='bold')
    axes[1].set_title(f"Diffused  (Ndop,surf = {Ndop_diffused:.0e} cm" + r"$^{\mathregular{-3}}$)",
                      fontsize=16, fontweight='bold')

    for sc in scenarios:
        params = {**base_params, **sc}
        ax = axes[params["plot_idx"]]
        print(f"  {params['label']}  Ndop_emitter={params['Ndop_emitter']:.1e}")

        qfix_C = -params["Qfix"] * elementary_charge
        tau_eff_ms = []
        for dn in dn_array:
            n, p = n0 + dn, p0 + dn
            tau_surf = surfaceLifetime(
                n0, p0, n, p, dn, qfix_C, T,
                params["Ndop_emitter"], Ndop_bulk, dop_type_emitter, dop_type_bulk,
                dn, Dit_tot, sigma_n, sigma_p)
            tau_intr = intrinsicLifetime(n0, p0, n, p, dn)
            tau_bulk = calculate_srh_lifetime_injection_dependent(
                dn, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)
            tau_eff_ms.append(effective_lifetime(tau_surf, tau_intr, tau_bulk) * 1e3)

        ax.loglog(dn_array, tau_eff_ms, label=params["label"],
                  color=params["color"], linestyle=params["linestyle"],
                  linewidth=lw, zorder=10)

    # --- Axes formatting ---
    for ax in axes:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.set_xlim([1e14, 1.1e17])
        ax.set_ylim([0.01, 100])
        ax.set_xticks([1e14, 1e15, 1e16, 1e17])
        ax.set_yticks([1e-1, 1e0, 1e1, 1e2])
        ax.set_xlabel('Minority Carrier Density (cm$^{-3}$)',
                      fontsize=16, fontweight='bold')
        ax.legend(frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 14})
        ax.grid(True, which="major", linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', direction='in', length=8,
                       labelsize=14, width=1.2, pad=8)
        ax.tick_params(axis='both', which='minor', direction='in', length=8,
                       labelsize=14, width=1.2)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_weight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    axes[0].set_ylabel('Effective Lifetime (ms)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.85, wspace=0.15)

    figures_dir = os.path.join(os.path.dirname(script_dir), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, 'figure_surface_doping_comparison.png')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.2)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    simulate_and_plot_surface_doping()
