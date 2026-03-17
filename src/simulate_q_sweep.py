#!/usr/bin/env python3
"""
Sweep the magnitude of negative fixed charge (Qfix) to explore
depletion-to-inversion transitions.

Output: figures/figure_q_sweep.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.constants import e as elementary_charge

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
    equilibrium_concentrations, effective_lifetime,
    load_experimental_data, build_dit_profile,
)


# ============================================================================
# MAIN
# ============================================================================

def simulate_q_sweep():
    print("Starting Q-Sweep Simulation...")

    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, "data")

    try:
        dn_exp, tau_exp = load_experimental_data(
            os.path.join(data_folder, "W3_ini.xlsx"))
    except Exception as e:
        print(f"Error loading experimental data: {e}")
        return

    base_params = {
        "Dit0_v": 1, "Ev_trap": 1.500e-02,
        "Dit0_c": 1, "Ec_trap": 1.000e-02,
        "Dit0_g": 6.000e+11, "E0_g": 0.6, "sigma_g": 5.000e-01,
    }

    q_magnitudes = [1e10, 5e10, 1e11, 5e11, 1e12, 2e12, 5e12, 1e13, 2e13, 5e13]

    E_array = create_energy_array(Ev, Ec, ENERGY_POINTS)
    sigma_n = calculate_gaussian_sigma(E_array, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E_array, SIGMA0_P, A_P, E0_P)
    dn_array = np.logspace(14, 17, 100)
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    Dit_tot = build_dit_profile(E_array, base_params, Dig_func, Ev, Ec)

    # --- Plot ---
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=16)
    plt.rc('legend', fontsize=10)
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.loglog(dn_exp, tau_exp * 1e3,
              marker='v', linestyle='', color='black', label="Before UV (Exp)",
              markersize=10, markerfacecolor='gray', markeredgecolor='black', zorder=100)

    colours = cm.viridis(np.linspace(0, 1, len(q_magnitudes)))

    for idx, Q_mag in enumerate(q_magnitudes):
        print(f"  Qfix = -{Q_mag:.1e} cm^-2")
        qfix_C = -Q_mag * elementary_charge
        tau_eff_ms = []
        for dn in dn_array:
            n, p = n0 + dn, p0 + dn
            tau_surf = surfaceLifetime(
                n0, p0, n, p, dn, qfix_C, T,
                0.0, Ndop_bulk, dop_type_emitter, dop_type_bulk,
                dn, Dit_tot, sigma_n, sigma_p)
            tau_intr = intrinsicLifetime(n0, p0, n, p, dn)
            tau_bulk = calculate_srh_lifetime_injection_dependent(
                dn, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)
            tau_eff_ms.append(effective_lifetime(tau_surf, tau_intr, tau_bulk) * 1e3)

        ax.loglog(dn_array, tau_eff_ms, label=f"Q = -{Q_mag:.0e}",
                  color=colours[idx], linewidth=2)

    ax.set_xlabel('Minority Carrier Density (cm$^{-3}$)')
    ax.set_ylabel('Effective Lifetime (ms)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title("Negative Charge Sweep: Depletion to Inversion?")
    ax.set_xlim([1e14, 1.1e17])
    ax.set_ylim([0.001, 100])
    ax.grid(True, which="major", linestyle='--', alpha=0.5)
    plt.tight_layout()

    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    out = os.path.join(figures_dir, 'figure_q_sweep.png')
    fig.savefig(out, bbox_inches='tight')
    print(f"  Saved {out}")


if __name__ == "__main__":
    simulate_q_sweep()
