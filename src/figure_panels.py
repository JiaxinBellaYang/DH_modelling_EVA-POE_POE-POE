"""
Figure generators for the three main parameter-sweep panels.

  generate_figure1() — Effect of varying Qfix and peak Dit
  generate_figure2() — Effect of Gaussian width and defect position
  generate_figure3() — Effect of capture cross-section scaling and correlation energy

All physics and constants are imported directly; no arguments needed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as pl
from scipy.constants import e as elementary_charge
from matplotlib.ticker import ScalarFormatter

import config

from physics import (
    surfaceLifetime,
    ni_func,
    Dig_func,
    create_energy_array,
    calculate_gaussian_sigma,
)

from helpers import equilibrium_concentrations


# ============================================================================
# SHARED SETUP
# ============================================================================

def _common_setup():
    """Return energy array, cross-section arrays, injection array, and ni_b."""
    E = create_energy_array(config.Ev, config.Ec, config.ENERGY_POINTS)
    sigma_n = calculate_gaussian_sigma(E, config.SIGMA0_N, config.A_N, config.E0_N)
    sigma_p = calculate_gaussian_sigma(E, config.SIGMA0_P, config.A_P, config.E0_P)
    dn_array = np.logspace(14, 17)
    ni_b = ni_func(config.T, config.Ndop_bulk, config.dop_type_bulk)
    return E, sigma_n, sigma_p, dn_array, ni_b


def _sweep_lifetime(dn_array, ni_b, Dit_g, Qfix_coulomb, sigma_n, sigma_p):
    """Run the injection sweep and return τ_surface array (ms)."""
    tau_ms = []
    n0, p0 = equilibrium_concentrations(config.Ndop_bulk, config.dop_type_bulk, ni_b)
    for dn in dn_array:
        n = n0 + dn
        p = p0 + dn
        tau_s = surfaceLifetime(
            n0, p0, n, p, dn, Qfix_coulomb, config.T,
            config.Ndop_emitter, config.Ndop_bulk, config.dop_type_emitter, config.dop_type_bulk,
            dn, Dit_g, sigma_n, sigma_p
        )
        tau_ms.append(tau_s * 1e3)
    return tau_ms


# ============================================================================
# FIGURE 1 — Varying Qfix  &  Varying peak Dit
# ============================================================================

def generate_figure1(output_dir='figures/'):
    """Figure 1: Lifetime components vs Injection Level for an undiffused wafer.
    Output: {output_dir}/figure 1.png
    """
    print("Starting calculations for Figure 1...")
    E, sigma_n, sigma_p, dn_array, ni_b = _common_setup()
    hsv = pl.cm.hsv(np.linspace(0, 1, 12))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # --- Panel (a): Varying Qfix ---
    print("  Panel (a): varying Qfix...")
    Ditmax_q = 1e10
    Dit_g_q = Dig_func(E, config.GAUSS_E0, Ditmax_q, config.GAUSS_SIGMA)
    for i, qfix_cm2 in enumerate(np.logspace(10, 11.5, 10)):
        Qfix_C = -qfix_cm2 * elementary_charge
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g_q, Qfix_C, sigma_n, sigma_p)
        ax[0].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1e15, 1.5), (1e15, 100),
                                    arrowstyle='->', mutation_scale=15, color='black')
    ax[0].add_patch(arrow)
    ax[0].text(1.5e15, 100, 'Higher charge', fontsize=10)
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')

    # --- Panel (b): Varying peak Dit ---
    print("  Panel (b): varying Ditmax...")
    Qfix_dit_C = -1e10 * elementary_charge
    for i, ditmax in enumerate(np.logspace(10, 12, 10)):
        Dit_g_d = Dig_func(E, config.GAUSS_E0, ditmax, config.GAUSS_SIGMA)
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g_d, Qfix_dit_C, sigma_n, sigma_p)
        ax[1].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1e15, 0.01), (1e15, 10),
                                    arrowstyle='<-', mutation_scale=15, color='black')
    ax[1].add_patch(arrow)
    ax[1].text(1.4e15, 0.02, 'Higher Dit', fontsize=10)
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')

    for a in ax:
        a.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        a.set_xlim([1e14, 1.1e16])
        a.set_xlabel('Carrier density $(cm^{-3})$')
        a.set_ylabel('Minority carrier lifetime $(ms)$')
        a.tick_params(axis='both', which='major', direction='in', length=8)
        a.tick_params(axis='both', which='minor', direction='in', length=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'figure 1.png')
    fig.savefig(fig_path)
    print(f"  Saved {fig_path}")


# ============================================================================
# FIGURE 2 — Varying Gaussian width  &  Defect position
# ============================================================================

def generate_figure2(output_dir='figures/'):
    """Generate Figure 2 and save to {output_dir}/figure 2.png."""
    print("\nStarting calculations for Figure 2...")
    E, sigma_n, sigma_p, dn_array, ni_b = _common_setup()
    hsv = pl.cm.hsv(np.linspace(0, 1, 12))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    Qtot = -5e10 * elementary_charge

    # --- Panel (a): Varying Gaussian width ---
    print("  Panel (a): varying Gaussian width...")
    for i, gw in enumerate(np.linspace(0.1, 0.5, 10)):
        Dit_g = Dig_func(E, 0.56, 1e10, gw)
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g, Qtot, sigma_n, sigma_p)
        ax[0].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1.5e14, 3), (1.5e14, 10),
                                    arrowstyle='<-', mutation_scale=15, color='black')
    ax[0].add_patch(arrow)
    ax[0].text(1.6e14, 2.5, 'Wider Gaussian distribution', fontsize=10)
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')

    # --- Panel (b): Varying defect position ---
    print("  Panel (b): varying defect position...")
    for i, pos in enumerate(np.linspace(0.1, 0.55, 10)):
        Dit_g = Dig_func(E, pos, 1e10, 0.18)
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g, Qtot, sigma_n, sigma_p)
        ax[1].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1.5e14, 5), (1.5e14, 10),
                                    arrowstyle='<-', mutation_scale=15, color='black')
    ax[1].add_patch(arrow)
    ax[1].text(1.6e14, 4, 'Deeper defect', fontsize=10)
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')

    for a in ax:
        a.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        a.set_xlim([1e14, 1.1e16])
        a.set_ylim([0.9, 12])
        a.set_xlabel('Carrier density $(cm^{-3})$')
        a.set_ylabel('Minority carrier lifetime $(ms)$')
        a.tick_params(axis='both', which='major', direction='in', length=8)
        a.tick_params(axis='both', which='minor', direction='in', length=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'figure 2.png')
    fig.savefig(fig_path)
    print(f"  Saved {fig_path}")


# ============================================================================
# FIGURE 3 — Capture cross-section scaling  &  Correlation energy
# ============================================================================

def generate_figure3(output_dir='figures/'):
    """Generate Figure 3 and save to {output_dir}/figure 3.png."""
    print("\nStarting calculations for Figure 3...")
    E, sigma_n, sigma_p, dn_array, ni_b = _common_setup()
    hsv = pl.cm.hsv(np.linspace(0, 1, 12))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    Qtot = -5e10 * elementary_charge

    # --- Panel (a): Varying capture cross-section scaling ---
    print("  Panel (a): varying capture cross-section...")
    Dit_g = Dig_func(E, 0.56, 1e10, 0.18)
    for i, scale in enumerate(np.logspace(0, 2, 10)):
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g, Qtot,
                                 sigma_n * scale, sigma_p * scale)
        ax[0].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1.5e14, 0.5), (1.5e14, 50),
                                    arrowstyle='<-', mutation_scale=15, color='black')
    ax[0].add_patch(arrow)
    ax[0].text(1.6e14, 0.55, 'Larger capture cross section', fontsize=10)
    ax[0].text(-0.1, 1.05, '(a)', transform=ax[0].transAxes, fontsize=12, va='bottom', ha='left')

    # --- Panel (b): Varying correlation energy ---
    print("  Panel (b): varying correlation energy...")
    for i, U in enumerate(np.linspace(0.05, 0.7, 10)):
        Dit_donor = Dig_func(E, 0.56 - U / 2, 1e10 / 2, 0.18)
        Dit_acceptor = Dig_func(E, 0.56 + U / 2, 1e10 / 2, 0.18)
        Dit_g_corr = Dit_donor + Dit_acceptor
        tau_ms = _sweep_lifetime(dn_array, ni_b, Dit_g_corr, Qtot, sigma_n, sigma_p)
        ax[1].loglog(dn_array, tau_ms, color=hsv[i], linewidth=config.lw)

    arrow = patches.FancyArrowPatch((1.5e14, 0.5), (1.5e14, 15),
                                    arrowstyle='->', mutation_scale=15, color='black')
    ax[1].add_patch(arrow)
    ax[1].text(1.6e14, 0.55, 'Larger correlation energy', fontsize=10)
    ax[1].text(-0.1, 1.05, '(b)', transform=ax[1].transAxes, fontsize=12, va='bottom', ha='left')

    for a in ax:
        a.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        a.set_xlim([1e14, 1.1e16])
        a.set_xlabel('Carrier density $(cm^{-3})$')
        a.set_ylabel('Minority carrier lifetime $(ms)$')
        a.tick_params(axis='both', which='major', direction='in', length=8)
        a.tick_params(axis='both', which='minor', direction='in', length=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'figure 3.png')
    fig.savefig(fig_path)
    print(f"  Saved {fig_path}")
