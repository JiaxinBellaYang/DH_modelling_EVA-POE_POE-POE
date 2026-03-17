"""
Shared utility functions used across simulation scripts.

Eliminates boilerplate for equilibrium concentrations, effective lifetime
calculation, data loading, Dit profile construction, and plot styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CARRIER HELPERS
# ============================================================================

def equilibrium_concentrations(Ndop, dop_type, ni):
    """Return (n0, p0) for a doped semiconductor in thermal equilibrium.

    Parameters
    ----------
    Ndop : float
        Doping concentration (cm⁻³).  Zero means intrinsic.
    dop_type : float
        1.0 for n-type, 0.0 for p-type.
    ni : float
        Effective intrinsic carrier concentration (cm⁻³).

    Returns
    -------
    n0, p0 : float
        Equilibrium electron and hole concentrations (cm⁻³).
    """
    if Ndop > 0:
        if dop_type == 1:
            n0 = Ndop
            p0 = ni ** 2 / Ndop
        else:
            n0 = ni ** 2 / Ndop
            p0 = Ndop
    else:
        n0 = ni
        p0 = ni
    return n0, p0


def effective_lifetime(tau_surf, tau_intr, tau_bulk):
    """Matthiessen's rule: 1/τ_eff = 1/τ_surf + 1/τ_intr + 1/τ_bulk.

    Safely handles infinite or zero lifetimes.

    Returns
    -------
    tau_eff : float
        Effective lifetime (s).
    """
    inv = 0.0
    for tau in (tau_surf, tau_intr, tau_bulk):
        if tau > 0 and np.isfinite(tau):
            inv += 1.0 / tau
    return 1.0 / inv if inv > 1e-20 else np.inf


# ============================================================================
# DATA LOADING
# ============================================================================

def load_experimental_data(filepath, sheet="RawData"):
    """Load lifetime vs carrier-density data from an Excel file.

    Applies a mask to keep only positive, finite values.

    Returns
    -------
    dn : np.ndarray
        Minority carrier density (cm⁻³).
    tau : np.ndarray
        Measured lifetime (s).
    """
    df = pd.read_excel(filepath, sheet_name=sheet)
    dn = df["Minority Carrier Density"].values
    tau = df["Tau (sec)"].values

    mask = (dn > 0) & (tau > 0) & np.isfinite(dn) & np.isfinite(tau)
    return dn[mask], tau[mask]


# ============================================================================
# Dit PROFILE CONSTRUCTION
# ============================================================================

def build_dit_profile(E, params, Dig_func, Ev, Ec):
    """Build Dit_tot = Dit_v + Dit_g + Dit_c from a parameter dictionary.

    Expected keys in *params*:
        Dit0_v, Ev_trap_sigma   — valence-band tail (Gaussian approx)
        Dit0_c, Ec_trap_sigma   — conduction-band tail (Gaussian approx)
        Dit0_g, E0_g, sigma_g   — midgap Gaussian peak

    Parameters
    ----------
    E : np.ndarray
        Energy grid (eV).
    params : dict
        Parameter dictionary (see above).
    Dig_func : callable
        Gaussian Dit function from physics module.
    Ev, Ec : float
        Valence / conduction band edges (eV).

    Returns
    -------
    Dit_tot : np.ndarray
        Total Dit distribution (cm⁻² eV⁻¹).
    """
    Dit_v = params["Dit0_v"] * np.exp(-(E - Ev) / params["Ev_trap"])
    Dit_c = params["Dit0_c"] * np.exp(-(Ec - E) / params["Ec_trap"])
    
    if params.get("multiple_peaks"):
        Dit_g = np.zeros_like(E)
        if "Dit0_g1" in params:
            Dit_g += Dig_func(E, params["E0_1"], params["Dit0_g1"], params["sigma_1"])
        if "Dit0_g2" in params:
            Dit_g += Dig_func(E, params["E0_2"], params["Dit0_g2"], params["sigma_2"])
        if "Dit0_g3" in params:
            Dit_g += Dig_func(E, params["E0_3"], params["Dit0_g3"], params["sigma_3"])
    elif params.get("flat_peak"):
        Dit_g = np.full_like(E, params["Dit0_const"])
    elif params.get("zero_peaks"):
        Dit_g = 0.0
    else:
        Dit_g = Dig_func(E, params["E0_g"], params["Dit0_g"], params["sigma_g"])
        
    return Dit_v + Dit_c + Dit_g


# ============================================================================
# PLOT STYLING
# ============================================================================

def setup_plot_style():
    """Apply the project's standard matplotlib rcParams."""
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['figure.dpi'] = 80
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    plt.rcParams['font.size'] = 16
