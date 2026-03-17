"""
Model fitting and evaluation.

Provides two taueff variants:
  - taueff()          — simple: returns (tau_fit_array, RMSE)
  - taueff_detailed() — returns (squares, abs_diff, diff) + optional plot

Plus evaluate_model() for quick diagnostics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as elementary_charge

# --- path setup so "python src/fitting.py" works ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    T, Ec, Ev, kT, W,
    Ndop_bulk, Ndop_emitter, dop_type_bulk, dop_type_emitter,
    SIGMA0_N, A_N, E0_N, SIGMA0_P, A_P, E0_P,
    GAUSS_E0, GAUSS_SIGMA,
    TAU_N0_BULK, TAU_P0_BULK,
    ENERGY_POINTS,
)

from physics import (
    surfaceLifetime,
    intrinsicLifetime,
    ni_func,
    create_energy_array,
    calculate_gaussian_sigma,
    calculate_srh_lifetime_injection_dependent,
    Dig_func,
)

from helpers import equilibrium_concentrations, effective_lifetime


# ============================================================================
# SIMPLE taueff  (from chargeanddit.py)
# ============================================================================

def taueff(dn, tau_exp, Dit0, Qfix_C, sigma_n, sigma_p):
    """Compute effective lifetime using the full ns-solving surfaceLifetime.

    Combines surface recombination (Eqs. A4–A7, with Q_it from Eq. 2),
    intrinsic recombination (Richter model), and bulk SRH recombination
    via Matthiessen's rule:
        1/τ_eff = 1/τ_surf + 1/τ_intr + 1/τ_bulk

    Returns
    -------
    tau_fit : np.ndarray
        Modelled effective lifetime at each injection level.
    rmse : float
        Root-mean-square error of log(τ_fit) vs log(τ_exp).
    """
    ni_b = ni_func(T, Ndop_bulk, dop_type_bulk)
    E = create_energy_array()
    Dit_E = Dig_func(E, GAUSS_E0, Dit0, GAUSS_SIGMA)

    tau_fit = []
    err = []

    for i, dn_i in enumerate(dn):
        n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)
        n = n0 + dn_i
        p = p0 + dn_i

        tau_surf = surfaceLifetime(
            n0, p0, n, p, dn_i,
            Qfix_C, T,
            Ndop_emitter, Ndop_bulk,
            dop_type_emitter, dop_type_bulk,
            dn_i, Dit_E, sigma_n, sigma_p
        )
        tau_intr = intrinsicLifetime(n0, p0, n, p, dn_i)
        tau_bulk = calculate_srh_lifetime_injection_dependent(
            dn_i, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b
        )

        tau_eff = effective_lifetime(tau_surf, tau_intr, tau_bulk)
        tau_fit.append(tau_eff)
        err.append((np.log(tau_eff) - np.log(tau_exp[i])) ** 2)

    return np.array(tau_fit), np.sqrt(np.mean(err))


# ============================================================================
# DETAILED taueff  (from bulk_SRH.py — returns error-metric arrays)
# ============================================================================

def taueff_detailed(dn_array, tau_array, Dit_0instance, Dit_bandedgeinstance, Qfixi,
                    sigma_n_arr, sigma_p_arr, export=False):
    """Compute effective lifetime with full error-metric arrays.

    Returns
    -------
    squares, absolutedifference, difference : list[float]
    """
    ni_b_local = ni_func(T, Ndop_bulk, dop_type_bulk)

    E_fit = np.linspace(Ev, Ec, 100000)
    params_g = (GAUSS_E0, Dit_0instance, GAUSS_SIGMA)
    Dit_g = Dig_func(E_fit, *params_g)
    Dit_tot_fit = Dit_g

    tau_eff_array = []
    tau_surface_array = []
    tau_intr_array = []
    tau_SRH_array = []
    squares = []
    absolutedifference = []
    difference = []

    for count, dn_exp_val in enumerate(dn_array):
        Delta_n = dn_exp_val
        n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b_local)
        n = Delta_n + n0
        p = Delta_n + p0

        tau_surface = surfaceLifetime(
            n0, p0, n, p, Delta_n, Qfixi, T,
            Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk,
            dn_exp_val, Dit_tot_fit, sigma_n_arr, sigma_p_arr
        )
        tau_intr = intrinsicLifetime(n0, p0, n, p, Delta_n)
        tau_srh_bulk_inj = calculate_srh_lifetime_injection_dependent(
            Delta_n, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b_local
        )

        tau_eff = effective_lifetime(tau_surface, tau_intr, tau_srh_bulk_inj)

        tau_SRH_array.append(tau_srh_bulk_inj)
        tau_intr_array.append(tau_intr)
        tau_surface_array.append(tau_surface)
        tau_eff_array.append(tau_eff)

        squares.append((np.log(tau_eff) - np.log(tau_array[count])) ** 2)
        absolutedifference.append(np.abs(np.log(tau_eff) - np.log(tau_array[count])))
        difference.append((np.log(tau_eff) - np.log(tau_array[count])))

    if export:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.loglog(dn_array, np.multiply(tau_array, 1E3), 'o',
                  markersize=10, markerfacecolor='blue', markeredgewidth=1.5,
                  markeredgecolor='dimgray', label='Experimental Data')
        ax.loglog(dn_array, np.multiply(tau_eff_array, 1E3), 'blue', linewidth=3.5)
        ax.set_xlim([1e14, 1.1e16])
        ax.set_ylim(np.min(tau_array) * 0.7 * 1E3, np.max(tau_array) * 1.3 * 1E3)
        ax.set_xlabel('Carrier density $(cm^{-3})$')
        ax.set_ylabel('Minority carrier lifetime $(ms)$')
        ax.tick_params(axis='both', which='major', direction='in', length=8)
        ax.tick_params(axis='both', which='minor', direction='in', length=8)
        ax.legend(frameon=False, ncol=1)

    return squares, absolutedifference, difference


# ============================================================================
# ERROR-METRIC WRAPPERS  (used by optimisers in __main__)
# ============================================================================

def calculate_mean_square_error(params, dn_exp, tau_exp, sigma_n, sigma_p):
    Qfixinit, Dit_0g, Dit_bandedge = params
    Qfix = -Qfixinit * elementary_charge
    results = taueff_detailed(dn_exp, tau_exp, Dit_0g, Dit_bandedge, Qfix,
                              sigma_n, sigma_p, export=False)
    squares = results[0]
    rmse = (sum(squares) / len(squares)) ** 0.5
    print(f'RMSE = {rmse}')
    return rmse


def calculate_mean_absolute_error(params, dn_exp, tau_exp, sigma_n, sigma_p):
    Qfixi, Dit_0g, Dit_bandedge = params
    Qfix = -Qfixi * elementary_charge
    results = taueff_detailed(dn_exp, tau_exp, Dit_0g, Dit_bandedge, Qfix,
                              sigma_n, sigma_p, export=False)
    mae = sum(results[1]) / len(results[1])
    print(f'MAE = {mae}')
    return mae


def calculate_mean_bias_error(params, dn_exp, tau_exp, sigma_n, sigma_p):
    Qfixi, Dit_0g, Dit_bandedge = params
    Qfix = -Qfixi * elementary_charge
    results = taueff_detailed(dn_exp, tau_exp, Dit_0g, Dit_bandedge, Qfix,
                              sigma_n, sigma_p, export=False)
    mbe = sum(results[2]) / len(results[2])
    print(f'MBE = {mbe}')
    return mbe


# ============================================================================
# QUICK DIAGNOSTICS
# ============================================================================

def evaluate_model(dn_exp, tau_exp, sigma_n, sigma_p, Dit0, Qfix_cm2):
    """Evaluate model against experimental data and print diagnostics."""
    Qfix_C = -Qfix_cm2 * elementary_charge

    print("\n================ MODEL EVALUATION =================")
    print(f"Dit_0g              = {Dit0:.3e} cm^-2 eV^-1")
    print(f"Qfix (number)       = {Qfix_cm2:.3e} cm^-2")
    print(f"Effective Qfix      = {Qfix_C:.3e} C/cm^2")
    print("--------------------------------------------------")

    tau_fit, rmse = taueff(dn_exp, tau_exp, Dit0, Qfix_C, sigma_n, sigma_p)

    print(f"RMSE (log tau)      = {rmse:.4e}")
    print("First 5 points:")
    for i in range(min(5, len(dn_exp))):
        print(f" dn={dn_exp[i]:.2e}, tau_exp={tau_exp[i]:.2e}, tau_fit={tau_fit[i]:.2e}")
    print("==================================================\n")


# ============================================================================
# MAIN — run standalone evaluation
# ============================================================================

if __name__ == "__main__":
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "W3_ini.xlsx")

    exp_data = pd.read_excel(data_path, sheet_name="RawData")
    dn_exp = exp_data["Minority Carrier Density"].values
    tau_exp = exp_data["Tau (sec)"].values

    mask = (dn_exp > 0) & (tau_exp > 0)
    dn_exp = dn_exp[mask]
    tau_exp = tau_exp[mask]

    E = create_energy_array()
    sigma_n = calculate_gaussian_sigma(E, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E, SIGMA0_P, A_P, E0_P)

    evaluate_model(
        dn_exp, tau_exp, sigma_n, sigma_p,
        Dit0=6.0e11,
        Qfix_cm2=1.0e12,
    )
