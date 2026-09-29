#!/usr/bin/env python3
"""
Fit Dit_0g and Qf to experimental DH lifetime data.

For each dataset (DH0hr = before, DH1000hrs = after) an independent optimisation
finds the Gaussian Dit amplitude (Dit_0g) and fixed surface charge (Qf) that
minimise the log-RMSE between the modelled and measured τ_eff vs Δn curves.

Physics model (unchanged):
  - Single Gaussian Dit:  D_it(E) = Dit_0g · exp[−½·((E−E0)/σ)²]
    with E0 = GAUSS_E0, σ = GAUSS_SIGMA (fixed from config)
  - Energy-dependent sigma: σ_n , σ_p are Gaussians with parameters from config.
  - τ_eff via Matthiessen: 1/τ_eff = 1/τ_surf + 1/τ_intr + 1/τ_bulk

Output:
  figures/fit_dit_qf/fit_comparison.png
  results/fit_dit_qf_results.xlsx
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as elementary_charge
from scipy.optimize import minimize

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N, A_N, E0_N,
    SIGMA0_P, A_P, E0_P,
    GAUSS_E0, GAUSS_SIGMA,
    T, W, Ndop_bulk, dop_type_bulk, Ndop_emitter, dop_type_emitter,
    TAU_N0_BULK, TAU_P0_BULK,
    lw,
)

from physics import (
    create_energy_array,
    calculate_gaussian_sigma,
    surfaceLifetime, intrinsicLifetime, ni_func, Dig_func,
    calculate_srh_lifetime_injection_dependent,
    rear_j0_lifetime,
)

from helpers import (
    equilibrium_concentrations,
    effective_lifetime,
    effective_lifetime_terms,
    setup_plot_style,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dh_data(filepath):
    """Load DH lifetime data with columns MCD_(cm-3) and Effective_Lifetime_(s).

    Tries each sheet in order and returns the first one that contains both
    required columns (needed because some 2_B files store data in Sheet2).
    """
    xl = pd.ExcelFile(filepath)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        if "MCD_(cm-3)" in df.columns and "Effective_Lifetime_(s)" in df.columns:
            dn  = df["MCD_(cm-3)"].values
            tau = df["Effective_Lifetime_(s)"].values
            mask = (dn > 0) & (tau > 0) & np.isfinite(dn) & np.isfinite(tau)
            return dn[mask], tau[mask]
    raise ValueError(
        f"No sheet with 'MCD_(cm-3)' and 'Effective_Lifetime_(s)' found in {filepath}. "
        f"Available sheets: {xl.sheet_names}"
    )


# ============================================================================
# FITTING
# ============================================================================

def subsample_log(dn, tau, n_pts=25):
    n = len(dn)
    if n <= n_pts:
        return dn, tau

    idx = np.unique(
        np.r_[
            0,
            np.round(np.logspace(0, np.log10(n - 1), n_pts)).astype(int),
            n - 1,
        ]
    )

    return dn[idx], tau[idx]


def taueff_front_only_no_rear(dn_sorted, tau_sorted, Dit0, Qfix_C, sigma_n, sigma_p,
                      E0_g=None, gauss_sigma=None):
    """Compute effective lifetime with continuity across injection levels.

    Identical physics to fitting.taueff() but processes dn in ascending
    order and passes the converged ns to the next injection level as the
    fsolve initial guess.  This prevents the coarse search from jumping
    between solution branches near flat-band (small |Qf|).

    Parameters
    ----------
    dn_sorted, tau_sorted : arrays sorted ascending by dn.
    E0_g : float or None
        Gaussian Dit peak position (eV).  Defaults to GAUSS_E0 from config.
    gauss_sigma : float or None
        Width of the Gaussian Dit distribution (eV).  Defaults to
        GAUSS_SIGMA from config.  Override for the after-DH dataset.

    Returns
    -------
    tau_fit : np.ndarray
    rmse    : float  (log-scale RMSE vs tau_sorted)
    """
    if E0_g is None:
        E0_g = GAUSS_E0
    if gauss_sigma is None:
        gauss_sigma = GAUSS_SIGMA
    ni_b   = ni_func(T, Ndop_bulk, dop_type_bulk)
    E      = create_energy_array()
    Dit_E  = Dig_func(E, E0_g, Dit0, gauss_sigma)

    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    tau_fit = []
    prev_ns = None   # continuity seed — None on first point -> coarse search

    for dn_i, _ in zip(dn_sorted, tau_sorted):
        n = n0 + dn_i
        p = p0 + dn_i

        result = surfaceLifetime(
            n0, p0, n, p, dn_i,
            Qfix_C, T,
            Ndop_emitter, Ndop_bulk,
            dop_type_emitter, dop_type_bulk,
            dn_i, Dit_E, sigma_n, sigma_p,
            return_diagnostics=True,
            ns_init=prev_ns,       # skip coarse search after first point
        )
        tau_surf, diag = result
        prev_ns = diag["ns"]       # seed for next injection level

        tau_intr = intrinsicLifetime(n0, p0, n, p, dn_i)
        tau_bulk = calculate_srh_lifetime_injection_dependent(
            dn_i, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)
        tau_eff  = effective_lifetime(tau_surf, tau_intr, tau_bulk)
        tau_fit.append(tau_eff)

    tau_fit = np.array(tau_fit)
    err  = (np.log(tau_fit) - np.log(tau_sorted)) ** 2
    rmse = np.sqrt(np.mean(err))
    return tau_fit, rmse


def taueff_continuous_with_j0rear(dn_sorted, tau_sorted, Dit0, Qfix_C, sigma_n, sigma_p,
                                   J0_rear=0.0, E0_g=None, gauss_sigma=None):
    """Like taueff_continuous() but separates front and rear surface contributions.

    The front surface (Dit/Qf) is treated as a SINGLE surface:
        tau_front = W / S   (not W / 2S)

    The rear surface is parameterised by a lumped saturation current:
        tau_rear  = rear_j0_lifetime(J0_rear, ...)

    Total effective lifetime via Matthiessen's rule:
        1/tau_eff = 1/tau_front + 1/tau_rear + 1/tau_intr + 1/tau_bulk

    When J0_rear = 0 the rear is perfectly passivated (tau_rear → ∞) and
    only the front contributes to surface recombination.

    Parameters
    ----------
    J0_rear : float
        Rear saturation current density [A/cm²].  May be fixed or fitted.
    E0_g, gauss_sigma : float or None
        Gaussian Dit peak position and width.  Defaults to GAUSS_E0 / GAUSS_SIGMA.

    Returns
    -------
    tau_fit : np.ndarray
    rmse    : float  (log-scale RMSE vs tau_sorted)
    """
    if E0_g is None:
        E0_g = GAUSS_E0
    if gauss_sigma is None:
        gauss_sigma = GAUSS_SIGMA

    ni_b  = ni_func(T, Ndop_bulk, dop_type_bulk)
    E     = create_energy_array()
    Dit_E = Dig_func(E, E0_g, Dit0, gauss_sigma)

    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    tau_fit = []
    prev_ns = None

    for dn_i, _ in zip(dn_sorted, tau_sorted):
        n = n0 + dn_i
        p = p0 + dn_i

        result = surfaceLifetime(
            n0, p0, n, p, dn_i,
            Qfix_C, T,
            Ndop_emitter, Ndop_bulk,
            dop_type_emitter, dop_type_bulk,
            dn_i, Dit_E, sigma_n, sigma_p,
            return_diagnostics=True,
            ns_init=prev_ns,
        )
        # surfaceLifetime now returns W/S (single front surface)
        tau_front, diag = result
        prev_ns = diag["ns"]

        tau_rear  = rear_j0_lifetime(J0_rear, n, p, dn_i, ni_b, W)
        tau_intr  = intrinsicLifetime(n0, p0, n, p, dn_i)
        tau_bulk  = calculate_srh_lifetime_injection_dependent(
            dn_i, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)

        tau_eff = effective_lifetime_terms(tau_front, tau_rear, tau_intr, tau_bulk)
        tau_fit.append(tau_eff)

    tau_fit = np.array(tau_fit)
    err  = (np.log(tau_fit) - np.log(tau_sorted)) ** 2
    rmse = np.sqrt(np.mean(err))
    return tau_fit, rmse


def compute_lifetime_components(dn_sorted, Dit0, Qfix_C, sigma_n, sigma_p,
                               J0_rear=0.0, E0_g=None, gauss_sigma=None):
    """Return per-injection-level breakdown of every lifetime component.

    Returns
    -------
    dict with arrays: tau_front, tau_rear, tau_intr, tau_bulk, tau_eff
    """
    if E0_g is None:
        E0_g = GAUSS_E0
    if gauss_sigma is None:
        gauss_sigma = GAUSS_SIGMA

    ni_b  = ni_func(T, Ndop_bulk, dop_type_bulk)
    E     = create_energy_array()
    Dit_E = Dig_func(E, E0_g, Dit0, gauss_sigma)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    tau_front_arr, tau_rear_arr, tau_intr_arr, tau_bulk_arr, tau_eff_arr = [], [], [], [], []
    prev_ns = None

    dummy_tau = np.ones_like(dn_sorted)   # placeholder for the signature
    for dn_i, _ in zip(dn_sorted, dummy_tau):
        n = n0 + dn_i
        p = p0 + dn_i

        result = surfaceLifetime(
            n0, p0, n, p, dn_i, Qfix_C, T,
            Ndop_emitter, Ndop_bulk, dop_type_emitter, dop_type_bulk,
            dn_i, Dit_E, sigma_n, sigma_p,
            return_diagnostics=True, ns_init=prev_ns,
        )
        tau_front, diag = result
        prev_ns = diag["ns"]

        tau_rear  = rear_j0_lifetime(J0_rear, n, p, dn_i, ni_b, W)
        tau_intr  = intrinsicLifetime(n0, p0, n, p, dn_i)
        tau_bulk  = calculate_srh_lifetime_injection_dependent(
            dn_i, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)
        tau_eff   = effective_lifetime_terms(tau_front, tau_rear, tau_intr, tau_bulk)

        tau_front_arr.append(tau_front)
        tau_rear_arr.append(tau_rear)
        tau_intr_arr.append(tau_intr)
        tau_bulk_arr.append(tau_bulk)
        tau_eff_arr.append(tau_eff)

    return {
        "tau_front": np.array(tau_front_arr),
        "tau_rear":  np.array(tau_rear_arr),
        "tau_intr":  np.array(tau_intr_arr),
        "tau_bulk":  np.array(tau_bulk_arr),
        "tau_eff":   np.array(tau_eff_arr),
    }


def print_component_table(label, dn_sorted, comps, tau_sorted_exp=None):
    """Print a short table of lifetime components at 5 representative Δn.

    Parameters
    ----------
    tau_sorted_exp : np.ndarray or None
        Experimental τ values (same order as dn_sorted).  When provided,
        an extra column showing the measured lifetime is appended so the
        model-vs-experiment comparison is immediately visible.
    """
    n = len(dn_sorted)
    idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    show_exp = tau_sorted_exp is not None
    print(f"\n  [{label}] Lifetime component breakdown (ms):")
    header = (f"  {'dn (cm-3)':>12}  {'t_front':>9}  {'t_rear':>9}"
              f"  {'t_intr':>9}  {'t_bulk':>9}  {'t_eff':>9}")
    if show_exp:
        header += f"  {'t_exp':>9}  {'ratio':>7}"
    header += f"  {'limit':>7}"
    print(header)
    print("  " + "-" * (95 if show_exp else 75))
    for i in idxs:
        taus = {k: comps[k][i] * 1e3 for k in comps}
        vals = [taus["tau_front"], taus["tau_rear"],
                taus["tau_intr"],  taus["tau_bulk"], taus["tau_eff"]]
        # identify the smallest finite component (limiting term)
        finite_keys = [(k, comps[k][i]) for k in
                       ("tau_front", "tau_rear", "tau_intr", "tau_bulk")
                       if np.isfinite(comps[k][i]) and comps[k][i] > 0]
        lim = min(finite_keys, key=lambda x: x[1])[0] if finite_keys else "-"
        lim_short = lim.replace("tau_", "")
        row = (f"  {dn_sorted[i]:>12.2e}  "
               + "  ".join(f"{v:>9.3f}" for v in vals))
        if show_exp:
            t_exp_ms = tau_sorted_exp[i] * 1e3
            ratio = comps["tau_eff"][i] / tau_sorted_exp[i]
            row += f"  {t_exp_ms:>9.3f}  {ratio:>7.3f}"
        row += f"  {lim_short:>7}"
        print(row)
    if show_exp:
        # Overall log-RMSE at these 5 representative points
        pred = np.array([comps["tau_eff"][i] for i in idxs])
        expt = np.array([tau_sorted_exp[i] for i in idxs])
        spot_rmse = np.sqrt(np.mean((np.log(pred) - np.log(expt)) ** 2))
        print(f"  (spot log-RMSE at these 5 points: {spot_rmse:.4f})")


def _bin_weighted_rmse(tau_pred, tau_actual, dn_vals, weights=(1.0, 1.0, 1.0)):
    """Log-RMSE averaged across 3 injection-level bins with configurable weights.

    Divides log10(Δn) into 3 equal-width bins (low / mid / high injection)
    and returns the weighted mean of per-bin RMSE values.

    Parameters
    ----------
    weights : (w_low, w_mid, w_high)
        Relative weights for each injection bin.  Increase w_high to penalise
        the high-injection region more (pushes Dit higher).
        Increase w_low to penalise the low-injection region more (adjusts Qf).
    """
    log_dn = np.log10(dn_vals)
    lo, hi = log_dn.min(), log_dn.max()
    step = (hi - lo) / 3.0
    errs, wts = [], []
    for i in range(3):
        lo_b = lo + i * step
        hi_b = lo_b + step + (1e-9 if i == 2 else 0)   # ensure last point included
        m = (log_dn >= lo_b) & (log_dn < hi_b)
        if m.sum() > 1:
            e = np.sqrt(np.mean((np.log(tau_pred[m]) - np.log(tau_actual[m])) ** 2))
            errs.append(e)
            wts.append(weights[i])
    if not errs:
        return 1e6
    return float(np.dot(wts, errs) / np.sum(wts))


def fit_dit_qf(dn_all, tau_all, sigma_n, sigma_p, label="",
               dit_range=None, qf_range=None,
               n_dit=12, n_qf=12, n_e0=8,
               bin_weights=(1.0, 1.0, 1.0),
               fixed_e0g=None,
               low_inj_anchor=0.0,
               peak_inj_anchor=0.0,
               high_inj_anchor=0.0,
               gauss_sigma=None,
               j0_rear=0.0,
               j0rear_range=None,
               n_j0rear=5,
               n_multistart=1):
    """Fit Dit_0g, Qf (and optionally J0_rear) to a single (dn, tau) dataset.

    Strategy:
      1. Sort data by Δn and subsample to ~25 log-spaced points for the
         objective function.
      2. Coarse grid scan.  Dimensionality depends on which parameters are free:
           - 2-D (log_Dit, log_Qf)               : fixed E0_g, fixed J0_rear
           - 3-D (log_Dit, log_Qf, log_J0rear)   : fixed E0_g, fitted J0_rear
           - 3-D (log_Dit, log_Qf, E0_g)          : free E0_g, fixed J0_rear
           - 4-D (log_Dit, log_Qf, E0_g, log_J0rear) : free E0_g, fitted J0_rear
      3. Nelder-Mead refinement from the best grid point(s).

    Parameters
    ----------
    dit_range  : (lo, hi)  log10 bounds for the Dit grid search.
    qf_range   : (lo, hi)  log10 bounds for the Qf  grid search.
    n_dit, n_qf, n_e0 : grid points along each axis.
    fixed_e0g  : float or None.
        If provided, E0_g is fixed to this value and not optimised.
        If None, E0_g is a free parameter (original behaviour).
    j0_rear    : float (default 0.0)
        Fixed rear saturation current density [A/cm²].  Used only when
        j0rear_range is None.  Ignored when j0rear_range is given.
    j0rear_range : (lo, hi) or None
        log10 bounds for J0_rear grid search (e.g. (-15.3, -13.8)).
        When provided J0_rear becomes a free fit parameter.
        When None, J0_rear is fixed to j0_rear.
    n_j0rear   : int (default 5)
        Grid points along the log10(J0_rear) axis.
    low_inj_anchor : float (default 0)
        Extra weight on the lowest-injection point error.
    high_inj_anchor : float (default 0)
        Extra weight on the highest-injection point error.

    Returns
    -------
    Dit_0g  : float  fitted Dit amplitude (cm⁻² eV⁻¹)
    Qf_cm2  : float  fitted fixed charge magnitude (cm⁻²); negative charge convention
    E0_g    : float  E0_g used (= fixed_e0g when fixed, otherwise optimised value)
    j0rear  : float  J0_rear used — fitted value when j0rear_range given, else j0_rear
    rmse    : float  final log-RMSE on the full dataset
    """
    if dit_range is None or qf_range is None:
        raise ValueError("dit_range and qf_range must be specified in the datasets list.")

    fit_j0rear = j0rear_range is not None   # True → J0_rear is a free parameter

    # Sort by carrier density (ascending)
    order = np.argsort(dn_all)
    dn_sorted, tau_sorted = dn_all[order], tau_all[order]

    # Subsample for fast objective evaluation
    dn_fit, tau_fit = subsample_log(dn_sorted, tau_sorted, n_pts=25)

    # ------------------------------------------------------------------
    # Helper to build the anchor-augmented RMSE from tau_pred
    # ------------------------------------------------------------------
    def _augmented_rmse(tau_pred):
        rmse_val = _bin_weighted_rmse(tau_pred, tau_fit, dn_fit, weights=bin_weights)
        if low_inj_anchor > 0:
            rmse_val += low_inj_anchor * abs(np.log(tau_pred[0] / tau_fit[0]))
        if high_inj_anchor > 0:
            rmse_val += high_inj_anchor * abs(np.log(tau_pred[-1] / tau_fit[-1]))
        if peak_inj_anchor > 0:
            idx_peak = np.argmax(tau_fit)
            rmse_val += peak_inj_anchor * abs(
                np.log(tau_pred[idx_peak] / tau_fit[idx_peak]))
        return rmse_val

    if fixed_e0g is not None:
        # ==============================================================
        # Fixed E0_g branches
        # ==============================================================
        _e0g = fixed_e0g

        if fit_j0rear:
            # ----------------------------------------------------------
            # 3-D: (log_Dit, log_Qf, log_J0rear)  with E0_g fixed
            # ----------------------------------------------------------
            def objective(x):
                log_Dit, log_Qf, log_J0rear = x
                if not (dit_range[0] <= log_Dit <= dit_range[1]):
                    return 1e6
                if not (qf_range[0] <= log_Qf <= qf_range[1]):
                    return 1e6
                if not (j0rear_range[0] <= log_J0rear <= j0rear_range[1]):
                    return 1e6
                Dit0   = 10.0 ** log_Dit
                Qfix_C = -(10.0 ** log_Qf) * elementary_charge
                J0r    = 10.0 ** log_J0rear
                try:
                    tau_pred, _ = taueff_continuous_with_j0rear(
                        dn_fit, tau_fit, Dit0, Qfix_C, sigma_n, sigma_p,
                        J0_rear=J0r, E0_g=_e0g, gauss_sigma=gauss_sigma)
                    return _augmented_rmse(tau_pred)
                except Exception:
                    return 1e6

            log_Dit_grid    = np.linspace(dit_range[0],    dit_range[1],    n_dit)
            log_Qf_grid     = np.linspace(qf_range[0],     qf_range[1],     n_qf)
            log_J0rear_grid = np.linspace(j0rear_range[0], j0rear_range[1], n_j0rear)
            n_eval = len(log_Dit_grid) * len(log_Qf_grid) * len(log_J0rear_grid)
            print(f"\n  [{label}] 3-D grid scan "
                  f"(E0_g fixed={_e0g:.4f} eV, J0_rear fitted), "
                  f"{n_dit}x{n_qf}x{n_j0rear} = {n_eval} evals...")

            grid_results = []
            for ld in log_Dit_grid:
                for lq in log_Qf_grid:
                    for lj in log_J0rear_grid:
                        val = objective([ld, lq, lj])
                        grid_results.append((val, [ld, lq, lj]))
            grid_results.sort(key=lambda t: t[0])
            best_grid_rmse, best_x = grid_results[0]
            print(f"  [{label}] Grid best: log_Dit={best_x[0]:.2f}, "
                  f"log_Qf={best_x[1]:.2f}, log_J0rear={best_x[2]:.2f}, "
                  f"RMSE={best_grid_rmse:.4f}")

            k = min(n_multistart, len(grid_results))
            print(f"  [{label}] Multi-start NM ({k} starts)...")
            best_nm_rmse = np.inf
            log_Dit_opt, log_Qf_opt, log_J0rear_opt = best_x
            for _, x0 in grid_results[:k]:
                res = minimize(objective, x0, method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6,
                                        "maxiter": 3000})
                if res.fun < best_nm_rmse:
                    best_nm_rmse = res.fun
                    log_Dit_opt, log_Qf_opt, log_J0rear_opt = res.x
            print(f"  [{label}] Best NM:   log_Dit={log_Dit_opt:.4f}, "
                  f"log_Qf={log_Qf_opt:.4f}, log_J0rear={log_J0rear_opt:.4f}, "
                  f"obj={best_nm_rmse:.4f}")
            E0_g_opt = _e0g

        else:
            # ----------------------------------------------------------
            # 2-D: (log_Dit, log_Qf)  with E0_g fixed — original
            # ----------------------------------------------------------
            def objective(x):
                log_Dit, log_Qf = x
                if not (dit_range[0] <= log_Dit <= dit_range[1]):
                    return 1e6
                if not (qf_range[0] <= log_Qf <= qf_range[1]):
                    return 1e6
                Dit0   = 10.0 ** log_Dit
                Qfix_C = -(10.0 ** log_Qf) * elementary_charge
                try:
                    tau_pred, _ = taueff_continuous_with_j0rear(
                        dn_fit, tau_fit, Dit0, Qfix_C, sigma_n, sigma_p,
                        J0_rear=j0_rear, E0_g=_e0g, gauss_sigma=gauss_sigma)
                    return _augmented_rmse(tau_pred)
                except Exception:
                    return 1e6

            log_Dit_grid = np.linspace(dit_range[0], dit_range[1], n_dit)
            log_Qf_grid  = np.linspace(qf_range[0],  qf_range[1],  n_qf)
            n_eval = len(log_Dit_grid) * len(log_Qf_grid)
            print(f"\n  [{label}] 2-D grid scan (E0_g fixed={_e0g:.4f} eV), "
                  f"{n_dit}x{n_qf} = {n_eval} evals...")

            grid_results = []
            for ld in log_Dit_grid:
                for lq in log_Qf_grid:
                    val = objective([ld, lq])
                    grid_results.append((val, [ld, lq]))
            grid_results.sort(key=lambda t: t[0])
            best_grid_rmse, best_x = grid_results[0]
            print(f"  [{label}] Grid best: log_Dit={best_x[0]:.2f}, "
                  f"log_Qf={best_x[1]:.2f}, RMSE={best_grid_rmse:.4f}")

            k = min(n_multistart, len(grid_results))
            print(f"  [{label}] Multi-start NM ({k} starts)...")
            best_nm_rmse = np.inf
            log_Dit_opt, log_Qf_opt = best_x
            for _, x0 in grid_results[:k]:
                res = minimize(objective, x0, method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6,
                                        "maxiter": 3000})
                if res.fun < best_nm_rmse:
                    best_nm_rmse = res.fun
                    log_Dit_opt, log_Qf_opt = res.x
            print(f"  [{label}] Best NM:   log_Dit={log_Dit_opt:.4f}, "
                  f"log_Qf={log_Qf_opt:.4f}, obj={best_nm_rmse:.4f}")
            E0_g_opt = _e0g

    else:
        # ==============================================================
        # Free E0_g branches
        # ==============================================================

        if fit_j0rear:
            # ----------------------------------------------------------
            # 4-D: (log_Dit, log_Qf, E0_g, log_J0rear)
            # ----------------------------------------------------------
            def objective(x):
                log_Dit, log_Qf, E0_g, log_J0rear = x
                if not (dit_range[0] <= log_Dit <= dit_range[1]):
                    return 1e6
                if not (qf_range[0] <= log_Qf <= qf_range[1]):
                    return 1e6
                if not (0.30 < E0_g < 0.85):
                    return 1e6
                if not (j0rear_range[0] <= log_J0rear <= j0rear_range[1]):
                    return 1e6
                Dit0   = 10.0 ** log_Dit
                Qfix_C = -(10.0 ** log_Qf) * elementary_charge
                J0r    = 10.0 ** log_J0rear
                try:
                    tau_pred, _ = taueff_continuous_with_j0rear(
                        dn_fit, tau_fit, Dit0, Qfix_C, sigma_n, sigma_p,
                        J0_rear=J0r, E0_g=E0_g, gauss_sigma=gauss_sigma)
                    return _augmented_rmse(tau_pred)
                except Exception:
                    return 1e6

            log_Dit_grid    = np.linspace(dit_range[0],    dit_range[1],    n_dit)
            log_Qf_grid     = np.linspace(qf_range[0],     qf_range[1],     n_qf)
            E0_g_grid       = np.linspace(0.30, 0.85, n_e0)
            log_J0rear_grid = np.linspace(j0rear_range[0], j0rear_range[1], n_j0rear)
            n_eval = (len(log_Dit_grid) * len(log_Qf_grid)
                      * len(E0_g_grid) * len(log_J0rear_grid))
            print(f"\n  [{label}] 4-D grid scan (J0_rear fitted), "
                  f"{n_dit}x{n_qf}x{n_e0}x{n_j0rear} = {n_eval} evals...")

            grid_results = []
            for ld in log_Dit_grid:
                for lq in log_Qf_grid:
                    for e0 in E0_g_grid:
                        for lj in log_J0rear_grid:
                            val = objective([ld, lq, e0, lj])
                            grid_results.append((val, [ld, lq, e0, lj]))
            grid_results.sort(key=lambda t: t[0])
            best_grid_rmse, best_x = grid_results[0]
            print(f"  [{label}] Grid best: log_Dit={best_x[0]:.2f}, "
                  f"log_Qf={best_x[1]:.2f}, E0_g={best_x[2]:.3f} eV, "
                  f"log_J0rear={best_x[3]:.2f}, RMSE={best_grid_rmse:.4f}")

            k = min(n_multistart, len(grid_results))
            print(f"  [{label}] Multi-start NM ({k} starts)...")
            best_nm_rmse = np.inf
            log_Dit_opt, log_Qf_opt, E0_g_opt, log_J0rear_opt = best_x
            for _, x0 in grid_results[:k]:
                res = minimize(objective, x0, method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6,
                                        "maxiter": 3000})
                if res.fun < best_nm_rmse:
                    best_nm_rmse = res.fun
                    log_Dit_opt, log_Qf_opt, E0_g_opt, log_J0rear_opt = res.x
            print(f"  [{label}] Best NM:   log_Dit={log_Dit_opt:.4f}, "
                  f"log_Qf={log_Qf_opt:.4f}, E0_g={E0_g_opt:.4f} eV, "
                  f"log_J0rear={log_J0rear_opt:.4f}, obj={best_nm_rmse:.4f}")

        else:
            # ----------------------------------------------------------
            # 3-D: (log_Dit, log_Qf, E0_g) — original behaviour
            # ----------------------------------------------------------
            def objective(x):
                log_Dit, log_Qf, E0_g = x
                if not (dit_range[0] <= log_Dit <= dit_range[1]):
                    return 1e6
                if not (qf_range[0] <= log_Qf <= qf_range[1]):
                    return 1e6
                if not (0.30 < E0_g < 0.85):
                    return 1e6
                Dit0   = 10.0 ** log_Dit
                Qfix_C = -(10.0 ** log_Qf) * elementary_charge
                try:
                    tau_pred, _ = taueff_continuous_with_j0rear(
                        dn_fit, tau_fit, Dit0, Qfix_C, sigma_n, sigma_p,
                        J0_rear=j0_rear, E0_g=E0_g, gauss_sigma=gauss_sigma)
                    return _augmented_rmse(tau_pred)
                except Exception:
                    return 1e6

            log_Dit_grid = np.linspace(dit_range[0], dit_range[1], n_dit)
            log_Qf_grid  = np.linspace(qf_range[0],  qf_range[1],  n_qf)
            E0_g_grid    = np.linspace(0.30, 0.85, n_e0)
            n_eval = len(log_Dit_grid) * len(log_Qf_grid) * len(E0_g_grid)
            print(f"\n  [{label}] 3-D grid scan "
                  f"{n_dit}x{n_qf}x{n_e0} = {n_eval} evals...")

            grid_results = []
            for ld in log_Dit_grid:
                for lq in log_Qf_grid:
                    for e0 in E0_g_grid:
                        val = objective([ld, lq, e0])
                        grid_results.append((val, [ld, lq, e0]))
            grid_results.sort(key=lambda t: t[0])
            best_grid_rmse, best_x = grid_results[0]
            print(f"  [{label}] Grid best: log_Dit={best_x[0]:.2f}, "
                  f"log_Qf={best_x[1]:.2f}, E0_g={best_x[2]:.3f} eV, "
                  f"RMSE={best_grid_rmse:.4f}")

            k = min(n_multistart, len(grid_results))
            print(f"  [{label}] Multi-start NM ({k} starts)...")
            best_nm_rmse = np.inf
            log_Dit_opt, log_Qf_opt, E0_g_opt = best_x
            for _, x0 in grid_results[:k]:
                res = minimize(objective, x0, method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-6,
                                        "maxiter": 3000})
                if res.fun < best_nm_rmse:
                    best_nm_rmse = res.fun
                    log_Dit_opt, log_Qf_opt, E0_g_opt = res.x
            print(f"  [{label}] Best NM:   log_Dit={log_Dit_opt:.4f}, "
                  f"log_Qf={log_Qf_opt:.4f}, E0_g={E0_g_opt:.4f} eV, "
                  f"obj={best_nm_rmse:.4f}")

    # --- Final RMSE on the full (non-subsampled) sorted dataset ---
    Dit_0g  = 10.0 ** log_Dit_opt
    Qf_mag  = 10.0 ** log_Qf_opt          # magnitude [cm⁻²], always positive
    Qf_cm2  = -Qf_mag                      # physical value: negative fixed charge
    Qfix_C_final = Qf_cm2 * elementary_charge   # [C/cm²], negative
    j0rear_final = 10.0 ** log_J0rear_opt if fit_j0rear else j0_rear
    _, rmse_full = taueff_continuous_with_j0rear(
        dn_sorted, tau_sorted, Dit_0g, Qfix_C_final, sigma_n, sigma_p,
        J0_rear=j0rear_final, E0_g=E0_g_opt, gauss_sigma=gauss_sigma)
    return Dit_0g, Qf_cm2, E0_g_opt, j0rear_final, rmse_full


# ============================================================================
# MAIN
# ============================================================================

def main():
    project_root = os.path.dirname(script_dir)
    data_dir     = os.path.join(project_root, "data")
    figures_dir  = os.path.join(project_root, "figures", "fit_dit_qf")
    results_dir  = os.path.join(project_root, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # USER INPUT — Gaussian Dit width for the after-DH dataset
    # =========================================================================
    # GAUSS_SIGMA (config) = 0.18 eV is used for DH 0 hr.
    # Change GAUSS_SIGMA_AFTER below to explore how a wider/narrower Dit
    # distribution affects the DH 1000 hr fit.  The optimiser (Dit_0g, Qf)
    # is re-run each time; only the shape parameter σ is fixed by this value.
    GAUSS_SIGMA_AFTER = 0.18   # <<< change this (eV)
    # =========================================================================

    # =========================================================================
    # USER INPUT — J0_rear search range (set independently per dataset)
    # =========================================================================
    # J0_rear (rear saturation current density [A/cm²]) is a constrained free
    # parameter.  Bounds are specified in log10 space.
    #
    # Typical literature values for silicon:
    #   Excellent passivation : ~1e-15 A/cm²  (log10 = -15.0)
    #   Good passivation      : ~5e-15 A/cm²  (log10 = -14.3)
    #   Mild degradation      : ~1e-14 A/cm²  (log10 = -14.0)
    #   Significant degradation: ~1e-13 A/cm² (log10 = -13.0)
    J0REAR_RANGE_BEFORE = (-15, -13)   # <<< log10 bounds for DH 0 hr
    J0REAR_RANGE_AFTER  = (-15, -13)   # <<< log10 bounds for DH 1000 hr
    N_J0REAR            = 10                # grid points along the J0_rear axis
    # =========================================================================

    # --- Gaussian energy-dependent capture cross sections (from config) ---
    E       = create_energy_array()
    sigma_n = calculate_gaussian_sigma(E, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E, SIGMA0_P, A_P, E0_P)

    # --- Datasets ---
    # grid_kwargs: per-dataset grid search bounds passed to fit_dit_qf.
    # After DH stress: Dit typically much higher, Qf can decrease → widen both ranges.
    datasets = [
        (
            "DH 0 hr (Before)",
            os.path.join(data_dir, "2_B_DH0hr.xlsx"),
            # qf_range upper bound raised to 14.0 — previous run hit the 13.0 ceiling.
            # J0_rear is now a free fit parameter; j0rear_range replaces the fixed value.
            dict(fixed_e0g=GAUSS_E0,
                 dit_range=(8.0, 12.0), qf_range=(9.0, 14.0),
                 n_dit=15, n_qf=15,
                 j0rear_range=J0REAR_RANGE_BEFORE, n_j0rear=N_J0REAR),
        ),
        (
            "DH 1000 hrs (After)",
            os.path.join(data_dir, "2_B_DH1000hrs_new.xlsx"),
            # After DH the curve typically shows a hump (low tau at low injection,
            # peak at mid injection, drops at high injection).  This requires:
            #   - large negative Qf (surface inversion in n-type)
            #   - significant Dit (SRH loss in inversion layer)
            #   - E0_g potentially shifted by DH-induced traps
            #
            # fixed_e0g=GAUSS_E0 → 3-D search (log_Dit, log_Qf, log_J0rear).
            # E0_g fixed at 0.56 eV (same as before-DH) for physical comparability.
            # qf_range covers the strong-inversion regime (>1e11.5 cm⁻²).
            dict(fixed_e0g=GAUSS_E0,
                 dit_range=(8.0, 12.0), qf_range=(9.0, 14.0),
                 n_dit=15, n_qf=15, n_e0=10,
                 bin_weights=(2.0, 4.0, 2.0),
                 low_inj_anchor=2.0,
                 peak_inj_anchor=5.0,
                 high_inj_anchor=2.0,
                 gauss_sigma=GAUSS_SIGMA_AFTER,
                 j0rear_range=J0REAR_RANGE_AFTER, n_j0rear=N_J0REAR,
                 n_multistart=1),
        ),
    ]

    fit_results = []

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    colors      = ["navy", "darkred"]          # experimental data markers (dark)
    line_colors = ["cornflowerblue", "tomato"]  # model fit lines (lighter)

    for (label, path, grid_kw), color, line_color in zip(datasets, colors, line_colors):
        print(f"\n{'='*55}")
        print(f"  Dataset: {label}")
        print(f"  File:    {os.path.basename(path)}")
        print(f"{'='*55}")

        dn_raw, tau_raw = load_dh_data(path)
        print(f"  Loaded {len(dn_raw)} data points  "
              f"(dn: {dn_raw.min():.2e}-{dn_raw.max():.2e} cm^-3)")
        # --- Quick summary of experimental data shape ---
        order_diag = np.argsort(dn_raw)
        _dn_d  = dn_raw[order_diag]
        _tau_d = tau_raw[order_diag] * 1e3   # convert to ms
        idx_peak_exp = np.argmax(_tau_d)
        print(f"  Exp tau range:   {_tau_d.min():.3f} - {_tau_d.max():.3f} ms  "
              f"(peak at dn={_dn_d[idx_peak_exp]:.2e} cm^-3, "
              f"tau_peak={_tau_d[idx_peak_exp]:.3f} ms)")
        print(f"  Exp tau at low:  {_tau_d[0]:.3f} ms  "
              f"(dn={_dn_d[0]:.2e})  |  "
              f"high: {_tau_d[-1]:.3f} ms  (dn={_dn_d[-1]:.2e})")

        # Sort ascending for continuity solver
        order = np.argsort(dn_raw)
        dn_s, tau_s = dn_raw[order], tau_raw[order]

        # Extract per-dataset options (None → config defaults)
        ds_gauss_sigma = grid_kw.get("gauss_sigma", None)
        gs_used = ds_gauss_sigma if ds_gauss_sigma is not None else GAUSS_SIGMA

        # fixed_e0g is per-dataset: pull from grid_kw if present, else use GAUSS_E0.
        ds_fixed_e0g = grid_kw.pop("fixed_e0g", GAUSS_E0)
        Dit_0g, Qf_cm2, E0_g, j0rear_opt, rmse = fit_dit_qf(
            dn_raw, tau_raw, sigma_n, sigma_p,
            label=label, fixed_e0g=ds_fixed_e0g,
            **grid_kw)

        print(f"\n  *** Fitted parameters ***")
        print(f"      Dit_0g       = {Dit_0g:.3e} cm^-2 eV^-1")
        print(f"      Qf           = {Qf_cm2:.3e} cm^-2")
        print(f"      J0_rear      = {j0rear_opt:.3e} A/cm^2  (fitted, "
              f"log10={np.log10(j0rear_opt):.2f})")
        print(f"      E0_g         = {E0_g:.4f} eV  (Dit peak position)")
        print(f"      GAUSS_SIGMA  = {gs_used:.4f} eV  (Dit Gaussian width)")
        print(f"      RMSE         = {rmse:.2f}  (log scale)")

        # --- Component breakdown diagnostic ---
        Qfix_C_diag = Qf_cm2 * elementary_charge
        comps = compute_lifetime_components(
            dn_s, Dit_0g, Qfix_C_diag, sigma_n, sigma_p,
            J0_rear=j0rear_opt, E0_g=E0_g, gauss_sigma=ds_gauss_sigma)
        print_component_table(label, dn_s, comps, tau_sorted_exp=tau_s)

        fit_results.append({
            "Dataset":              label,
            "Dit_0g (cm-2 eV-1)":  Dit_0g,
            "Qf (cm-2)":           Qf_cm2,   # negative value
            "J0_rear (A/cm2)":     j0rear_opt,
            "log10_J0rear":        np.log10(j0rear_opt),
            "E0_g (eV)":           E0_g,
            "GAUSS_SIGMA (eV)":    gs_used,
            "RMSE (log)":          rmse,
        })

        # --- Per-dataset component breakdown figure ---
        fig_comp, ax_comp = plt.subplots(figsize=(9, 6))
        ax_comp.loglog(dn_s, tau_s * 1e3, "o", color=color,
                       markersize=9, alpha=0.7, label="Exp data")
        ax_comp.loglog(dn_s, comps["tau_eff"]   * 1e3, "-",  color=color,
                       lw=3, label=r"$\tau_{eff}$ (model)")
        ax_comp.loglog(dn_s, comps["tau_front"] * 1e3, "--", color="darkorange",
                       lw=2, label=r"$\tau_{front}$ (Dit/Qf)")
        finite_rear = np.isfinite(comps["tau_rear"]) & (comps["tau_rear"] < 1e3)
        if finite_rear.any():
            ax_comp.loglog(dn_s[finite_rear], comps["tau_rear"][finite_rear] * 1e3,
                           ":", color="green", lw=2, label=r"$\tau_{rear}$ (J0,rear)")
        # ax_comp.loglog(dn_s, comps["tau_intr"] * 1e3, "-.", color="purple",
                       # lw=2, label=r"$\tau_{intr}$ (Auger+rad)")
        # ax_comp.loglog(dn_s, comps["tau_bulk"] * 1e3, ":",  color="gray",
                       # lw=2, label=r"$\tau_{bulk}$ (SRH)")
        ax_comp.set_xlabel(r"$\Delta n$ (cm$^{-3}$)", fontsize=15)
        ax_comp.set_ylabel(r"Lifetime (ms)", fontsize=15)
        ax_comp.set_title(f"Component breakdown — {label}", fontsize=13)
        ax_comp.legend(frameon=False, fontsize=11)
        ax_comp.tick_params(which="both", direction="in", top=True, right=True)
        fig_comp.tight_layout()
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        comp_path = os.path.join(figures_dir, f"components_{safe_label}_2B.png")
        fig_comp.savefig(comp_path, dpi=200, bbox_inches="tight")
        plt.close(fig_comp)
        print(f"  Component figure -> {comp_path}")

        # Plot experimental data
        ax.loglog(dn_s, tau_s * 1e3, "o", color=color, markersize=16,
                  alpha=0.6, label=f"Exp — {label}")

        # Plot fitted model curve (sorted, continuous solver)
        # Qf_cm2 is negative → multiply by e gives negative C/cm²
        Qfix_C = Qf_cm2 * elementary_charge
        tau_fit, _ = taueff_continuous_with_j0rear(
            dn_s, tau_s, Dit_0g, Qfix_C, sigma_n, sigma_p,
            J0_rear=j0rear_opt, E0_g=E0_g, gauss_sigma=ds_gauss_sigma)
        ax.loglog(dn_s, tau_fit * 1e3, "-", color=line_color,
                  linewidth=lw, label=f"Fit — {label}")

    # --- Figure formatting ---
    ax.set_xlabel(r"Minority carrier density $\Delta n$ (cm$^{-3}$)", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Effective lifetime $\tau_{eff}$ (ms)", fontsize=18, fontweight="bold")
    ax.legend(frameon=False, prop={"family": "Arial", "weight": "bold", "size": 13})
    ax.tick_params(axis="both", which="major", direction="in", length=8, labelsize=14,
                   width=1.5, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=4,
                   width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_ylim(1e-1, 1e1)

    fig.tight_layout()
    fig_path = os.path.join(figures_dir, "fit_comparison_2B.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved -> {fig_path}")

    # --- Save results to Excel ---
    df_out = pd.DataFrame(fit_results)
    xl_path = os.path.join(results_dir, "fit_dit_qf_results_2B.xlsx")
    df_out.to_excel(xl_path, index=False)
    print(f"Results saved -> {xl_path}")

    # --- Summary table ---
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(df_out.to_string(index=False))
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
