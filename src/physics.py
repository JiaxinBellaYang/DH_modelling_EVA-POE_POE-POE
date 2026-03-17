"""
Pure physics functions for the surface recombination model.

This module contains every physics calculation used by the project:
- Carrier concentrations and band-gap narrowing  (ni_func)
- Energy-dependent capture cross sections        (calculate_gaussian_sigma)
- Gaussian Dit distributions                     (Dig_func)
- Bulk SRH lifetime                              (calculate_srh_lifetime_injection_dependent)
- Surface potential solving                      (ns_zero_func, ns_zero_func_full)
- Surface recombination rate                     (J0sv2_func, surfaceLifetime)
- Intrinsic recombination (Richter model)        (intrinsicLifetime)

All shared constants are imported from config.py.
No matplotlib, no data I/O, no optimisation logic.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.constants import e as elementary_charge

import config

# ============================================================================
# RICHTER INTRINSIC-LIFETIME CONSTANTS
# ============================================================================

INTR_RMIN = 0.0
INTR_RMAX = 0.2
INTR_SMIN = 1e7
INTR_SMAX = 1.5e18
INTR_WMAX = 4.0e-18
INTR_WMIN = 1e19
INTR_B2 = 0.54
INTR_B4 = 1.25
INTR_R1 = 320.0
INTR_R2 = 2.5
INTR_S1 = 550.0
INTR_S2 = 3.0
INTR_W1 = 365.0
INTR_W2 = 3.54
INTR_N0EEH = 3.3e17
INTR_N0EHH = 7.0e17
INTR_GEEH_FACTOR = 13.0
INTR_GEHH_FACTOR = 7.5
INTR_GEEH_EXP = 0.6
INTR_GEHH_EXP = 0.63
INTR_AUGER_EEH = 2.5e-31
INTR_AUGER_EHH = 8.5e-32
INTR_AUGER_XXX_COEFF = 3.0e-29
INTR_AUGER_XXX_EXP = 0.92


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_energy_array(start=None, end=None, points=None):
    if start is None: start = config.Ev
    if end is None: end = config.Ec
    if points is None: points = config.ENERGY_POINTS
    """Linearly spaced energy grid from *start* to *end*."""
    return np.linspace(start, end, points)


def calculate_gaussian_sigma(E, sigma0, A, E0):
    """Energy-dependent capture cross section — Eq. (A12).

    σ(E) = σ₀ · exp[−A · (E − E_mid − E₀)²]

    Related to Eq. (A12) by  A = 1/(2μ²),  where μ is the Gaussian width.
    E₀ is specified relative to midgap.
    """
    E_midgap = (config.Ec + config.Ev) / 2
    E_relative_to_midgap = E - E_midgap
    return sigma0 * np.exp(-A * (E_relative_to_midgap - E0) ** 2)


def Dig_func(E, *params):
    """Gaussian Dit distribution — Eq. (A10).

    D_it,g(E) = D_it,0g · exp[−½ · ((E − E₀) / σ)²]

    Also used to approximate band-edge Dit tails (Eqs. A8, A9) by
    centring Gaussians at Ev and Ec.  The original equations define
    exponential tails; the Gaussian form is a modelling choice.
    """
    E0_gauss, Dit_0g, sigma_gauss = params
    return Dit_0g * np.exp(-((E - E0_gauss) / sigma_gauss) ** 2 / 2)


def ni_func(T_local, Ndop, dop_type):
    """Effective intrinsic carrier concentration with band-gap narrowing.

    Uses the Schenk BGN model threshold at Ndop > 1e14 cm⁻³.
    """
    k = 8.617e-5
    Eth = k * T_local
    Egi = 1.206 - 2.73e-4 * T_local
    ni = 1.541e15 * T_local ** 1.712 * np.exp(-Egi / (2 * Eth))
    if Ndop > 1e14:
        Delta_Eg = 4.2e-5 * np.log(Ndop / 1e14) ** 3
        BGN = np.exp(Delta_Eg / Eth)
        ni_eff = ni * np.sqrt(BGN)
    else:
        ni_eff = ni
    return ni_eff


def calculate_srh_lifetime_injection_dependent(Delta_n, n0, p0, tau_n0, tau_p0, ni_eff):
    """Injection-dependent bulk SRH lifetime for a single mid-gap trap level."""
    n1 = ni_eff
    p1 = ni_eff
    numerator = tau_p0 * (n0 + n1 + Delta_n) + tau_n0 * (p0 + p1 + Delta_n)
    denominator = n0 + p0 + Delta_n
    if denominator > 1e-10:
        return numerator / denominator
    return np.inf


# ============================================================================
# SURFACE POTENTIAL SOLVING
# ============================================================================

def ns_zero_func(ns, *params):
    """Charge balance without Q_it — vectorised coarse search — Eq. (A5).

    Solves for ns from the surface charge-neutrality condition:

        ps + ns − pd − nd + N_dop · ψs/(kT)  =  Q_f² / (2q · ε_Si · kT)

    Surface concentrations follow Eq. (A4):
        ns = nd · exp(−ψs / kT),    ps = pd · exp(+ψs / kT)

    NOTE: The variable Phi_s herein calculates the Band Bending Energy
    shift (E_c - E_c,bulk) in eV. Therefore, bands bending "up" (negative 
    electrostatic potential) will mathematically yield a POSITIVE Phi_s.

    Q here is the fixed charge only; Q_it is added in ns_zero_func_full.
    """
    Q, T_local, Ndop_emitter, Ndop_bulk_arg, dop_type_emitter, dop_type_bulk_arg, dn = params

    # Clamp ns to prevent log(negative)
    ns = np.maximum(ns, 1e-20)

    Ndop_surface_n = dop_type_emitter * Ndop_emitter + dop_type_bulk_arg * Ndop_bulk_arg
    Ndop_surface_p = (1 - dop_type_emitter) * Ndop_emitter + (1 - dop_type_bulk_arg) * Ndop_bulk_arg
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)

    Eth = config.kB * T_local
    ni_b = ni_func(T_local, Ndop_bulk_arg, dop_type_bulk_arg)
    ni_e = ni_func(T_local, Ndop_surface, dop_type_emitter)

    nd0 = dop_type_bulk_arg * Ndop_bulk_arg + (1 - dop_type_bulk_arg) * ni_b ** 2 / Ndop_bulk_arg
    pd0 = (1 - dop_type_bulk_arg) * Ndop_bulk_arg + dop_type_bulk_arg * ni_b ** 2 / Ndop_bulk_arg

    pd = pd0 + dn
    nd = nd0 + dn

    ps = (ni_e ** 2 / ni_b ** 2) * pd * nd / ns
    Phi_s = -Eth * np.log((ns * ni_b) / (nd * ni_e))

    fzero = ps - pd + ns - nd + Ndop_surface * Phi_s / Eth - Q ** 2 / (
        2 * elementary_charge * config.eps_Si * Eth
    )
    return fzero


def ns_zero_func_full(ns, *params):
    """Charge balance with Q_it — scalar fsolve refinement — Eqs. (A5) + (2).

    Extends ns_zero_func by replacing Q_f with Q_total = Q_f + Q_it
    in the charge-neutrality condition (Eq. A5).

    Q_it is computed self-consistently via the amphoteric model (Eq. 2):
      Donors  (E < E_mid):   contribute +q when empty   → +q · D_it · (1 − f_t)
      Acceptors (E ≥ E_mid): contribute −q when occupied → −q · D_it · f_t

    f_t is the SRH trap occupation fraction:
        f_t(E) = [σ_n · v_th,n · ns  +  σ_p · v_th,p · p₁(E)]
               / [σ_n · v_th,n · (ns + n₁) + σ_p · v_th,p · (ps + p₁)]
    """
    (Q, T_local, Ndop_emitter, Ndop_bulk_arg, dop_type_emitter, dop_type_bulk_arg,
     dn, Dit_E, sigma_n_arr, sigma_p_arr) = params

    # Clamp ns to prevent log(negative)
    ns = max(float(ns), 1e-20)

    Ndop_surface_n = dop_type_emitter * Ndop_emitter + dop_type_bulk_arg * Ndop_bulk_arg
    Ndop_surface_p = (1 - dop_type_emitter) * Ndop_emitter + (1 - dop_type_bulk_arg) * Ndop_bulk_arg
    Ndop_surface = abs(Ndop_surface_n - Ndop_surface_p)

    Eth = config.kB * T_local
    ni_b = ni_func(T_local, Ndop_bulk_arg, dop_type_bulk_arg)
    ni_e = ni_func(T_local, Ndop_surface, dop_type_emitter)

    nd0 = dop_type_bulk_arg * Ndop_bulk_arg + (1 - dop_type_bulk_arg) * ni_b ** 2 / Ndop_bulk_arg
    pd0 = (1 - dop_type_bulk_arg) * Ndop_bulk_arg + dop_type_bulk_arg * ni_b ** 2 / Ndop_bulk_arg

    pd = pd0 + dn
    nd = nd0 + dn

    ps = (ni_e ** 2 / ni_b ** 2) * pd * nd / ns
    Phi_s = -Eth * np.log((ns * ni_b) / (nd * ni_e))

    # --- Compute interface trapped charge Q_it ---
    E = create_energy_array()
    cn = sigma_n_arr * config.vth_n
    cp = sigma_p_arr * config.vth_p
    n1 = config.NC * np.exp(-(config.Ec - E) / config.kT)
    p1 = config.NV * np.exp(-(E - config.Ev) / config.kT)

    # SRH occupation fraction f_t(E) for each trap level
    denom_ft = cn * (ns + n1) + cp * (ps + p1)
    f_t = np.divide(cn * ns + cp * p1, denom_ft,
                    out=np.full_like(denom_ft, 0.5), where=denom_ft > 0)

    E_mid = (config.Ec + config.Ev) / 2
    mask_don = E < E_mid
    mask_acc = ~mask_don

    # Q_it in C/cm^2  (positive = net positive interface charge)
    Q_it = elementary_charge * (
        np.trapz(Dit_E[mask_don] * (1 - f_t[mask_don]), E[mask_don]) -
        np.trapz(Dit_E[mask_acc] * f_t[mask_acc], E[mask_acc])
    )

    Q_total = Q + Q_it

    fzero = ps - pd + ns - nd + Ndop_surface * Phi_s / Eth - Q_total ** 2 / (
        2 * elementary_charge * config.eps_Si * Eth
    )
    return fzero


def lookup(Y, yval, X):
    """Return the element of X whose corresponding Y is closest to *yval*."""
    Y = np.array(Y)
    index = np.argmin(abs(Y - yval))
    if len(X) == 0:
        return index
    return X[index]


# ============================================================================
# SURFACE RECOMBINATION
# ============================================================================

def J0sv2_func(X, *params):
    """Energy-resolved surface recombination rate Us and J0s — Eqs. (A2, A6, A7).

    Surface recombination rate (Eq. A6):
        Us = ∫(Ev→Ec) [ps·ns − ni²] / [(ps+p₁)/Sn0 + (ns+n₁)/Sp0] dE

    Energy-dependent surface recombination velocities (Eq. A7):
        Sn0(E) = D_it(E) · σ_n(E) · v_th,n
        Sp0(E) = D_it(E) · σ_p(E) · v_th,p

    SRH reference concentrations (Eq. A2):
        n₁(E) = Nc · exp[−(Ec − E) / (kT)]
        p₁(E) = Nv · exp[−(E − Ev) / (kT)]

    Returns
    -------
    J0s, Uint (= Us), ps, ns, Phi_s, Sn0, Sp0, J0s_err
    """
    ns_in, T_arg, Ndop_emitter, Ndop_bulk_arg, dop_type_emitter, dop_type_bulk_arg, dn, Nit_err = X
    Dit_E = params[0]
    sigma_n = params[1]
    sigma_p = params[2]

    Eth = config.kB * T_arg
    ni_b = ni_func(T_arg, Ndop_bulk_arg, dop_type_bulk_arg)

    Ndop_surface_n = dop_type_emitter * Ndop_emitter + dop_type_bulk_arg * Ndop_bulk_arg
    Ndop_surface_p = (1 - dop_type_emitter) * Ndop_emitter + (1 - dop_type_bulk_arg) * Ndop_bulk_arg
    Ndop_surface = np.abs(Ndop_surface_n - Ndop_surface_p)
    ni_e = ni_func(T_arg, Ndop_surface, dop_type_emitter)

    nd0 = dop_type_bulk_arg * Ndop_bulk_arg + (1 - dop_type_bulk_arg) * ni_b ** 2 / Ndop_bulk_arg
    pd0 = (1 - dop_type_bulk_arg) * Ndop_bulk_arg + dop_type_bulk_arg * ni_b ** 2 / Ndop_bulk_arg

    pd = pd0 + dn
    nd = nd0 + dn

    ps = (ni_e ** 2 / ni_b ** 2) * pd * nd / ns_in

    n = nd
    p = pd

    E = create_energy_array()
    dE = E[1:] - E[:-1]

    cn = sigma_n * config.vth_n
    cp = sigma_p * config.vth_p

    n11 = config.NC * np.exp(-(config.Ec - E) / config.kT)
    p11 = config.NV * np.exp(-(E - config.Ev) / config.kT)

    R_denominator = (ns_in + n11) / cp + (ps + p11) / cn
    R = np.divide(
        Dit_E * (ps * ns_in - ni_e ** 2),
        R_denominator,
        out=np.zeros_like(R_denominator),
        where=R_denominator != 0
    )

    UofE = R
    Uint = np.sum((UofE[1:] + UofE[:-1]) / 2 * dE)

    k_ratio = np.divide(sigma_n, sigma_p, out=np.zeros_like(sigma_n), where=sigma_p != 0)
    alpha_den = (k_ratio * n + p11)
    alpha = np.divide((k_ratio * n11 + p), alpha_den, out=np.zeros_like(alpha_den), where=alpha_den != 0)

    denom = alpha + 1
    Nsplus = np.divide(Dit_E, denom, out=np.zeros_like(Dit_E), where=denom != 0)
    Ns = alpha * Nsplus
    N = Nsplus + Ns

    Sn0_tem = N * sigma_n * config.vth_n
    Sp0_tem = N * sigma_p * config.vth_p

    denom1 = np.divide(1, Sn0_tem, out=np.full_like(Sn0_tem, np.inf), where=Sn0_tem != 0) + \
             np.divide(1, Sp0_tem, out=np.full_like(Sp0_tem, np.inf), where=Sp0_tem != 0)

    J0s_tem = np.divide(elementary_charge * ni_e ** 2, denom1, out=np.zeros_like(denom1), where=denom1 != 0)
    J0s = np.sum((J0s_tem[1:] + J0s_tem[:-1]) / 2 * dE)

    Sn0 = np.sum((Sn0_tem[1:] + Sn0_tem[:-1]) / 2 * dE)
    Sp0 = np.sum((Sp0_tem[1:] + Sp0_tem[:-1]) / 2 * dE)

    Phi_s = -Eth * np.log((ns_in * ni_b) / (nd * ni_e))

    J0s_err = 0.0
    return J0s, Uint, ps, ns_in, Phi_s, Sn0, Sp0, J0s_err


def surfaceLifetime(n0, p0, n, p, Delta_n, Qfixi, T_arg,
                    Ndop_emitter, Ndop_bulk_arg, dop_type_emitter, dop_type_bulk_arg,
                    dn, Dit_tot, sigma_n, sigma_p, return_diagnostics=False):
    """Surface recombination lifetime — two-step ns solve.

    Workflow
    -------
    1. Coarse search — vectorised evaluation of Eq. (A5) without Q_it
       to find a good initial guess for ns.
    2. Fine solve — scalar fsolve of Eqs. (A5) + (2) including
       self-consistent Q_it.
    3. Compute Us via Eq. (A6) at the solved ns, then:
          S = Us / Δn,     τ_surface = W / (2·S)
    """
    # --- Step 1: Coarse search (without Q_it, fast vectorised) ---
    params_ns_coarse = (Qfixi, T_arg, Ndop_emitter, Ndop_bulk_arg,
                        dop_type_emitter, dop_type_bulk_arg, dn)

    ni_b_curr = ni_func(T_arg, Ndop_bulk_arg, dop_type_bulk_arg)
    nd0_curr = dop_type_bulk_arg * Ndop_bulk_arg + (1 - dop_type_bulk_arg) * ni_b_curr ** 2 / Ndop_bulk_arg
    nd_curr = nd0_curr + dn

    if Qfixi < 0:
        ns_search_range = np.logspace(2, np.log10(nd_curr) + 0.1, 10000)
    else:
        ns_search_range = np.logspace(np.log10(nd_curr) - 0.1, 21, 10000)

    fzero_values = ns_zero_func(ns_search_range, *params_ns_coarse)
    fzero_abs = np.abs(fzero_values)
    ns_guess = lookup(fzero_abs, fzero_abs.min(), ns_search_range)

    # --- Step 2: Fine solve with Q_it (scalar, self-consistent) ---
    params_ns_full = (Qfixi, T_arg, Ndop_emitter, Ndop_bulk_arg,
                      dop_type_emitter, dop_type_bulk_arg, dn,
                      Dit_tot, sigma_n, sigma_p)
    ns_solved = fsolve(ns_zero_func_full, ns_guess, args=params_ns_full)[0]
    ns_solved = max(ns_solved, 1e-20)

    X_j0s = (ns_solved, T_arg, Ndop_emitter, Ndop_bulk_arg, dop_type_emitter, dop_type_bulk_arg, dn, 0)
    params_j0s = (Dit_tot, sigma_n, sigma_p)

    J0s_results = J0sv2_func(X_j0s, *params_j0s)
    Uint = J0s_results[1]
    Phi_s = J0s_results[4]

    if Delta_n > 1e-10:
        S = Uint / Delta_n
    else:
        S = 0

    if S > 1e-10:
        tau_surface = config.W / (2 * S)
    else:
        tau_surface = np.inf

    if return_diagnostics:
        return tau_surface, {"ns": ns_solved, "Phi_s_eV": Phi_s}
    return tau_surface


# ============================================================================
# INTRINSIC LIFETIME (Richter model)
# ============================================================================

def intrinsicLifetime(n0, p0, n, p, Delta_n):
    """Intrinsic (Auger + radiative) lifetime using the Richter parametrisation."""
    bmax = 1.0

    bmin = INTR_RMAX + (INTR_RMIN - INTR_RMAX) / (1 + (config.T / INTR_R1) ** INTR_R2)
    b1 = INTR_SMAX + (INTR_SMIN - INTR_SMAX) / (1 + (config.T / INTR_S1) ** INTR_S2)
    b3 = INTR_WMAX + (INTR_WMIN - INTR_WMAX) / (1 + (config.T / INTR_W1) ** INTR_W2)

    Blow = 10 ** (-9.6514 - 8.0525e-2 * config.T + 6.0269e-4 * config.T ** 2 - 2.294e-6 * config.T ** 3
                  + 4.3193e-9 * config.T ** 4 - 3.16154e-12 * config.T ** 5)

    Brel = bmin + (bmax - bmin) / (1 + ((n + p) / b1) ** INTR_B2 + ((n + p) / b3) ** INTR_B4)
    Brad = Brel * Blow

    geeh = 1 + INTR_GEEH_FACTOR * (1 - np.tanh(n0 / INTR_N0EEH) ** INTR_GEEH_EXP)
    gehh = 1 + INTR_GEHH_FACTOR * (1 - np.tanh(p0 / INTR_N0EHH) ** INTR_GEHH_EXP)

    ni_local = ni_func(config.T, config.Ndop_bulk, config.dop_type_bulk)
    Uintr = (n * p - ni_local ** 2) * (
        INTR_AUGER_EEH * geeh * n0 +
        INTR_AUGER_EHH * gehh * p0 +
        INTR_AUGER_XXX_COEFF * Delta_n ** INTR_AUGER_XXX_EXP +
        Brad
    )

    if Uintr > 1e-10:
        tau_intr = Delta_n / Uintr
    else:
        tau_intr = np.inf

    return tau_intr
