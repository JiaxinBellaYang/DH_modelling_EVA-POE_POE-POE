#!/usr/bin/env python3
"""
3-D RMSE landscape — Dit, Qf, J0_rear identifiability diagnostic.

For each DH dataset (before / after), a full 3-D RMSE grid is computed over
(log10_Dit, log10_|Qf|, log10_J0_rear), matching the search bounds used in
fit_lifetime_to_dit_qf_1.py.

Three marginalised 2-D projections are plotted per dataset:
  1. |Qf|  vs Dit       (pointwise minimum over J0_rear axis)
  2. |Qf|  vs J0_rear   (pointwise minimum over Dit axis)
  3. Dit   vs J0_rear   (pointwise minimum over |Qf| axis)

The best-fit star marks the global 3-D minimum projected onto each plane.
Contour lines mark RMSE / RMSE_min = 1.1×, 1.5×, 2×, 5×.

How to read the output
----------------------
  Tight circular basin  → well-identified parameter pair.
  Elongated ridge       → degeneracy; a family of combinations gives similar RMSE.

Output
------
  figures/rmse_landscape_2/landscape_2_combined.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as elementary_charge

# ── path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import (
    SIGMA0_N, A_N, E0_N,
    SIGMA0_P, A_P, E0_P,
    GAUSS_E0, GAUSS_SIGMA,
    T, W, Ndop_bulk, dop_type_bulk,
    Ndop_emitter, dop_type_emitter,
    TAU_N0_BULK, TAU_P0_BULK,
)
from physics import (
    create_energy_array, calculate_gaussian_sigma,
    surfaceLifetime, intrinsicLifetime, ni_func, Dig_func,
    calculate_srh_lifetime_injection_dependent,
    rear_j0_lifetime,
)
from helpers import equilibrium_concentrations, effective_lifetime_terms, setup_plot_style


# ── data loading ──────────────────────────────────────────────────────────────

def load_dh_data(filepath):
    """Load DH lifetime data from whichever sheet contains the actual MCD/lifetime columns."""
    xl = pd.ExcelFile(filepath)
    df = None
    dn_col = tau_col = None

    for sheet_name in xl.sheet_names:
        try:
            candidate = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception:
            continue
        if candidate is None or candidate.empty:
            continue

        cols = list(candidate.columns)
        dn_candidates = [
            col for col in cols
            if 'mcd' in ''.join(ch.lower() for ch in str(col) if ch.isalnum())
        ]
        tau_candidates = [
            col for col in cols
            if 'lifetime' in ''.join(ch.lower() for ch in str(col) if ch.isalnum())
            or 'tau' in ''.join(ch.lower() for ch in str(col) if ch.isalnum())
        ]

        if dn_candidates and tau_candidates:
            dn_col = dn_candidates[0]
            tau_col = tau_candidates[0]
            df = candidate
            break

    if df is None or dn_col is None or tau_col is None:
        for sheet_name in xl.sheet_names:
            candidate = pd.read_excel(filepath, sheet_name=sheet_name)
            raise ValueError(
                f"No DH lifetime columns found in {filepath}. "
                f"Available sheets: {xl.sheet_names}. Columns: {candidate.columns.tolist()}"
            )

    dn = pd.to_numeric(df[dn_col], errors='coerce').to_numpy()
    tau = pd.to_numeric(df[tau_col], errors='coerce').to_numpy()
    mask = (dn > 0) & (tau > 0) & np.isfinite(dn) & np.isfinite(tau)
    return dn[mask], tau[mask]


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


# ── single-point RMSE ─────────────────────────────────────────────────────────

def _bin_weighted_rmse(tau_pred, tau_actual, dn_vals, weights=(1.0, 1.0, 1.0)):
    """Log-RMSE averaged across 3 injection-level bins (identical to fitting code)."""
    log_dn = np.log10(dn_vals)
    lo, hi = log_dn.min(), log_dn.max()
    step = (hi - lo) / 3.0
    errs, wts = [], []
    for i in range(3):
        lo_b = lo + i * step
        hi_b = lo_b + step + (1e-9 if i == 2 else 0)
        m = (log_dn >= lo_b) & (log_dn < hi_b)
        if m.sum() > 1:
            e = np.sqrt(np.mean((np.log(tau_pred[m]) - np.log(tau_actual[m])) ** 2))
            errs.append(e)
            wts.append(weights[i])
    if not errs:
        return 1e6
    return float(np.dot(wts, errs) / np.sum(wts))


def compute_rmse(dn_sorted, tau_sorted, Dit0, Qfix_C, sigma_n, sigma_p,
                 J0_rear=0.0, E0_g=None, gauss_sigma=None,
                 bin_weights=(1.0, 1.0, 1.0),
                 low_inj_anchor=0.0, peak_inj_anchor=0.0, high_inj_anchor=0.0,
                 simple=False):
    """Log-RMSE for one (Dit0, Qfix_C, J0_rear) triplet.

    Parameters match those used in fit_lifetime_to_dit_qf_1.taueff_continuous_with_j0rear.

    simple=True : plain uniform log-RMSE on all points — matches the final
                  RMSE reported by fit_lifetime_to_dit_qf_1.py.
    simple=False: bin-weighted RMSE + anchor penalties (used during optimisation).
    """
    if E0_g is None:
        E0_g = GAUSS_E0
    if gauss_sigma is None:
        gauss_sigma = GAUSS_SIGMA

    ni_b   = ni_func(T, Ndop_bulk, dop_type_bulk)
    E      = create_energy_array()
    Dit_E  = Dig_func(E, E0_g, Dit0, gauss_sigma)
    n0, p0 = equilibrium_concentrations(Ndop_bulk, dop_type_bulk, ni_b)

    tau_pred = []
    ns_prev  = None
    for dn_i in dn_sorted:
        n = n0 + dn_i
        p = p0 + dn_i
        tau_front, diag = surfaceLifetime(
            n0, p0, n, p, dn_i, Qfix_C, T,
            Ndop_emitter, Ndop_bulk,
            dop_type_emitter, dop_type_bulk,
            dn_i, Dit_E, sigma_n, sigma_p,
            return_diagnostics=True,
            ns_init=ns_prev,
        )
        ns_prev  = diag["ns"]
        tau_rear = rear_j0_lifetime(J0_rear, n, p, dn_i, ni_b, W)
        tau_intr = intrinsicLifetime(n0, p0, n, p, dn_i)
        tau_bulk = calculate_srh_lifetime_injection_dependent(
            dn_i, n0, p0, TAU_N0_BULK, TAU_P0_BULK, ni_b)
        tau_pred.append(effective_lifetime_terms(tau_front, tau_rear, tau_intr, tau_bulk))

    tau_pred = np.array(tau_pred)
    ok = np.isfinite(tau_pred) & (tau_pred > 0)
    if ok.sum() < 3:
        return np.inf
    if simple:
        # Plain uniform log-RMSE — matches fit_lifetime_to_dit_qf_1.py's reported RMSE
        return float(np.sqrt(np.mean((np.log(tau_pred[ok]) - np.log(tau_sorted[ok])) ** 2)))
    rmse_val = _bin_weighted_rmse(tau_pred[ok], tau_sorted[ok], dn_sorted[ok],
                                  weights=bin_weights)
    if low_inj_anchor > 0:
        rmse_val += low_inj_anchor * abs(np.log(tau_pred[0] / tau_sorted[0]))
    if peak_inj_anchor > 0:
        idx_peak = np.argmax(tau_sorted)
        rmse_val += peak_inj_anchor * abs(np.log(tau_pred[idx_peak] / tau_sorted[idx_peak]))
    if high_inj_anchor > 0:
        rmse_val += high_inj_anchor * abs(np.log(tau_pred[-1] / tau_sorted[-1]))
    return rmse_val


# ── 3-D grid scan ─────────────────────────────────────────────────────────────

def rmse_landscape_3d(dn_sorted, tau_sorted, sigma_n, sigma_p,
                      ld_range, lq_range, lj_range,
                      n_dit=20, n_qf=20, n_j0rear=10,
                      E0_g=None, gauss_sigma=None,
                      bin_weights=(1.0, 1.0, 1.0),
                      low_inj_anchor=0.0, peak_inj_anchor=0.0, high_inj_anchor=0.0):
    """Scan a 3-D (Dit × Qf × J0rear) grid and return arrays plus the RMSE cube.

    Returns
    -------
    ld_arr  : (n_dit,)      log10(Dit) grid
    lq_arr  : (n_qf,)       log10(|Qf|) grid
    lj_arr  : (n_j0rear,)   log10(J0_rear) grid
    RMSE_3d : (n_dit, n_qf, n_j0rear)
              RMSE_3d[i, j, k] → RMSE at (ld_arr[i], lq_arr[j], lj_arr[k])
    """
    dn_s, tau_s = subsample_log(dn_sorted, tau_sorted, n_pts=25)
    ld_arr = np.linspace(*ld_range,  n_dit)
    lq_arr = np.linspace(*lq_range,  n_qf)
    lj_arr = np.linspace(*lj_range,  n_j0rear)
    RMSE_3d = np.full((n_dit, n_qf, n_j0rear), np.nan)

    total = n_dit * n_qf * n_j0rear
    done  = 0
    for i, ld in enumerate(ld_arr):
        for j, lq in enumerate(lq_arr):
            Dit0   = 10.0 ** ld
            Qfix_C = -(10.0 ** lq) * elementary_charge
            for k, lj in enumerate(lj_arr):
                J0r = 10.0 ** lj
                try:
                    RMSE_3d[i, j, k] = compute_rmse(
                        dn_s, tau_s, Dit0, Qfix_C, sigma_n, sigma_p,
                        J0_rear=J0r, E0_g=E0_g, gauss_sigma=gauss_sigma,
                        bin_weights=bin_weights,
                        low_inj_anchor=low_inj_anchor,
                        peak_inj_anchor=peak_inj_anchor,
                        high_inj_anchor=high_inj_anchor)
                except Exception:
                    RMSE_3d[i, j, k] = np.nan
                done += 1
                if done % 50 == 0 or done == total:
                    pct = 100 * done / total
                    print(f"    [{pct:5.1f}%]  {done}/{total} grid points", end="\r", flush=True)
    print()
    return ld_arr, lq_arr, lj_arr, RMSE_3d


# ── 2-D projection panel ──────────────────────────────────────────────────────

def plot_panel_2d(ax, x_arr, y_arr, RMSE_2d, x_best, y_best, rmse_min,
                  x_label, y_label, title, marker_color, annot_text=""):
    """
    Draw filled-contour heatmap of log10(RMSE / RMSE_min) on *ax*.

    Parameters
    ----------
    x_arr, y_arr : 1-D arrays   axes tick values
    RMSE_2d      : 2-D array    shape (len(y_arr), len(x_arr))
    x_best, y_best : floats     best-fit location (global 3-D minimum projected)
    rmse_min     : float        global 3-D RMSE minimum
    annot_text   : str          multi-line annotation shown in the top-left box
    """
    finite = np.isfinite(RMSE_2d)
    if not finite.any():
        ax.set_title(f"{title}\n(no valid RMSE values)", fontsize=10)
        return None

    ratio     = np.where(finite, RMSE_2d / rmse_min, np.nan)
    log_ratio = np.log10(np.clip(ratio, 1.0, 10.0))

    # Filled contours
    cf = ax.contourf(x_arr, y_arr, log_ratio,
                     levels=60, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    # Overlaid contour lines
    ratio_levels = [1.1, 1.5, 2.0, 5.0]
    log_levels   = [np.log10(r) for r in ratio_levels
                    if np.log10(r) < np.nanmax(log_ratio) - 1e-6]
    if log_levels:
        cs = ax.contour(x_arr, y_arr, log_ratio,
                        levels=log_levels, colors="white",
                        linewidths=1.0, alpha=0.85)
        fmt = {v: f"{10**v:.1f}×" for v in log_levels}
        ax.clabel(cs, fmt=fmt, fontsize=8, inline=True, inline_spacing=4)

    # Best-fit star (projection of global 3-D minimum)
    ax.plot(x_best, y_best,
            "*", color=marker_color, markersize=18,
            markeredgecolor="black", markeredgewidth=1.0,
            zorder=6, label="Best fit (global RMSE min)")

    # Annotation box
    if annot_text:
        ax.text(0.03, 0.97, annot_text,
                transform=ax.transAxes,
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.80, ec="gray"))

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(direction="in", which="both",
                   top=True, right=True, length=4, labelsize=8)
    ax.legend(fontsize=8, frameon=True, loc="lower right")
    return cf


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(script_dir)
    data_dir     = os.path.join(project_root, "data")
    out_dir      = os.path.join(project_root, "figures", "rmse_landscape_1")
    os.makedirs(out_dir, exist_ok=True)

    E       = create_energy_array()
    sigma_n = calculate_gaussian_sigma(E, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E, SIGMA0_P, A_P, E0_P)

    # ── dataset config ────────────────────────────────────────────────────────
    # Bounds must match those used in fit_lifetime_to_dit_qf_1.py.
    # n_dit / n_qf / n_j0rear: grid resolution — increase for finer landscapes.
    GAUSS_SIGMA_AFTER = 0.18   # match fit_lifetime_to_dit_qf_1.py

    datasets = [
        dict(
            label        = "DH 0 hr (Before)",
            path         = os.path.join(data_dir, "1_B_DH0hr.xlsx"),
            dit_range    = (9.0,  12.0),
            qf_range     = (9.0, 13.5),
            j0rear_range = (-15.0, -13.0),   # matches J0REAR_RANGE_BEFORE
            n_dit        = 30,
            n_qf         = 30,
            n_j0rear     = 15,
            e0g          = GAUSS_E0,
            gauss_sigma  = GAUSS_SIGMA,
            color        = "royalblue",
        ),
        dict(
            label        = "DH 1000 hrs (After)",
            path         = os.path.join(data_dir, "1_B_DH1000hrs_new.xlsx"),
            dit_range    = (9.0,  12.0),
            qf_range     = (9.0, 13.5),
            j0rear_range = (-15.0, -13.0),   # matches J0REAR_RANGE_AFTER
            n_dit        = 30,
            n_qf         = 30,
            n_j0rear     = 15,
            e0g          = GAUSS_E0,
            gauss_sigma  = GAUSS_SIGMA_AFTER,
            color        = "crimson",
            bin_weights     = (2.0, 4.0, 2.0),
            low_inj_anchor  = 2.0,
            peak_inj_anchor = 5.0,
            high_inj_anchor = 2.0,
        ),
    ]

    setup_plot_style()

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 3, figsize=(18, 6.0 * n_ds))
    # Ensure axes is always 2-D even with one dataset
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        r"RMSE landscape ($D_{it}$, $Q_f$, $J_{0,rear}$) — 3-D identifiability",
        # "\n"
        # r"Projections via pointwise min · contours = RMSE / RMSE$_{min}$ · ★ = global best fit",
        fontsize=13, fontweight="bold",
    )

    all_cf = []   # collect contourf objects for a shared colorbar

    for row, ds in enumerate(datasets):
        print(f"\n{'='*65}")
        print(f"  {ds['label']}")
        print(f"{'='*65}")

        dn_raw, tau_raw = load_dh_data(ds["path"])
        order  = np.argsort(dn_raw)
        dn_s   = dn_raw[order]
        tau_s  = tau_raw[order]
        print(f"  Loaded {len(dn_s)} points  "
              f"(Δn: {dn_s.min():.1e} – {dn_s.max():.1e} cm⁻³)")

        # ── Grid scan (for landscape display) ────────────────────────────────
        n_dit  = ds["n_dit"]
        n_qf   = ds["n_qf"]
        n_j0r  = ds["n_j0rear"]
        print(f"  Grid: {n_dit}×{n_qf}×{n_j0r} = {n_dit*n_qf*n_j0r} points")
        print(f"    Dit   ∈ {ds['dit_range']}  (log10)")
        print(f"    |Qf|  ∈ {ds['qf_range']}   (log10)")
        print(f"    J0r   ∈ {ds['j0rear_range']}  (log10)")

        ld_arr, lq_arr, lj_arr, RMSE_3d = rmse_landscape_3d(
            dn_s, tau_s, sigma_n, sigma_p,
            ds["dit_range"], ds["qf_range"], ds["j0rear_range"],
            n_dit=n_dit, n_qf=n_qf, n_j0rear=n_j0r,
            E0_g=ds["e0g"], gauss_sigma=ds["gauss_sigma"],
            bin_weights=ds.get("bin_weights", (1.0, 1.0, 1.0)),
            low_inj_anchor=0.0, peak_inj_anchor=0.0, high_inj_anchor=0.0,
        )

        # Grid minimum — used for colormap normalization so the landscape is
        # always visible (not all-red).
        rmse_grid_min = float(np.nanmin(RMSE_3d))
        i_b, j_b, k_b = np.unravel_index(np.nanargmin(RMSE_3d), RMSE_3d.shape)
        ld_best = ld_arr[i_b];  lq_best = lq_arr[j_b];  lj_best = lj_arr[k_b]

        print(f"\n  Grid min:  RMSE = {rmse_grid_min:.2f}")
        print(f"    log10(Dit)   = {ld_best:.3f}  →  {10**ld_best:.3e} cm⁻² eV⁻¹")
        print(f"    log10(|Qf|)  = {lq_best:.3f}  →  {10**lq_best:.3e} cm⁻²")
        print(f"    log10(J0r)   = {lj_best:.3f}  →  {10**lj_best:.3e} A/cm²")

        # Annotation: grid best-fit parameters + simple RMSE on full dataset
        rmse_report = compute_rmse(
            dn_s, tau_s,
            10.0 ** ld_best, -(10.0 ** lq_best) * elementary_charge,
            sigma_n, sigma_p,
            J0_rear=10.0 ** lj_best,
            E0_g=ds["e0g"], gauss_sigma=ds["gauss_sigma"],
            simple=True)
        print(f"  RMSE (simple, full data) = {rmse_report:.2f}")

        annot = (
            f"$D_{{it}}$   = 10$^{{{ld_best:.2f}}}$ cm$^{{-2}}$eV$^{{-1}}$\n"
            f"$|Q_f|$  = 10$^{{{lq_best:.2f}}}$ cm$^{{-2}}$\n"
            f"$J_{{0,r}}$ = 10$^{{{lj_best:.2f}}}$ A cm$^{{-2}}$\n"
            f"RMSE = {rmse_report:.2f}"
        )

        # ── 2-D projections ───────────────────────────────────────────────────
        # RMSE_3d shape: (n_dit, n_qf, n_j0rear)
        #   axis 0 = Dit, axis 1 = Qf, axis 2 = J0rear

        # 1. |Qf| vs Dit  — minimise over J0rear (axis 2)
        #    contourf(ld_arr, lq_arr, Z) needs Z shape (n_qf, n_dit)
        RMSE_ditqf  = np.nanmin(RMSE_3d, axis=2).T   # (n_dit, n_qf) → T → (n_qf, n_dit)

        # 2. |Qf| vs J0rear — minimise over Dit (axis 0)
        #    contourf(lj_arr, lq_arr, Z) needs Z shape (n_qf, n_j0rear)
        RMSE_qfj0r  = np.nanmin(RMSE_3d, axis=0)     # already (n_qf, n_j0rear)

        # 3. Dit vs J0rear — minimise over Qf (axis 1)
        #    contourf(lj_arr, ld_arr, Z) needs Z shape (n_dit, n_j0rear)
        RMSE_ditj0r = np.nanmin(RMSE_3d, axis=1)     # already (n_dit, n_j0rear)

        color = ds["color"]

        # Panel 1: |Qf| vs Dit
        cf = plot_panel_2d(
            axes[row, 0],
            x_arr=ld_arr, y_arr=lq_arr, RMSE_2d=RMSE_ditqf,
            x_best=ld_best, y_best=lq_best, rmse_min=rmse_grid_min,
            x_label=r"$\log_{10}(D_{it,0g})$  [cm$^{-2}$ eV$^{-1}$]",
            y_label=r"$\log_{10}(|Q_f|)$  [cm$^{-2}$]",
            title=f"{ds['label']} — $|Q_f|$ vs $D_{{it}}$\n(min over $J_{{0,rear}}$)",
            marker_color=color,
            annot_text=annot,
        )
        if cf is not None:
            all_cf.append(cf)

        # Panel 2: |Qf| vs J0rear
        cf = plot_panel_2d(
            axes[row, 1],
            x_arr=lj_arr, y_arr=lq_arr, RMSE_2d=RMSE_qfj0r,
            x_best=lj_best, y_best=lq_best, rmse_min=rmse_grid_min,
            x_label=r"$\log_{10}(J_{0,rear})$  [A cm$^{-2}$]",
            y_label=r"$\log_{10}(|Q_f|)$  [cm$^{-2}$]",
            title=f"{ds['label']} — $|Q_f|$ vs $J_{{0,rear}}$\n(min over $D_{{it}}$)",
            marker_color=color,
            annot_text=annot,
        )
        if cf is not None:
            all_cf.append(cf)

        # Panel 3: Dit vs J0rear
        cf = plot_panel_2d(
            axes[row, 2],
            x_arr=lj_arr, y_arr=ld_arr, RMSE_2d=RMSE_ditj0r,
            x_best=lj_best, y_best=ld_best, rmse_min=rmse_grid_min,
            x_label=r"$\log_{10}(J_{0,rear})$  [A cm$^{-2}$]",
            y_label=r"$\log_{10}(D_{it,0g})$  [cm$^{-2}$ eV$^{-1}$]",
            title=f"{ds['label']} — $D_{{it}}$ vs $J_{{0,rear}}$\n(min over $|Q_f|$)",
            marker_color=color,
            annot_text=annot,
        )
        if cf is not None:
            all_cf.append(cf)

    # ── shared colorbar ───────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.07, right=0.87, top=0.88, bottom=0.07,
                        wspace=0.42, hspace=0.55)
    if all_cf:
        cbar_ax = fig.add_axes([0.90, 0.07, 0.016, 0.81])
        cbar = fig.colorbar(all_cf[-1], cax=cbar_ax,
                            label=r"$\log_{10}$(RMSE / RMSE$_{min}$)")
        cbar.ax.tick_params(labelsize=9)
        tick_ratios = [1.0, 1.1, 1.5, 2.0, 5.0, 10.0]
        tick_vals   = [np.log10(r) for r in tick_ratios if np.log10(r) <= 1.0]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f"{r:.1f}×" for r in tick_ratios if np.log10(r) <= 1.0])

    out_path = os.path.join(out_dir, "landscape_1B_combined.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
