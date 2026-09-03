#!/usr/bin/env python3
"""
DH 1000 hrs comparison — samples 1A, 1B, 2B.

Loads best-fit parameters from each sample's Excel results file and plots on
a single τ_eff vs Δn figure:
  · Experimental data (filled circles)
  · Total τ_eff model (solid line)
  · τ_front / J0,front component (dashed line)

Output:
  figures/fit_dit_qf/compare_DH1000hrs_1A_1B_2B.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as elementary_charge

# --- path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Re-use simulation functions from the 1A fit script (identical in 1A / 1B / 2B)
from fit_lifetime_to_dit_qf_1A import compute_lifetime_components


def load_data(filepath):
    """Load DH lifetime data.

    Tries Sheet1 first; if it lacks the expected columns falls back to Sheet2.
    """
    required = {"MCD_(cm-3)", "Effective_Lifetime_(s)"}
    for sheet in ("Sheet1", "Sheet2"):
        try:
            df = pd.read_excel(filepath, sheet_name=sheet)
            if required.issubset(df.columns):
                break
        except ValueError:
            continue
    else:
        raise RuntimeError(
            f"Could not find sheet with columns {required} in {filepath}")
    dn  = df["MCD_(cm-3)"].values
    tau = df["Effective_Lifetime_(s)"].values
    mask = (dn > 0) & (tau > 0) & np.isfinite(dn) & np.isfinite(tau)
    return dn[mask], tau[mask]

from config import SIGMA0_N, A_N, E0_N, SIGMA0_P, A_P, E0_P, lw
from physics import create_energy_array, calculate_gaussian_sigma
from helpers import setup_plot_style


# ============================================================================
# SAMPLE DEFINITIONS
# Change data_file / results_file here if filenames ever differ.
# Each entry also carries the colour scheme for that sample:
#   marker_color → experimental data markers
#   line_color   → total τ_eff model line (solid)
#   front_color  → τ_front / J0,front component (dashed)
# ============================================================================
SAMPLES = [
    dict(
        name="EVA/EVA",
        data_file="1_A_DH1000hrs.xlsx",
        results_file="fit_dit_qf_results_1A.xlsx",
        marker_color="navy",
        line_color="cornflowerblue",
        front_color="lightskyblue",
    ),
    dict(
        name="EVA/POE",
        data_file="1_B_DH1000hrs_new.xlsx",
        results_file="fit_dit_qf_results_1B.xlsx",
        marker_color="darkred",
        line_color="tomato",
        front_color="lightsalmon",
    ),
    dict(
        name="POE/POE",
        data_file="2_B_DH1000hrs_new.xlsx",
        results_file="fit_dit_qf_results_2B.xlsx",
        marker_color="darkgreen",
        line_color="mediumseagreen",
        front_color="lightgreen",
    ),
]


def main():
    project_root = os.path.dirname(script_dir)
    data_dir    = os.path.join(project_root, "data")
    figures_dir = os.path.join(project_root, "figures", "fit_dit_qf")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(figures_dir, exist_ok=True)

    # Shared Gaussian capture cross sections (same for all samples)
    E       = create_energy_array()
    sigma_n = calculate_gaussian_sigma(E, SIGMA0_N, A_N, E0_N)
    sigma_p = calculate_gaussian_sigma(E, SIGMA0_P, A_P, E0_P)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    for s in SAMPLES:
        name = s["name"]
        print(f"\n{'='*55}")
        print(f"  Sample {name}  -  DH 1000 hrs")
        print(f"{'='*55}")

        # ------------------------------------------------------------------
        # Load best-fit parameters (filter for the DH 1000 hrs row)
        # ------------------------------------------------------------------
        df_res = pd.read_excel(os.path.join(results_dir, s["results_file"]))
        row    = df_res[df_res["Dataset"].str.contains("1000")].iloc[0]

        Dit_0g = float(row["Dit_0g (cm-2 eV-1)"])
        Qf_cm2 = float(row["Qf (cm-2)"])           # negative (−|Qf|)
        j0rear = float(row["J0_rear (A/cm2)"])
        E0_g   = float(row["E0_g (eV)"])
        gs     = float(row["GAUSS_SIGMA (eV)"])
        Qfix_C = Qf_cm2 * elementary_charge         # [C/cm²], negative

        print(f"  Dit_0g      = {Dit_0g:.3e} cm^-2 eV^-1")
        print(f"  Qf          = {Qf_cm2:.3e} cm^-2")
        print(f"  J0_rear     = {j0rear:.3e} A/cm^2  (log10 = {np.log10(j0rear):.2f})")
        print(f"  E0_g        = {E0_g:.4f} eV")
        print(f"  GAUSS_SIGMA = {gs:.4f} eV")

        # ------------------------------------------------------------------
        # Load experimental data and sort ascending by dn
        # ------------------------------------------------------------------
        dn_raw, tau_raw = load_data(os.path.join(data_dir, s["data_file"]))
        order  = np.argsort(dn_raw)
        dn_s, tau_s = dn_raw[order], tau_raw[order]
        print(f"  Data points = {len(dn_s)}  "
              f"(dn: {dn_s.min():.2e} - {dn_s.max():.2e} cm^-3)")

        # ------------------------------------------------------------------
        # Compute lifetime components using saved fit parameters
        # ------------------------------------------------------------------
        comps = compute_lifetime_components(
            dn_s, Dit_0g, Qfix_C, sigma_n, sigma_p,
            J0_rear=j0rear, E0_g=E0_g, gauss_sigma=gs)

        # ------------------------------------------------------------------
        # Plot
        # Legend entries are added in order: exp → τ_eff → τ_front
        # so that ncol=3 arranges them as 3 tidy rows (one per sample).
        # ------------------------------------------------------------------

        # Experimental data markers
        ax.loglog(dn_s, tau_s * 1e3, "o",
                  color=s["marker_color"], markersize=10, alpha=0.75,
                  label=f"Exp — {name}")

        # Total effective lifetime model (solid line)
        ax.loglog(dn_s, comps["tau_eff"] * 1e3, "-",
                  color=s["line_color"], linewidth=lw,
                  label=r"$\tau_{eff}$ model" + f" — {name}")

        # τ_front (J0,front component): dashed line, mask out unphysical values
        tf   = comps["tau_front"]
        mask = np.isfinite(tf) & (tf > 0) & (tf < 1e2)   # keep τ < 100 s
        if mask.any():
            ax.loglog(dn_s[mask], tf[mask] * 1e3, "--",
                      color=s["front_color"], linewidth=lw * 0.85,
                      label=r"$\tau_{front}$ ($J_{0,\mathrm{front}}$)" + f" — {name}")

    # -----------------------------------------------------------------------
    # Figure formatting
    # -----------------------------------------------------------------------
    ax.set_xlabel(r"Minority carrier density $\Delta n$ (cm$^{-3}$)",
                  fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Effective lifetime $\tau_{eff}$ (ms)",
                  fontsize=18, fontweight="bold")

    # Legend: 3 columns → each row corresponds to one sample (exp / model / τ_front)
    ax.legend(frameon=False, fontsize=10.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.15))

    ax.tick_params(axis="both", which="major", direction="in",
                   length=8, labelsize=14, width=1.5, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in",
                   length=4, width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    
    ax.set_ylim(1.5e-1, 5)

    # Leave room at the bottom for the 3-row legend
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig_path = os.path.join(figures_dir, "compare_DH1000hrs_1A_1B_2B.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved -> {fig_path}")


if __name__ == "__main__":
    main()
