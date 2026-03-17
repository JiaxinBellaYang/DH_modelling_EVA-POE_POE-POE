"""
model_dit.py

Fits a 1-, 2-, and 3-component Gaussian midgap state (plus band tails) 
to measured interface trap density data stored in Excel files.
Exports:
  - Figures for each model in a respective subfolder
  - A summary Excel workbook for each model

Usage
-----
    python model_dit.py [--data-dir <path>] [--figures-dir <path>]
                        [--results-dir <path>] [--glob <pattern>] [--no-show]

Defaults
--------
    --data-dir     data/
    --figures-dir  figures/dit_fits/
    --results-dir  results/
    --glob         Wafer_Dit_Modelling_*.xlsx
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Physics models
# ---------------------------------------------------------------------------

def dit_model_0peak(E, Dit0_v, Ev_trap, Dit0_c, Ec_trap, Ev, Ec):
    Dit_v = Dit0_v * np.exp(-(E - Ev) / Ev_trap)
    Dit_c = Dit0_c * np.exp(-(Ec - E) / Ec_trap)
    return np.log10(Dit_v + Dit_c)

def dit_model_flat(E, Dit0_v, Ev_trap, Dit0_c, Ec_trap, Dit0_const, Ev, Ec):
    Dit_v = Dit0_v * np.exp(-(E - Ev) / Ev_trap)
    Dit_c = Dit0_c * np.exp(-(Ec - E) / Ec_trap)
    return np.log10(Dit_v + Dit_c + Dit0_const)

def dit_model_1peak(E, Dit0_v, Ev_trap, Dit0_c, Ec_trap, Dit0_g, E0, sigma, Ev, Ec):
    Dit_v = Dit0_v * np.exp(-(E - Ev) / Ev_trap)
    Dit_c = Dit0_c * np.exp(-(Ec - E) / Ec_trap)
    Dit_g = Dit0_g * np.exp(-((E - E0) ** 2) / (2 * sigma ** 2))
    return np.log10(Dit_v + Dit_c + Dit_g)

def dit_model_2peaks(E, Dit0_v, Ev_trap, Dit0_c, Ec_trap, 
                     Dit0_g1, E0_1, sigma_1, 
                     Dit0_g2, E0_2, sigma_2, Ev, Ec):
    Dit_v = Dit0_v * np.exp(-(E - Ev) / Ev_trap)
    Dit_c = Dit0_c * np.exp(-(Ec - E) / Ec_trap)
    Dit_g1 = Dit0_g1 * np.exp(-((E - E0_1) ** 2) / (2 * sigma_1 ** 2))
    Dit_g2 = Dit0_g2 * np.exp(-((E - E0_2) ** 2) / (2 * sigma_2 ** 2))
    return np.log10(Dit_v + Dit_c + Dit_g1 + Dit_g2)

def dit_model_3peaks(E, Dit0_v, Ev_trap, Dit0_c, Ec_trap, 
                     Dit0_g1, E0_1, sigma_1, 
                     Dit0_g2, E0_2, sigma_2, 
                     Dit0_g3, E0_3, sigma_3, Ev, Ec):
    Dit_v = Dit0_v * np.exp(-(E - Ev) / Ev_trap)
    Dit_c = Dit0_c * np.exp(-(Ec - E) / Ec_trap)
    Dit_g1 = Dit0_g1 * np.exp(-((E - E0_1) ** 2) / (2 * sigma_1 ** 2))
    Dit_g2 = Dit0_g2 * np.exp(-((E - E0_2) ** 2) / (2 * sigma_2 ** 2))
    Dit_g3 = Dit0_g3 * np.exp(-((E - E0_3) ** 2) / (2 * sigma_3 ** 2))
    return np.log10(Dit_v + Dit_c + Dit_g1 + Dit_g2 + Dit_g3)


# ---------------------------------------------------------------------------
# Fitting parameter definitions
# ---------------------------------------------------------------------------

PARAM_NAMES_0 = ["Dit0_v", "Ev_trap", "Dit0_c", "Ec_trap"]
BOUNDS_0 = (
    [1e10, 0.005, 1e10, 0.005],
    [5e15, 0.1,   5e15, 0.1]
)
P0_0 = [1e13, 0.04, 1e13, 0.04]

PARAM_NAMES_FLAT = ["Dit0_v", "Ev_trap", "Dit0_c", "Ec_trap", "Dit0_const"]
BOUNDS_FLAT = (
    [1e10, 0.005, 1e10, 0.005, 1e10],
    [5e15, 0.1,   5e15, 0.1,   5e13]
)
P0_FLAT = [1e13, 0.04, 1e13, 0.04, 1e12]

PARAM_NAMES_1 = ["Dit0_v", "Ev_trap", "Dit0_c", "Ec_trap", "Dit0_g", "E0", "sigma"]
BOUNDS_1 = (
    [1e10, 0.005, 1e10, 0.005, 1e10, 0.4, 0.1],
    [5e15, 0.1,   5e15, 0.1,   5e13, 0.8, 0.4]
)
P0_1 = [1e13, 0.04, 1e13, 0.04, 1e12, 0.55, 0.2]

PARAM_NAMES_2 = ["Dit0_v", "Ev_trap", "Dit0_c", "Ec_trap", 
                 "Dit0_g1", "E0_1", "sigma_1",
                 "Dit0_g2", "E0_2", "sigma_2"]
BOUNDS_2 = (
    [1e10, 0.005, 1e10, 0.005, 1e10, 0.2, 0.05, 1e10, 0.6, 0.05],
    [5e15, 0.1,   5e15, 0.1,   5e13, 0.5, 0.3,  5e13, 0.9, 0.3]
)
P0_2 = [1e13, 0.04, 1e13, 0.04, 1e12, 0.35, 0.1, 1e12, 0.75, 0.1]

PARAM_NAMES_3 = ["Dit0_v", "Ev_trap", "Dit0_c", "Ec_trap", 
                 "Dit0_g1", "E0_1", "sigma_1",
                 "Dit0_g2", "E0_2", "sigma_2",
                 "Dit0_g3", "E0_3", "sigma_3"]
BOUNDS_3 = (
    [1e10, 0.005, 1e10, 0.005, 1e10, 0.2, 0.05, 1e10, 0.4, 0.05, 1e10, 0.7, 0.05],
    [5e15, 0.1,   5e15, 0.1,   5e13, 0.4, 0.3,  5e13, 0.7, 0.3,  5e13, 0.9, 0.3]
)
P0_3 = [1e13, 0.04, 1e13, 0.04, 1e12, 0.3, 0.1, 1e12, 0.55, 0.1, 1e12, 0.8, 0.1]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def fit_and_plot(num_peaks, E_data, Dit_exp, Ev, Ec, label, out_prefix):
    """Fits the dataset with the specified number of peaks, saves figure."""
    log_Dit_exp = np.log10(Dit_exp)

    if num_peaks == "0":
        func = lambda E, *args: dit_model_0peak(E, *args, Ev, Ec)
        p0, bounds, param_names = P0_0, BOUNDS_0, PARAM_NAMES_0
    elif num_peaks == "flat":
        func = lambda E, *args: dit_model_flat(E, *args, Ev, Ec)
        p0, bounds, param_names = P0_FLAT, BOUNDS_FLAT, PARAM_NAMES_FLAT
    elif num_peaks == 1:
        func = lambda E, *args: dit_model_1peak(E, *args, Ev, Ec)
        p0, bounds, param_names = P0_1, BOUNDS_1, PARAM_NAMES_1
    elif num_peaks == 2:
        func = lambda E, *args: dit_model_2peaks(E, *args, Ev, Ec)
        p0, bounds, param_names = P0_2, BOUNDS_2, PARAM_NAMES_2
    else:
        func = lambda E, *args: dit_model_3peaks(E, *args, Ev, Ec)
        p0, bounds, param_names = P0_3, BOUNDS_3, PARAM_NAMES_3

    popt, _ = curve_fit(func, E_data, log_Dit_exp, p0=p0, bounds=bounds, maxfev=150000)

    # Calculate Goodness-of-Fit
    log_Dit_fit = func(E_data, *popt)
    ssr = np.sum((log_Dit_fit - log_Dit_exp)**2)
    sst = np.sum((log_Dit_exp - np.mean(log_Dit_exp))**2)
    r2 = 1 - (ssr / sst) if sst > 0 else np.nan

    # We append R2 and SSR so they get exported automatically
    param_names = list(param_names) + ["R2", "SSR"]
    popt = list(popt) + [r2, ssr]

    # Plotting
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False

    E_dense = np.linspace(Ev, Ec, 1000)
    Dit_v = popt[0] * np.exp(-(E_dense - Ev) / popt[1])
    Dit_c = popt[2] * np.exp(-(Ec - E_dense) / popt[3])
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(
        E_data, Dit_exp,
        "o", linestyle="None", markerfacecolor="none",
        markeredgecolor="k", markeredgewidth=3, markersize=12,
        label=f"Measured $D_{{it}}$ – {label}",
    )
    ax.semilogy(E_dense, Dit_v, "r-", linewidth=3, label=r"$D_{it,v}$")
    ax.semilogy(E_dense, Dit_c, "b-", linewidth=3, label=r"$D_{it,c}$")

    if num_peaks == "0":
        Dit_total = Dit_v + Dit_c
    elif num_peaks == "flat":
        Dit_total = Dit_v + Dit_c + popt[4]
        ax.semilogy(E_dense, np.full_like(E_dense, popt[4]), color="orange", linewidth=3, label=r"$D_{it,const}$")
    elif num_peaks == 1:
        Dit_g = popt[4] * np.exp(-((E_dense - popt[5]) ** 2) / (2 * popt[6] ** 2))
        Dit_total = Dit_v + Dit_c + Dit_g
        ax.semilogy(E_dense, Dit_g, color="orange", linewidth=3, label=r"$D_{it,g}$")
    elif num_peaks == 2:
        Dit_g1 = popt[4] * np.exp(-((E_dense - popt[5]) ** 2) / (2 * popt[6] ** 2))
        Dit_g2 = popt[7] * np.exp(-((E_dense - popt[8]) ** 2) / (2 * popt[9] ** 2))
        Dit_total = Dit_v + Dit_c + Dit_g1 + Dit_g2
        ax.semilogy(E_dense, Dit_g1, color="orange", linestyle="--", linewidth=3, label=r"$D_{it,g1}$")
        ax.semilogy(E_dense, Dit_g2, color="xkcd:yellow orange", linestyle="--", linewidth=3, label=r"$D_{it,g2}$")
    else:
        Dit_g1 = popt[4] * np.exp(-((E_dense - popt[5]) ** 2) / (2 * popt[6] ** 2))
        Dit_g2 = popt[7] * np.exp(-((E_dense - popt[8]) ** 2) / (2 * popt[9] ** 2))
        Dit_g3 = popt[10] * np.exp(-((E_dense - popt[11]) ** 2) / (2 * popt[12] ** 2))
        Dit_total = Dit_v + Dit_c + Dit_g1 + Dit_g2 + Dit_g3
        ax.semilogy(E_dense, Dit_g1, color="orange", linestyle="--", linewidth=3, label=r"$D_{it,g1}$")
        ax.semilogy(E_dense, Dit_g2, color="xkcd:yellow orange", linestyle="--", linewidth=3, label=r"$D_{it,g2}$")
        ax.semilogy(E_dense, Dit_g3, color="xkcd:goldenrod", linestyle="--", linewidth=3, label=r"$D_{it,g3}$")

    ax.semilogy(E_dense, Dit_total, "g-", linewidth=3, label=r"$D_{it,tot}$")

    ax.set_xlabel("E (eV)", fontweight="bold", fontsize=18)
    ax.set_ylabel(r"$D_{it}$ (eV$^{-1}$·cm$^{-2}$)", fontweight="bold", fontsize=18)
    ax.legend(loc="upper left", frameon=False, prop={"family": "Arial", "weight": "bold", "size": 13})
    ax.tick_params(axis="both", which="major", labelsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.grid(False)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(1e11, 1e15)

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)

    return popt, param_names


def process_file(xlsx_path, figures_dir, results_dir):
    """Load, fit (0, flat, 1, 2, 3 peaks), plot, and return rows for summary."""
    basename = os.path.splitext(os.path.basename(xlsx_path))[0]
    print(f"\nProcessing: {basename}")

    df = pd.read_excel(xlsx_path)
    E_col = df.columns[0]
    dit_col = df.columns[1]

    E_data  = df[E_col].values.astype(float)
    Dit_exp = df[dit_col].values.astype(float)
    Ev = float(np.min(E_data))
    Ec = float(np.max(E_data))

    rows = {"0": None, "flat": None, 1: None, 2: None, 3: None}

    for num_peaks in ["0", "flat", 1, 2, 3]:
        # Subdirectories
        fig_sub = os.path.join(figures_dir, f"{num_peaks}peak")
        os.makedirs(fig_sub, exist_ok=True)
        out_prefix = os.path.join(fig_sub, f"dit_fit_{basename}")

        try:
            popt, param_names = fit_and_plot(num_peaks, E_data, Dit_exp, Ev, Ec, label=dit_col, out_prefix=out_prefix)
            
            row = {"File": basename, "Dataset": dit_col}
            row.update({name: val for name, val in zip(param_names, popt)})
            rows[num_peaks] = row
            print(f"  {num_peaks}-peak fit successful")
            
        except Exception as exc:
            print(f"  ERROR processing {num_peaks}-peak fit for {basename}: {exc}", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Model Dit from Excel files.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_script_dir, "..", "data"),
        help="Directory to search for Excel files (default: data/).",
    )
    parser.add_argument(
        "--figures-dir",
        default=os.path.join(_script_dir, "..", "figures", "dit_fits"),
        help="Directory for output figures PNG/PDF (default: figures/dit_fits/).",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_script_dir, "..", "results"),
        help="Directory for summary Excel workbook (default: results/).",
    )
    parser.add_argument(
        "--glob", default="Wafer_Dit_Modelling_*.xlsx",
        help="Glob pattern for Excel files (default: Wafer_Dit_Modelling_*.xlsx).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display figures interactively (only save to disk).",
    )
    args = parser.parse_args()

    data_dir    = os.path.abspath(args.data_dir)
    figures_dir = os.path.abspath(args.figures_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pattern = os.path.join(data_dir, args.glob)
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s) matching '{args.glob}'")

    rows_0peak = []
    rows_flat = []
    rows_1peak = []
    rows_2peaks = []
    rows_3peaks = []
    
    for xlsx_path in files:
        file_rows = process_file(xlsx_path, figures_dir, results_dir)
        if file_rows["0"]:    rows_0peak.append(file_rows["0"])
        if file_rows["flat"]: rows_flat.append(file_rows["flat"])
        if file_rows[1]:      rows_1peak.append(file_rows[1])
        if file_rows[2]:      rows_2peaks.append(file_rows[2])
        if file_rows[3]:      rows_3peaks.append(file_rows[3])

    # Write summary Excel workbooks into subdirectories
    modes = ["0", "flat", 1, 2, 3]
    dataset_rows = [rows_0peak, rows_flat, rows_1peak, rows_2peaks, rows_3peaks]
    
    for num_peaks, rows in zip(modes, dataset_rows):
        if rows:
            res_sub = os.path.join(results_dir, f"{num_peaks}peak")
            os.makedirs(res_sub, exist_ok=True)
            summary_df = pd.DataFrame(rows)
            summary_path = os.path.join(res_sub, "dit_fit_summary.xlsx")
            summary_df.to_excel(summary_path, index=False)
            print(f"\n{num_peaks}-peak summary saved: {summary_path}")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()

