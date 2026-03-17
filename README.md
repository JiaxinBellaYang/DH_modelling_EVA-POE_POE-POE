# Surface Recombination Modelling

Simulates minority carrier lifetime in silicon wafers using an energy-resolved
Shockley–Read–Hall (SRH) model with self-consistent surface potential solving,
amphoteric interface defects, and the Richter intrinsic lifetime model.

## Project Structure

```
.
├── src/
│   ├── config.py           — All shared physics constants (single source of truth)
│   ├── physics.py          — Pure physics functions (surface, intrinsic, bulk SRH)
│   ├── helpers.py          — Shared utilities (equilibrium conc., Matthiessen, data loading)
│   ├── fitting.py          — Model fitting and error metrics
│   ├── figure_panels.py    — Figures 1, 2, 3: parameter sweep panels
│   ├── generate_figures.py — Entry point: generates all figures and validates
│   ├── model_dit.py        — Standalone Dit modelling script (0, flat, 1, 2, 3 peaks)
│   ├── simulate_uv.py      — Before/after UV comparison with experimental data
│   ├── simulate_q_sweep.py — Sweep of negative fixed-charge magnitude
│   ├── simulate_doping.py  — Undiffused vs diffused surface comparison
│   ├── simulate_impact_doping.py — Delta J0 vs surface doping using 3-peak params
│   ├── generate_summary.py — Aggregates Dit and Lifetime metrics into a single table
│   └── run_pipeline.py     — End-to-end execution script
├── tools/
│   ├── test_polarity.py    — Compare positive vs negative Qfix
│   └── debug_solver.py     — Visualise charge-balance solver residual
├── data/
│   ├── W3_ini.xlsx                        — Experimental data (before UV)
│   ├── W3_UV20h.xlsx                      — Experimental data (after 20 h UV)
│   ├── Wafer_Dit_Modelling_Initial_1.xlsx — Dit data (initial state)
│   └── Wafer_Dit_Modelling_UV20h_1.xlsx   — Dit data (after 20 h UV)
├── figures/                — Generated output figures (gitignored)
├── results/                — Fit tables and numerical outputs (gitignored)
├── reference/              — Reference figures for regression validation
├── validation/
│   └── validate_results.py — Compare generated figures against references
├── archive/                — Old notebooks and reference PDFs
├── requirements.txt
└── tan2021_equations_appendix.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

**Full End-to-End Pipeline (Recommended):**

This script executes the complete workflow:
1. Reads the experimental Excel data and fits 5 different $D_{it}$ models (0-peak, flat, 1-peak, 2-peak, 3-peak).
2. Runs the minority carrier lifetime simulations for each configuration, extracting effective lifetime and Kane-Swanson $J_0$.
3. Aggregates all model parameters and goodness-of-fit metrics into a final `master_model_comparison_summary.xlsx`.

```bash
python3 src/run_pipeline.py
```

---

### Individual Scripts

Generate all standard parameter-sweep and validation figures:

```bash
python3 src/generate_figures.py
```

UV exposure comparison (simulated lifetime curves overlaying experimental data):

```bash
python3 src/simulate_uv.py
```

Fixed-charge magnitude sweep:

```bash
python3 src/simulate_q_sweep.py
```

Undiffused vs diffused surface comparison:

```bash
python3 src/simulate_doping.py
```

Delta J0 simulation over varying surface dopings using 3-peak parameters:

```bash
python3 src/simulate_impact_doping.py
```

Model fitting diagnostics:

```bash
python3 src/fitting.py
```

## Physics Model

The surface recombination model uses the extended SRH formalism for a
continuum of interface states. The full set of equations is documented in
[`tan2021_equations_appendix.md`](tan2021_equations_appendix.md) and
referenced inline in the source code docstrings.

### Key equations

| Eq. | Description | Implemented in |
|-----|-------------|----------------|
| A2  | SRH reference concentrations n₁(E), p₁(E) | `J0sv2_func`, `ns_zero_func_full` |
| A4  | Surface carrier concentrations ns, ps from ψs | `ns_zero_func`, `ns_zero_func_full` |
| A5  | Surface charge balance (Gauss's law) | `ns_zero_func`, `ns_zero_func_full` |
| A6  | Energy-resolved surface recombination rate Us | `J0sv2_func` |
| A7  | Energy-dependent Sn0(E), Sp0(E) | `J0sv2_func` |
| A8–A9 | Band-edge Dit tails (Gaussian approximation) | `Dig_func` |
| A10 | Gaussian midgap Dit peak | `Dig_func` |
| A11 | Total Dit = Dit,v + Dit,g + Dit,c | `simulate_uv.py`, `helpers.build_dit_profile` |
| A12 | Gaussian capture cross sections σn(E), σp(E) | `calculate_gaussian_sigma` |
| 2   | Interface trapped charge Q_it (amphoteric) | `ns_zero_func_full` |

### Implementation notes

- **Eqs. A8, A9** (band-edge tails): The equations define exponential tails;
  the code uses Gaussian approximations centred at Ev and Ec via `Dig_func`.
- **Eq. A12** (capture cross sections): Code uses `A = 1/(2μ²)` parametrisation.
- **Eq. A5** (charge balance): `ns_zero_func_full` extends this with Q_it so
  that `Q → Q_f + Q_it`.
- **Eq. A5 erratum**: The published form contains a typo (`−ps` instead of `−nd`);
  the code implements the correct four-term version.

## Module Dependency Graph

```
config.py
    └── physics.py
            ├── helpers.py
            │       ├── fitting.py
            │       ├── figure_panels.py  →  generate_figures.py
            │       ├── simulate_uv.py
            │       ├── simulate_q_sweep.py
            │       └── simulate_doping.py
            └── tools/
                    ├── test_polarity.py
                    └── debug_solver.py
```
