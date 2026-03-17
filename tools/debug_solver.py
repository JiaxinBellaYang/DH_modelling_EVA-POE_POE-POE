#!/usr/bin/env python3
"""
Debug the surface potential solver (ns_zero_func) behaviour.

Visualises how the charge-balance residual varies with ns over different
search ranges, and tests whether fsolve converges to a correct root.

Output: figures/solver_debug.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.constants import e as elementary_charge

# --- path setup: add src/ so config, physics, helpers are importable ---
tools_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tools_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from physics import ns_zero_func, lookup


# ============================================================================
# DEBUGGING SCENARIO
# ============================================================================

T = 300.0
Ndop_bulk = 1.5e15
dop_type_bulk = 1.0
dop_type_emitter = 1.0
Ndop_emitter = 1e19
dn = 1e15
Qfix = 1.5e13
Qfix_charge = -Qfix * elementary_charge

params_ns_zero = (Qfix_charge, T, Ndop_emitter, Ndop_bulk,
                  dop_type_emitter, dop_type_bulk, dn)

# Test search ranges
ranges = [
    ("Old Range (16–20)",            np.logspace(16, 20, 10000)),
    ("New Range (2–21)",             np.logspace(2, 21, 10000)),
    ("Dense Wide Range (2–21, 100k)", np.logspace(2, 21, 100000)),
]

print(f"Debugging solver for Ndop_emitter={Ndop_emitter:.1e}, Qfix={Qfix:.1e}")

for name, ns_search_range in ranges:
    print(f"\n--- {name} ---")
    fzero_values = ns_zero_func(ns_search_range, *params_ns_zero)
    fzero_abs = np.abs(fzero_values)
    ns_guess = lookup(fzero_abs, fzero_abs.min(), ns_search_range)
    print(f"Guess ns: {ns_guess:.4e},  f(guess): {ns_zero_func(np.array([ns_guess]), *params_ns_zero)[0]:.4e}")
    try:
        ns_solved = fsolve(ns_zero_func, ns_guess, args=params_ns_zero)[0]
        print(f"Solved ns: {ns_solved:.4e},  f(solved): {ns_zero_func(np.array([ns_solved]), *params_ns_zero)[0]:.4e}")
    except Exception as e:
        print(f"Solver failed: {e}")

# --- Residual plot ---
ns_plot = np.logspace(15, 20, 1000)
y_plot = ns_zero_func(ns_plot, *params_ns_zero)

plt.figure()
plt.semilogx(ns_plot, y_plot)
plt.xlabel('ns')
plt.ylabel('f(ns)')
plt.title(f'Solver residual  (Q = {Qfix:.1e})')
plt.grid(True)
plt.ylim(-1e15, 1e15)
plt.axhline(0, color='red')

figures_dir = os.path.join(project_root, 'figures')
os.makedirs(figures_dir, exist_ok=True)
out = os.path.join(figures_dir, 'solver_debug.png')
plt.savefig(out)
print(f"\nSaved {out}")
