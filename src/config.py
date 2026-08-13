"""
Single source of truth for all physics constants and simulation parameters.

Every module in this project should import shared constants from here
rather than defining its own copy.
"""

import numpy as np
from scipy.constants import k as boltzmann_k, e as elementary_charge, epsilon_0

# ============================================================================
# TEMPERATURE AND BAND STRUCTURE
# ============================================================================
T = 300.0           # Temperature (K)
Ec = 1.12           # Conduction band energy (eV)
Ev = 0.0            # Valence band energy (eV)
Eg = Ec - Ev        # Band gap (eV)
kB = boltzmann_k / elementary_charge  # Boltzmann constant (eV/K)
kT = kB * T         # Thermal energy (eV)

# ============================================================================
# EFFECTIVE DENSITY OF STATES
# ============================================================================
NC = 2.86e19        # Conduction band (cm^-3)
NV = 3.11e19        # Valence band (cm^-3)

# ============================================================================
# DIELECTRIC PROPERTIES
# ============================================================================
eps_rel = 11.68
eps_Si = eps_rel * epsilon_0 * 1e-2   # F/cm

# ============================================================================
# THERMAL VELOCITIES
# ============================================================================
vth_n = 2.046e7 * (T / 300.0) ** 0.5  # cm/s
vth_p = 1.688e7 * (T / 300.0) ** 0.5  # cm/s

# ============================================================================
# SAMPLE PARAMETERS
# ============================================================================
W = 0.014              # Wafer thickness (cm)
Ndop_bulk = 1.55e15    # Bulk doping concentration (cm^-3) 
Ndop_emitter = 1.0e19     # Emitter doping (cm^-3)
dop_type_bulk = 1.0    # 1 = n-type, 0 = p-type
dop_type_emitter = 0.0

# ============================================================================
# GAUSSIAN CAPTURE CROSS SECTION PARAMETERS (Comparison Set)
# ============================================================================
# Keep these as default/shared or legacy usage
SIGMA0_N = 2.5e-14   # Electron prefactor (cm^2)
A_N = 80.0            # Electron Gaussian width (eV^-2)
E0_N = 0.02           # Electron peak relative to midgap (eV)

SIGMA0_P = 2.5e-16   # Hole prefactor (cm^2)
A_P = 110.0           # Hole Gaussian width (eV^-2)
E0_P = -0.17          # Hole peak relative to midgap (eV)

# --- Before UV ---
# SIGMA0_N_BEFORE = 1e-15
# A_N_BEFORE = 80.0
# E0_N_BEFORE = 0.09    # Adjusted to match Schmidt (~0.65eV - 0.56eV)

# SIGMA0_P_BEFORE = 4e-17
# A_P_BEFORE = 110.0
# E0_P_BEFORE = -0.16   # Adjusted to match Schmidt (~0.40eV - 0.56eV)

# --- After UV ---
# SIGMA0_N_AFTER = 5e-15
# A_N_AFTER = 80.0
# E0_N_AFTER = 0.09

# SIGMA0_P_AFTER = 4e-16
# A_P_AFTER = 110.0
# E0_P_AFTER = -0.16

# ============================================================================
# GAUSSIAN Dit DISTRIBUTION PARAMETERS
# ============================================================================
GAUSS_E0 = 0.56       # Centre of Gaussian Dit (eV)
GAUSS_SIGMA = 0.18    # Width of Gaussian Dit (eV)

# ============================================================================
# BULK SRH LIFETIME PARAMETERS
# ============================================================================
TAU_N0_BULK = 2e-3    # Electron SRH lifetime (s)
TAU_P0_BULK = 3.1e-3  # Hole SRH lifetime (s)

# ============================================================================
# SIMULATION
# ============================================================================
ENERGY_POINTS = 1000

# ============================================================================
# PLOTTING
# ============================================================================
lw = 4
