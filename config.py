"""
Configuration parameters for COMSOL rotating anode X-ray source simulation.

All units are SI unless otherwise noted.
"""

# =============================================================================
# Geometry Parameters
# =============================================================================
SUBSTRATE_SIZE = (10e-3, 10e-3, 1e-3)  # (x, y, z) in meters
ANODE_THICKNESS = 20e-6  # Default anode thickness in meters (20 μm)

# =============================================================================
# Beam Parameters
# =============================================================================
# Beam spot size (σ values, FWHM ≈ 2.355 × σ)
FWHM_TO_SIGMA = 1/1.82 # For n=4 supergaussian
BEAM_FWHM_X = 70e-6 # m
BEAM_FWHM_Y = 700e-6 # m
BEAM_SIGMA_X = FWHM_X * FWHM_TO_SIGMA  # m (FWHM = 70μm -> σ = 70/2.355 ≈ 29.7μm)
BEAM_SIGMA_Y = FWHM_Y * FWHM_TO_SIGMA   # m (FWHM = 700μm -> σ = 700/2.355 ≈ 297μm)

# Track parameters
TRACK_LENGTH = 3e-3  # m (L = 3mm)
PERIOD = 1e-3        # s (rotation period = 1ms)
VELOCITY = 100.0     # m/s

# Beam motion: velocity is fixed at 100 m/s
# During beam-on, beam moves linear distance L at velocity v
# t_on = L / v

# Default voltage for Gruen function
DEFAULT_VOLTAGE_KV = 100  # kV

# =============================================================================
# Thermal Parameters
# =============================================================================
T_TARGET = 1773.15   # K (1500°C) - target max temperature
T_AMBIENT = 293.15   # K (20°C) - room temperature / boundary condition
T_MELTING = {        # Melting points in K
    'Mo': 2896,
    'W': 3695,
    'W25Rh': 3500,   # Approximate
    'Cu': 1358,
    'TZM': 2896,     # Similar to Mo
    'Diamond': 3820, # Sublimation temperature under pressure
}

# =============================================================================
# Simulation Parameters
# =============================================================================
N_PERIODS_MIN = 10           # Minimum periods to run for equilibrium
N_PERIODS_MAX = 100          # Maximum periods before giving up
DT_BEAM_ON = 1e-6            # Time step during beam-on (1 μs)
DT_BEAM_OFF = 100e-6          # Time step during beam-off (10 μs)

# =============================================================================
# Optimization Parameters
# =============================================================================
POWER_INITIAL = 100.0        # Initial power guess in Watts
POWER_MIN = 1.0              # Minimum power to search
POWER_MAX = 2000.0          # Maximum power to search
POWER_TOLERANCE = 0.01       # Relative tolerance for power convergence
TEMP_TOLERANCE = 5.0         # K, tolerance for temperature match to target
EQUILIBRIUM_TOLERANCE = 1.0  # K, tolerance for base temp equilibrium detection

# =============================================================================
# Material Options
# =============================================================================
ANODE_MATERIALS = ['Mo', 'W', 'W25Rh']
SUBSTRATE_MATERIALS = ['Cu', 'TZM', 'Diamond']
ANODE_THICKNESSES = [10e-6, 50e-6, 100e-6]  # 10, 50, 100 μm
DEGRADATION_FACTOR = 0.5  # 50% conductivity for degraded condition
VOLTAGE_RANGE = [30, 50, 70, 100, 130, 160]  # kV

# =============================================================================
# COMSOL Settings
# =============================================================================
COMSOL_VERSION = '6.4'
MODEL_NAME = 'rotating_anode_heat_transfer'
