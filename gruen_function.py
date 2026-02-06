"""
Gruen function module for electron energy deposition depth profile.

The Gruen range formula describes the penetration depth of electrons in materials.
The depth profile models how energy is deposited as a function of depth z.

Reference: Gruen (1957), modified by various authors for different materials.
"""

import math
from typing import Tuple


# Material-specific Gruen parameters
# R_g = (A / rho) * E^n  where E is in keV, rho in g/cm³, R_g in μm
# These are empirical fits for electron range in various materials
GRUEN_PARAMS = {
    'Mo': {'A': 0.0248, 'n': 1.75, 'rho': 10.22},  # g/cm³
    'W':  {'A': 0.0178, 'n': 1.75, 'rho': 19.30},
    'W25Rh': {'A': 0.0180, 'n': 1.75, 'rho': 19.05},  # Approximate
    'Cu': {'A': 0.0398, 'n': 1.75, 'rho': 8.96},
    'TZM': {'A': 0.0248, 'n': 1.75, 'rho': 10.16},  # Similar to Mo
    'Diamond': {'A': 0.0435, 'n': 1.75, 'rho': 3.51},
}


def gruen_range(voltage_kV: float, material: str = 'Mo') -> float:
    """
    Calculate the Gruen range (maximum penetration depth) for electrons.
    
    Args:
        voltage_kV: Accelerating voltage in kV
        material: Target material name
        
    Returns:
        Gruen range in meters
    """
    if material not in GRUEN_PARAMS:
        raise ValueError(f"Unknown material: {material}")
    
    params = GRUEN_PARAMS[material]
    A = params['A']
    n = params['n']
    rho = params['rho']
    
    # Convert voltage to keV (same numerical value)
    E_keV = voltage_kV
    
    # Gruen range in μm: R_g = (A/ρ) * E^n
    # Note: A is pre-divided by density in our parameterization
    R_g_um = A * (E_keV ** n) / rho
    
    # Convert to meters
    return R_g_um * 1e-6


def depth_profile_params(voltage_kV: float, material: str = 'Mo') -> Tuple[float, float]:
    """
    Get depth profile parameters for energy deposition.
    
    The energy deposition follows a modified Gaussian-like profile:
    dE/dz ~ (z/z_peak) * exp(-(z - z_peak)² / (2 * sigma_z²))
    
    where z_peak ≈ 0.3 * R_g and sigma_z ≈ 0.2 * R_g
    
    Args:
        voltage_kV: Accelerating voltage in kV
        material: Target material name
        
    Returns:
        Tuple of (z_peak, sigma_z) in meters
    """
    R_g = gruen_range(voltage_kV, material)
    
    # Peak energy deposition typically occurs at ~30% of Gruen range
    z_peak = 0.3 * R_g
    
    # Width of the distribution
    sigma_z = 0.2 * R_g
    
    return z_peak, sigma_z


def get_comsol_depth_expression(voltage_kV: float, material: str = 'Mo') -> str:
    """
    Generate a COMSOL expression for the normalized depth profile.
    
    The expression assumes z=0 is at the anode surface (top) and z increases
    downward (into the material).
    
    The profile is normalized so that the integral equals 1.
    
    Args:
        voltage_kV: Accelerating voltage in kV
        material: Target material name
        
    Returns:
        COMSOL expression string for depth profile G_z(z)
    """
    z_peak, sigma_z = depth_profile_params(voltage_kV, material)
    R_g = gruen_range(voltage_kV, material)
    
    # Use depth measured from anode surface (z_local = z_anode_top - z)
    # In COMSOL, we'll define z_local in the heat source expression
    
    # Modified Gaussian profile with surface suppression
    # G_z = (z_local/z_peak) * exp(-(z_local - z_peak)² / (2*sigma_z²)) for z_local > 0
    # G_z = 0 for z_local < 0 or z_local > R_g
    
    # Normalization factor (approximate)
    norm = 1.0 / (sigma_z * math.sqrt(2 * math.pi))
    
    expr = (
        f"({norm:.6e})*"
        f"(z_local/{z_peak:.6e})*"
        f"exp(-((z_local-{z_peak:.6e})^2)/(2*{sigma_z:.6e}^2))*"
        f"(z_local>0)*(z_local<{R_g:.6e})"
    )
    
    return expr


def get_simple_depth_expression(voltage_kV: float, material: str = 'Mo') -> dict:
    """
    Get depth profile parameters as a dictionary for COMSOL parameter definition.
    
    This is more flexible - parameters can be defined in COMSOL and the expression
    can reference them.
    
    Returns:
        Dictionary with parameter values and a template expression
    """
    z_peak, sigma_z = depth_profile_params(voltage_kV, material)
    R_g = gruen_range(voltage_kV, material)
    norm = 1.0 / (sigma_z * math.sqrt(2 * math.pi))
    
    return {
        'z_peak': z_peak,          # m
        'sigma_z': sigma_z,        # m  
        'R_gruen': R_g,            # m
        'norm_z': norm,            # 1/m
        'expression': (
            'norm_z*(z_local/z_peak)*'
            'exp(-((z_local-z_peak)^2)/(2*sigma_z^2))*'
            '(z_local>0)*(z_local<R_gruen)'
        )
    }


def print_gruen_table():
    """Print a table of Gruen ranges for all materials and voltages."""
    voltages = [30, 50, 70, 100, 130, 160]
    materials = list(GRUEN_PARAMS.keys())
    
    print("Gruen Range (μm) for Electron Beam Penetration")
    print("=" * 60)
    header = "Voltage (kV) |" + " | ".join(f"{m:>8}" for m in materials)
    print(header)
    print("-" * 60)
    
    for V in voltages:
        row = f"{V:>11} |"
        for mat in materials:
            R_g = gruen_range(V, mat) * 1e6  # Convert to μm
            row += f" {R_g:>8.2f}"
        print(row)


if __name__ == '__main__':
    print_gruen_table()
