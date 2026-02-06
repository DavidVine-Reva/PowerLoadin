"""
Temperature-dependent material properties for rotating anode simulation.

All properties are given as analytical expressions suitable for COMSOL.
Temperature T is in Kelvin.

Data sources:
- Molybdenum: CRC Handbook, NIST
- Tungsten: CRC Handbook, NIST  
- Copper: CRC Handbook
- TZM: Manufacturer data (similar to Mo with slight improvements)
- Diamond: Literature values for CVD diamond
"""

from typing import Dict, Any


def get_material_properties(material: str, degraded: bool = False) -> Dict[str, Any]:
    """
    Get temperature-dependent material properties for COMSOL.
    
    Args:
        material: Material name ('Mo', 'W', 'W25Rh', 'Cu', 'TZM', 'Diamond')
        degraded: If True, reduce thermal conductivity by 50%
        
    Returns:
        Dictionary with 'density', 'thermal_conductivity', 'heat_capacity' expressions
    """
    
    materials = {
        'Mo': _molybdenum_properties(),
        'W': _tungsten_properties(),
        'W25Rh': _w25rh_properties(),
        'Cu': _copper_properties(),
        'TZM': _tzm_properties(),
        'Diamond': _diamond_properties(),
    }
    
    if material not in materials:
        raise ValueError(f"Unknown material: {material}. Choose from {list(materials.keys())}")
    
    props = materials[material].copy()
    
    if degraded:
        # Apply degradation factor to thermal conductivity
        k_expr = props['thermal_conductivity']
        props['thermal_conductivity'] = f"0.5*({k_expr})"
        props['description'] += ' (degraded - 50% conductivity)'
    
    return props


def _molybdenum_properties() -> Dict[str, Any]:
    """Molybdenum (Mo) temperature-dependent properties."""
    return {
        'name': 'Molybdenum',
        'description': 'Molybdenum (Mo) with temperature-dependent properties',
        # Density: slight decrease with temperature
        # ρ(T) ≈ 10220 - 0.5*(T-300) kg/m³ (simplified linear fit)
        'density': '10220 - 0.5*(T-300)',
        # Thermal conductivity: k decreases with T
        # k(T) ≈ 138 W/(m·K) at 300K, decreases to ~100 W/(m·K) at 1500K
        'thermal_conductivity': '138 - 0.025*(T-300)',
        # Specific heat: increases with temperature
        # cp(T) ≈ 251 + 0.03*T J/(kg·K)
        'heat_capacity': '251 + 0.03*T',
    }


def _tungsten_properties() -> Dict[str, Any]:
    """Tungsten (W) temperature-dependent properties."""
    return {
        'name': 'Tungsten',
        'description': 'Tungsten (W) with temperature-dependent properties',
        # Density: ρ ≈ 19300 kg/m³, slight decrease with T
        'density': '19300 - 0.3*(T-300)',
        # Thermal conductivity: k(T) ≈ 173 W/(m·K) at 300K, decreases with T
        # Polynomial fit: k = 174.9 - 0.0342*T + 1.14e-5*T^2
        'thermal_conductivity': '174.9 - 0.0342*T + 1.14e-5*T^2',
        # Specific heat: cp ≈ 132 + 0.02*T J/(kg·K)
        'heat_capacity': '132 + 0.02*T',
    }


def _w25rh_properties() -> Dict[str, Any]:
    """Tungsten-25% Rhenium alloy (W25Rh) properties."""
    # W-Re alloys have slightly lower conductivity but better high-temp properties
    return {
        'name': 'W25Rh',
        'description': 'Tungsten-25% Rhenium alloy with temperature-dependent properties',
        # Density: slightly less than pure W due to Re content
        'density': '19050 - 0.3*(T-300)',
        # Thermal conductivity: lower than pure W
        'thermal_conductivity': '0.85*(174.9 - 0.0342*T + 1.14e-5*T^2)',
        # Specific heat: similar to W
        'heat_capacity': '135 + 0.02*T',
    }


def _copper_properties() -> Dict[str, Any]:
    """Copper (Cu) temperature-dependent properties."""
    return {
        'name': 'Copper',
        'description': 'Copper (Cu) with temperature-dependent properties',
        # Density
        'density': '8960 - 0.5*(T-300)',
        # Thermal conductivity: very high, decreases with T
        # k(T) ≈ 401 W/(m·K) at 300K
        'thermal_conductivity': '401 - 0.07*(T-300)',
        # Specific heat
        'heat_capacity': '385 + 0.1*(T-300)',
    }


def _tzm_properties() -> Dict[str, Any]:
    """TZM alloy (Ti-Zr-Mo) temperature-dependent properties."""
    # TZM has slightly better properties than pure Mo at high temperatures
    return {
        'name': 'TZM',
        'description': 'TZM alloy (Mo-0.5Ti-0.08Zr) with temperature-dependent properties',
        # Density: similar to Mo
        'density': '10160 - 0.5*(T-300)',
        # Thermal conductivity: slightly lower than Mo at room temp, better retention at high T
        'thermal_conductivity': '130 - 0.02*(T-300)',
        # Specific heat: similar to Mo
        'heat_capacity': '250 + 0.03*T',
    }


def _diamond_properties() -> Dict[str, Any]:
    """CVD Diamond temperature-dependent properties."""
    return {
        'name': 'Diamond',
        'description': 'CVD Diamond with temperature-dependent properties',
        # Density: essentially constant
        'density': '3510',
        # Thermal conductivity: exceptional at room temp, drops significantly with T
        # k(T) ≈ 2000 W/(m·K) at 300K, follows ~1/T dependence
        # k = 2000 * (300/T)^1.2 for T > 300K
        'thermal_conductivity': '2000*pow(300/T, 1.2)',
        # Specific heat: increases strongly with T
        # Debye model approximation
        'heat_capacity': '500 + 2.0*(T-300)',
    }


def get_comsol_material_definition(material: str, degraded: bool = False) -> Dict[str, str]:
    """
    Get COMSOL-ready material property expressions.
    
    Returns expressions that can be directly used in COMSOL analytic functions.
    """
    props = get_material_properties(material, degraded)
    
    return {
        'rho': props['density'],      # kg/m³
        'k': props['thermal_conductivity'],  # W/(m·K)
        'Cp': props['heat_capacity'],  # J/(kg·K)
    }


# Material melting points for reference (K)
MELTING_POINTS = {
    'Mo': 2896,
    'W': 3695,
    'W25Rh': 3500,
    'Cu': 1358,
    'TZM': 2896,
    'Diamond': 3820,  # Sublimation
}
