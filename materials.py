"""
Temperature-dependent material properties for rotating anode simulation.

Material data sourced from TempDependentMaterials.xml (COMSOL material library).
Interpolation tables are provided for: Mo, TZM, W, Diamond.
Analytical fits are used for: Cu, W25Rh (not in XML).

For COMSOL usage, call create_comsol_interpolation_functions() to register
the interpolation functions in the model, then reference them as e.g. k_Mo(T).
"""

from typing import Dict, Any, List, Tuple


# =============================================================================
# Interpolation Table Data (from TempDependentMaterials.xml)
# Each table is a list of (T_kelvin, value) tuples.
# =============================================================================

# --- Molybdenum (Mo) ---
# Source: CRC Handbook, NIST
_k_Mo_table = [
    (298.15, 138.3), (398.15, 134), (498.15, 129.9), (598.15, 125.9),
    (698.15, 122.1), (798.15, 118.4), (898.15, 114.9), (998.15, 111.6),
    (1098.15, 108.4), (1198.15, 105.4), (1298.15, 102.6), (1398.15, 99.9),
    (1498.15, 97.4), (1598.15, 95), (1698.15, 92.8), (1798.15, 90.7),
    (1898.15, 88.7), (1998.15, 86.9), (2098.15, 85.2), (2198.15, 82.9),
    (2298.15, 81.5), (2398.15, 80.3), (2498.15, 79.2), (2598.15, 78.2),
    (2698.15, 77.4), (2798.15, 76.8), (2898.15, 76.3),
]  # W/(m*K)

_Cp_Mo_table = [
    (298.15, 251), (398.15, 254.4), (498.15, 257.8), (598.15, 261),
    (698.15, 264.1), (798.15, 267.2), (898.15, 270.2), (998.15, 273),
    (1098.15, 275.7), (1198.15, 278.3), (1298.15, 280.8), (1398.15, 283.2),
    (1498.15, 285.5), (1598.15, 287.7), (1698.15, 289.8), (1798.15, 291.8),
    (1898.15, 293.7), (1998.15, 295.6), (2098.15, 297.4), (2198.15, 300.3),
    (2298.15, 302), (2398.15, 303.5), (2498.15, 305), (2598.15, 306.4),
    (2698.15, 307.7), (2798.15, 308.9), (2898.15, 310),
]  # J/(kg*K)

_rho_Mo_table = [
    (298.15, 10220), (398.15, 10205), (498.15, 10191), (598.15, 10176),
    (698.15, 10161), (798.15, 10147), (898.15, 10132), (998.15, 10118),
    (1098.15, 10103), (1198.15, 10089), (1298.15, 10074), (1398.15, 10060),
    (1498.15, 10046), (1598.15, 10031), (1698.15, 10017), (1798.15, 10003),
    (1898.15, 9989), (1998.15, 9975), (2098.15, 9962), (2198.15, 9948),
    (2298.15, 9934), (2398.15, 9920), (2498.15, 9906), (2598.15, 9892),
    (2698.15, 9879), (2798.15, 9865), (2898.15, 9851),
]  # kg/m^3


# --- TZM alloy (Mo-0.5Ti-0.08Zr) ---
# Source: Briggs & Barr (1971), Southern Research Institute (1966)
_k_TZM_table = [
    (298.15, 129.7891466), (398.15, 125.7267685), (498.15, 121.9455545),
    (598.15, 118.4257725), (698.15, 115.1476908), (798.15, 112.0915775),
    (898.15, 109.2377005), (998.15, 106.566328), (1098.15, 104.0577281),
    (1198.15, 101.6921689), (1298.15, 99.44991851), (1398.15, 97.31124497),
    (1498.15, 95.25641641), (1598.15, 93.26570091), (1698.15, 91.31936656),
    (1798.15, 89.39768147), (1898.15, 87.48091373), (1998.15, 85.54933143),
    (2098.15, 83.58320266), (2198.15, 81.56279551), (2298.15, 79.46837809),
    (2398.15, 77.28021849), (2498.15, 74.97858479), (2598.15, 72.5437451),
    (2698.15, 69.9559675), (2798.15, 67.19552009),
]  # W/(m*K)

_Cp_TZM_table = [
    (298.15, 250.2146522), (398.15, 254.5653324), (498.15, 259.0389531),
    (598.15, 263.6547714), (698.15, 268.4320444), (798.15, 273.390029),
    (898.15, 278.5479822), (998.15, 283.925161), (1098.15, 289.5408223),
    (1198.15, 295.4142233), (1298.15, 301.5646209), (1398.15, 308.0112721),
    (1498.15, 314.7734339), (1598.15, 321.8703633), (1698.15, 329.3213173),
    (1798.15, 337.1455529), (1898.15, 345.3623271), (1998.15, 353.9908968),
    (2098.15, 363.0505192), (2198.15, 372.5604511), (2298.15, 382.5399496),
    (2398.15, 393.0082717), (2498.15, 403.9846744), (2598.15, 415.4884146),
    (2698.15, 427.5387494), (2798.15, 440.1549358),
]  # J/(kg*K)

_rho_TZM_table = [
    (298.15, 10199.15857), (398.15, 10182.35906), (498.15, 10165.61481),
    (598.15, 10148.92553), (698.15, 10132.29097), (798.15, 10115.71084),
    (898.15, 10099.18489), (998.15, 10082.71284), (1098.15, 10066.29444),
    (1198.15, 10049.92943), (1298.15, 10033.61754), (1398.15, 10017.35851),
    (1498.15, 10001.15209), (1598.15, 9984.99803), (1698.15, 9968.896067),
    (1798.15, 9952.845953), (1898.15, 9936.847438), (1998.15, 9920.900273),
    (2098.15, 9905.004212), (2198.15, 9889.159009), (2298.15, 9873.364422),
    (2398.15, 9857.620206), (2498.15, 9841.926123), (2598.15, 9826.281932),
    (2698.15, 9810.687397), (2798.15, 9795.142282),
]  # kg/m^3


# --- Tungsten (W) ---
# Source: CRC Handbook, NIST, White & Collocott (1984)
_k_W_table = [
    (298.15, 175.6314705), (398.15, 161.7151304), (498.15, 151.1317565),
    (598.15, 142.724791), (698.15, 135.862904), (798.15, 130.1523911),
    (898.15, 125.3285852), (998.15, 121.2046533), (1098.15, 117.6439892),
    (1198.15, 114.5439725), (1298.15, 111.8258135), (1398.15, 109.4279107),
    (1498.15, 107.3013493), (1598.15, 105.4067635), (1698.15, 103.7120955),
    (1798.15, 102.190964), (1898.15, 100.8214536), (1998.15, 99.58520359),
    (2098.15, 98.46671244), (2198.15, 97.45279994), (2298.15, 96.53218705),
    (2398.15, 95.69516406), (2498.15, 94.93332612), (2598.15, 94.23936068),
    (2698.15, 93.6068753), (2798.15, 93.03025703), (2898.15, 92.50455696),
    (2998.15, 92.02539468), (3098.15, 91.58887879), (3198.15, 91.19154047),
    (3298.15, 90.83027753), (3398.15, 90.50230717), (3498.15, 90.2051259),
    (3598.15, 89.93647528),
]  # W/(m*K)

_Cp_W_table = [
    (298.15, 132), (398.15, 135.3022864), (498.15, 138.5918762),
    (598.15, 141.8687695), (698.15, 145.1329663), (798.15, 148.3844666),
    (898.15, 151.6232703), (998.15, 154.8493776), (1098.15, 158.0627883),
    (1198.15, 161.2635025), (1298.15, 164.4515201), (1398.15, 167.6268412),
    (1498.15, 170.7894659), (1598.15, 173.9393939), (1698.15, 177.0766255),
    (1798.15, 180.2011605), (1898.15, 183.3129991), (1998.15, 186.4121411),
    (2098.15, 189.4985865), (2198.15, 192.5723355), (2298.15, 195.6333879),
    (2398.15, 198.6817438), (2498.15, 201.7174032), (2598.15, 204.740366),
    (2698.15, 207.7506323), (2798.15, 210.7482022), (2898.15, 213.7330754),
    (2998.15, 216.7052522), (3098.15, 219.6647324), (3198.15, 222.6115161),
    (3298.15, 225.5456033), (3398.15, 228.466994), (3498.15, 231.3756881),
    (3598.15, 234.2716858),
]  # J/(kg*K)

_rho_W_table = [
    (298.15, 19300), (398.15, 19273.98013), (498.15, 19248.03032),
    (598.15, 19222.15029), (698.15, 19196.33977), (798.15, 19170.59846),
    (898.15, 19144.9261), (998.15, 19119.3224), (1098.15, 19093.7871),
    (1198.15, 19068.31991), (1298.15, 19042.92057), (1398.15, 19017.58881),
    (1498.15, 18992.32435), (1598.15, 18967.12692), (1698.15, 18941.99627),
    (1798.15, 18916.93212), (1898.15, 18891.93422), (1998.15, 18867.0023),
    (2098.15, 18842.13609), (2198.15, 18817.33535), (2298.15, 18792.59981),
    (2398.15, 18767.92921), (2498.15, 18743.3233), (2598.15, 18718.78182),
    (2698.15, 18694.30453), (2798.15, 18669.89117), (2898.15, 18645.54149),
    (2998.15, 18621.25525), (3098.15, 18597.03218), (3198.15, 18572.87206),
    (3298.15, 18548.77463), (3398.15, 18524.73965), (3498.15, 18500.76687),
    (3598.15, 18476.85606),
]  # kg/m^3


# --- Diamond (CVD, type I) ---
# Source: Ho et al. (1972), Pankratz (1982), Reeber & Wang (1996)
_k_Dia_table = [
    (298.15, 1811.168875), (398.15, 1356.272762), (498.15, 1084.01084),
    (598.15, 902.7835827), (698.15, 773.4727494), (798.15, 676.5645555),
    (898.15, 601.2358737), (998.15, 541.0008516), (1098.15, 491.7361016),
    (1198.15, 450.6948212), (1298.15, 415.9765821), (1398.15, 386.224654),
    (1498.15, 360.4445483), (1598.15, 337.8906861), (1698.15, 317.9931101),
    (1798.15, 300.3086506), (1898.15, 284.4875273), (1998.15, 270.2499812),
    (2098.15, 257.3695875), (2198.15, 245.6611241), (2298.15, 234.9716076),
    (2398.15, 225.1735713), (2498.15, 216.1599584), (2598.15, 207.840194),
    (2698.15, 200.137131), (2798.15, 192.9846506), (2898.15, 186.3257595),
    (2998.15, 180.1110685), (3098.15, 174.2975647), (3198.15, 168.847615),
    (3298.15, 163.7281506), (3398.15, 158.9099951), (3498.15, 154.3673084),
    (3598.15, 150.077123), (3698.15, 146.0189554), (3798.15, 142.1744797),
]  # W/(m*K)

_Cp_Dia_table = [
    (298.15, 502), (398.15, 510.6293224), (498.15, 520.4667333),
    (598.15, 531.3920635), (698.15, 543.2851432), (798.15, 556.025803),
    (898.15, 569.4938735), (998.15, 583.569185), (1098.15, 598.131568),
    (1198.15, 613.0608532), (1298.15, 628.2368708), (1398.15, 643.5394515),
    (1498.15, 658.8484257), (1598.15, 674.043624), (1698.15, 689.0048767),
    (1798.15, 703.6120144), (1898.15, 717.7448675), (1998.15, 731.2832667),
    (2098.15, 744.1070422), (2198.15, 756.0960247), (2298.15, 767.1300446),
    (2398.15, 777.0889325), (2498.15, 785.8525187), (2598.15, 793.3006338),
    (2698.15, 799.3131083), (2798.15, 803.7697726), (2898.15, 806.5504573),
    (2998.15, 807.5349928), (3098.15, 806.6032097), (3198.15, 803.6349384),
    (3298.15, 798.5100093), (3398.15, 791.1082531), (3498.15, 781.3095001),
    (3598.15, 768.9935809), (3698.15, 754.0403259), (3798.15, 736.3295657),
]  # J/(kg*K)

_rho_Dia_table = [
    (298.15, 3515), (398.15, 3513.945816), (498.15, 3512.892265),
    (598.15, 3511.839345), (698.15, 3510.787056), (798.15, 3509.735397),
    (898.15, 3508.684368), (998.15, 3507.633969), (1098.15, 3506.584198),
    (1198.15, 3505.535055), (1298.15, 3504.48654), (1398.15, 3503.438652),
    (1498.15, 3502.391391), (1598.15, 3501.344755), (1698.15, 3500.298745),
    (1798.15, 3499.25336), (1898.15, 3498.208599), (1998.15, 3497.164461),
    (2098.15, 3496.120947), (2198.15, 3495.078055), (2298.15, 3494.035785),
    (2398.15, 3492.994137), (2498.15, 3491.953109), (2598.15, 3490.912702),
    (2698.15, 3489.872915), (2798.15, 3488.833747), (2898.15, 3487.795197),
    (2998.15, 3486.757266), (3098.15, 3485.719952), (3198.15, 3484.683256),
    (3298.15, 3483.647175), (3398.15, 3482.611711), (3498.15, 3481.576862),
    (3598.15, 3480.542628), (3698.15, 3479.509008), (3798.15, 3478.476002),
]  # kg/m^3


# =============================================================================
# Interpolation data registry: material -> {property: (funcname, table, unit)}
# =============================================================================
INTERPOLATION_DATA = {
    'Mo': {
        'k': ('k_Mo', _k_Mo_table, 'W/(m*K)'),
        'Cp': ('Cp_Mo', _Cp_Mo_table, 'J/(kg*K)'),
        'rho': ('rho_Mo', _rho_Mo_table, 'kg/m^3'),
    },
    'TZM': {
        'k': ('k_TZM', _k_TZM_table, 'W/(m*K)'),
        'Cp': ('Cp_TZM', _Cp_TZM_table, 'J/(kg*K)'),
        'rho': ('rho_TZM', _rho_TZM_table, 'kg/m^3'),
    },
    'W': {
        'k': ('k_W', _k_W_table, 'W/(m*K)'),
        'Cp': ('Cp_W', _Cp_W_table, 'J/(kg*K)'),
        'rho': ('rho_W', _rho_W_table, 'kg/m^3'),
    },
    'Diamond': {
        'k': ('k_Dia', _k_Dia_table, 'W/(m*K)'),
        'Cp': ('Cp_Dia', _Cp_Dia_table, 'J/(kg*K)'),
        'rho': ('rho_Dia', _rho_Dia_table, 'kg/m^3'),
    },
}


def _table_to_comsol_string(table: List[Tuple[float, float]]) -> str:
    """Convert a Python table to COMSOL interpolation table string format.

    Returns string like: {{'298.15','138.3'},{'398.15','134'},...}
    """
    entries = [f"{{'{t}','{v}'}}" for t, v in table]
    return '{' + ','.join(entries) + '}'


def create_comsol_interpolation_functions(java_model, material: str) -> None:
    """Create COMSOL interpolation functions for a material's properties.

    Registers interpolation functions (e.g. k_Mo, Cp_Mo, rho_Mo) in the
    model's component function container. These can then be referenced
    as 'k_Mo(T)' in material property definitions.

    Args:
        java_model: The COMSOL Java model object (model.java)
        material: Material name ('Mo', 'W', 'TZM', 'Diamond')
    """
    if material not in INTERPOLATION_DATA:
        return  # No interpolation data for this material (Cu, W25Rh)

    comp = java_model.component("comp1")
    func_container = comp.func()

    for prop, (funcname, table, unit) in INTERPOLATION_DATA[material].items():
        tag = f"int_{funcname}"
        interp = func_container.create(tag, "Interpolation")
        interp.label(f"{funcname} interpolation")
        interp.set("funcname", funcname)
        interp.set("table", _table_to_comsol_string(table))
        interp.set("fununit", f"{{'{unit}'}}")
        interp.set("argunit", "{{'K'}}")
        interp.set("extrap", "linear")  # Linear extrapolation beyond table


def get_material_properties(material: str, degraded: bool = False) -> Dict[str, Any]:
    """
    Get temperature-dependent material properties for COMSOL.

    For materials with interpolation data (Mo, W, TZM, Diamond), returns
    function references like 'k_Mo(T)'. For others (Cu, W25Rh), returns
    analytical expressions.

    Args:
        material: Material name ('Mo', 'W', 'W25Rh', 'Cu', 'TZM', 'Diamond')
        degraded: If True, reduce thermal conductivity by 50%

    Returns:
        Dictionary with 'density', 'thermal_conductivity', 'heat_capacity',
        'name', 'description', 'has_interpolation' keys
    """
    material_funcs = {
        'Mo': _molybdenum_properties,
        'W': _tungsten_properties,
        'W25Rh': _w25rh_properties,
        'Cu': _copper_properties,
        'TZM': _tzm_properties,
        'Diamond': _diamond_properties,
    }

    if material not in material_funcs:
        raise ValueError(
            f"Unknown material: {material}. Choose from {list(material_funcs.keys())}")

    props = material_funcs[material]()

    if degraded:
        k_expr = props['thermal_conductivity']
        props['thermal_conductivity'] = f"0.5*({k_expr})"
        props['description'] += ' (degraded - 50% conductivity)'

    return props


def _molybdenum_properties() -> Dict[str, Any]:
    """Molybdenum (Mo) - uses interpolation functions from XML data."""
    return {
        'name': 'Molybdenum',
        'description': 'Molybdenum (Mo) with temperature-dependent properties',
        'density': 'rho_Mo(T)',
        'thermal_conductivity': 'k_Mo(T)',
        'heat_capacity': 'Cp_Mo(T)',
        'has_interpolation': True,
    }


def _tungsten_properties() -> Dict[str, Any]:
    """Tungsten (W) - uses interpolation functions from XML data."""
    return {
        'name': 'Tungsten',
        'description': 'Tungsten (W) with temperature-dependent properties',
        'density': 'rho_W(T)',
        'thermal_conductivity': 'k_W(T)',
        'heat_capacity': 'Cp_W(T)',
        'has_interpolation': True,
    }


def _w25rh_properties() -> Dict[str, Any]:
    """Tungsten-25% Rhenium alloy (W25Rh) - analytical fit, no XML data."""
    return {
        'name': 'W25Rh',
        'description': 'Tungsten-25% Rhenium alloy with temperature-dependent properties',
        'density': '19050 - 0.3*(T-300)',
        'thermal_conductivity': '0.85*(k_W(T))',
        'heat_capacity': '135 + 0.02*T',
        'has_interpolation': False,
    }


def _copper_properties() -> Dict[str, Any]:
    """Copper (Cu) - analytical fit, not in XML."""
    return {
        'name': 'Copper',
        'description': 'Copper (Cu) with temperature-dependent properties',
        'density': '8960 - 0.5*(T-300)',
        'thermal_conductivity': '401 - 0.07*(T-300)',
        'heat_capacity': '385 + 0.1*(T-300)',
        'has_interpolation': False,
    }


def _tzm_properties() -> Dict[str, Any]:
    """TZM alloy (Ti-Zr-Mo) - uses interpolation functions from XML data."""
    return {
        'name': 'TZM',
        'description': 'TZM alloy (Mo-0.5Ti-0.08Zr) with temperature-dependent properties',
        'density': 'rho_TZM(T)',
        'thermal_conductivity': 'k_TZM(T)',
        'heat_capacity': 'Cp_TZM(T)',
        'has_interpolation': True,
    }


def _diamond_properties() -> Dict[str, Any]:
    """CVD Diamond - uses interpolation functions from XML data."""
    return {
        'name': 'Diamond',
        'description': 'CVD Diamond with temperature-dependent properties',
        'density': 'rho_Dia(T)',
        'thermal_conductivity': 'k_Dia(T)',
        'heat_capacity': 'Cp_Dia(T)',
        'has_interpolation': True,
    }


def get_comsol_material_definition(material: str, degraded: bool = False) -> Dict[str, str]:
    """
    Get COMSOL-ready material property expressions.

    Returns expressions that can be directly used in COMSOL material definitions.
    For materials with interpolation data, these reference interpolation functions
    that must first be created via create_comsol_interpolation_functions().
    """
    props = get_material_properties(material, degraded)

    return {
        'rho': props['density'],             # kg/m^3
        'k': props['thermal_conductivity'],  # W/(m*K)
        'Cp': props['heat_capacity'],        # J/(kg*K)
    }


def get_interpolation_materials_used(anode_material: str,
                                     substrate_material: str) -> List[str]:
    """Return list of materials that need COMSOL interpolation functions.

    Includes W if W25Rh is used (since W25Rh references k_W).
    """
    needs_interp = set()
    for mat in [anode_material, substrate_material]:
        if mat in INTERPOLATION_DATA:
            needs_interp.add(mat)
        if mat == 'W25Rh':
            needs_interp.add('W')  # W25Rh references k_W(T)
    return list(needs_interp)


# Material melting points for reference (K)
MELTING_POINTS = {
    'Mo': 2896,
    'W': 3695,
    'W25Rh': 3500,
    'Cu': 1358,
    'TZM': 2896,
    'Diamond': 3820,  # Sublimation
}
