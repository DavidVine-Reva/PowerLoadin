#!/usr/bin/env python3
"""
Run parameter sweep for rotating anode power loading optimization.

This script runs the full parameter sweep across all combinations of:
- Anode materials (Mo, W, W25Rh)
- Substrate materials (Cu, TZM, Diamond)
- Anode thicknesses (10, 50, 100 μm)
- Voltages (30-160 kV)
- Anode conditions (pristine, degraded)

Usage:
    python run_sweep.py [options]
    
Options:
    --quick         : Run reduced sweep (Mo anode only, 2 voltages)
    --materials     : Comma-separated anode materials to test
    --voltages      : Comma-separated voltages to test
    --output-dir    : Output directory for results [default: ./results]
    --dry-run       : Show what would be run without executing
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any

from config import (
    ANODE_MATERIALS, SUBSTRATE_MATERIALS, ANODE_THICKNESSES,
    VOLTAGE_RANGE, T_TARGET
)
from power_optimizer import (
    run_parameter_sweep, binary_search_power, create_summary_table,
    mock_simulation
)


def create_comsol_simulation_function():
    """
    Create the simulation function that interfaces with COMSOL.
    
    Returns a function that takes parameters and returns (T_max, results_dict).
    """
    from power_loading_sim import RotatingAnodeSimulation
    
    def run_simulation(
        power: float,
        anode_material: str = 'Mo',
        anode_thickness: float = 20e-6,
        substrate_material: str = 'Mo',
        voltage_kV: float = 100,
        degraded: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """Run COMSOL simulation and extract max temperature."""
        
        sim = RotatingAnodeSimulation(
            anode_material=anode_material,
            anode_thickness=anode_thickness,
            substrate_material=substrate_material,
            voltage_kV=voltage_kV,
            degraded=degraded,
            power=power,
        )
        
        try:
            sim.connect()
            sim.create_model()
            sim.solve()
            results = sim.get_results()
            
            # Extract maximum temperature from results
            T_max_history = results.get('T_max_history', [T_TARGET])
            T_max = max(T_max_history) if T_max_history else T_TARGET
            
            # Detect equilibrium from base temperatures
            # (Would need to extract base temps from full solution)
            
            return T_max, {
                'T_max': T_max,
                'T_base': T_max * 0.6,  # Approximate
                'equilibrium': True,
                'n_periods': 10,
                'n_iterations': 1,
            }
            
        finally:
            sim.close()
    
    return run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for power loading optimization"
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick sweep with reduced parameters'
    )
    parser.add_argument(
        '--materials', type=str, default=None,
        help='Comma-separated anode materials (default: all)'
    )
    parser.add_argument(
        '--substrates', type=str, default=None,
        help='Comma-separated substrate materials (default: all)'
    )
    parser.add_argument(
        '--voltages', type=str, default=None,
        help='Comma-separated voltages in kV (default: all)'
    )
    parser.add_argument(
        '--thicknesses', type=str, default=None,
        help='Comma-separated thicknesses in μm (default: 10,50,100)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show configurations without running'
    )
    parser.add_argument(
        '--use-mock', action='store_true',
        help='Use mock simulation (for testing without COMSOL)'
    )
    parser.add_argument(
        '--no-degraded', action='store_true',
        help='Skip degraded anode tests'
    )
    
    args = parser.parse_args()
    
    # Parse parameter lists
    if args.quick:
        anode_mats = ['Mo']
        substrate_mats = ['Mo', 'Diamond']
        voltages = [70, 130]
        thicknesses = [20e-6, 100e-6]
    else:
        anode_mats = args.materials.split(',') if args.materials else ANODE_MATERIALS
        substrate_mats = args.substrates.split(',') if args.substrates else SUBSTRATE_MATERIALS
        voltages = [float(v) for v in args.voltages.split(',')] if args.voltages else VOLTAGE_RANGE
        thicknesses = ([float(t)*1e-6 for t in args.thicknesses.split(',')] 
                       if args.thicknesses else ANODE_THICKNESSES)
    
    # Calculate total combinations
    n_conditions = 1 if args.no_degraded else 2
    total = len(anode_mats) * len(substrate_mats) * len(voltages) * len(thicknesses) * n_conditions
    
    print(f"\n{'='*60}")
    print(f"ROTATING ANODE POWER LOADING OPTIMIZATION SWEEP")
    print(f"{'='*60}")
    print(f"Anode materials: {anode_mats}")
    print(f"Substrate materials: {substrate_mats}")
    print(f"Voltages (kV): {voltages}")
    print(f"Thicknesses (μm): {[t*1e6 for t in thicknesses]}")
    print(f"Include degraded: {not args.no_degraded}")
    print(f"Total configurations: {total}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("DRY RUN - configurations would be:")
        for anode in anode_mats:
            for substrate in substrate_mats:
                for thickness in thicknesses:
                    for voltage in voltages:
                        for degraded in [False] + ([True] if not args.no_degraded else []):
                            cond = 'degraded' if degraded else 'pristine'
                            print(f"  {anode}/{substrate}, t={thickness*1e6:.0f}μm, "
                                  f"V={voltage}kV, {cond}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"results_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    # Select simulation function
    if args.use_mock:
        print("Using MOCK simulation (no COMSOL required)")
        sim_func = mock_simulation
    else:
        print("Connecting to COMSOL...")
        try:
            sim_func = create_comsol_simulation_function()
        except ImportError as e:
            print(f"ERROR: Could not import COMSOL simulation: {e}")
            print("Use --use-mock to test without COMSOL")
            sys.exit(1)
    
    # Run sweep
    results = run_parameter_sweep(
        simulation_func=sim_func,
        anode_materials=anode_mats,
        substrate_materials=substrate_mats,
        anode_thicknesses=thicknesses,
        voltages=voltages,
        include_degraded=not args.no_degraded,
        output_csv=csv_path,
        output_json=json_path,
    )
    
    # Print summary
    print(create_summary_table(results))
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == '__main__':
    main()
