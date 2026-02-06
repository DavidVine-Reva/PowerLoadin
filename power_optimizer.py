"""
Power optimization module for rotating anode simulation.

Implements:
1. Binary search to find power loading that achieves target temperature
2. Thermal equilibrium detection
3. Parameter sweep automation across materials, thicknesses, voltages
"""

import csv
import json
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from config import (
    ANODE_MATERIALS, SUBSTRATE_MATERIALS, ANODE_THICKNESSES,
    VOLTAGE_RANGE, DEGRADATION_FACTOR,
    T_TARGET, POWER_INITIAL, POWER_MIN, POWER_MAX,
    POWER_TOLERANCE, TEMP_TOLERANCE, EQUILIBRIUM_TOLERANCE,
    N_PERIODS_MIN, N_PERIODS_MAX, PERIOD
)


@dataclass
class OptimizationResult:
    """Result of a single power optimization run."""
    anode_material: str
    anode_thickness_um: float
    substrate_material: str
    voltage_kV: float
    degraded: bool
    
    # Results
    optimal_power_W: float
    max_temperature_K: float
    max_temperature_C: float
    equilibrium_reached: bool
    n_periods_to_equilibrium: int
    base_temperature_K: float
    
    # Metadata
    n_iterations: int
    total_time_s: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_equilibrium(
    base_temps: List[float],
    tolerance: float = EQUILIBRIUM_TOLERANCE
) -> bool:
    """
    Detect if thermal equilibrium has been reached.
    
    Equilibrium is reached when the base temperature (temperature just before
    beam turns on again) stabilizes.
    
    Args:
        base_temps: List of base temperatures for each period
        tolerance: Maximum allowed variation in K
        
    Returns:
        True if equilibrium detected
    """
    if len(base_temps) < 3:
        return False
    
    # Check last 3 values
    recent = base_temps[-3:]
    variation = max(recent) - min(recent)
    
    return variation < tolerance


def predict_equilibrium_temperature(
    base_temps: List[float],
    periods: Optional[List[float]] = None,
    min_points: int = 5,
) -> Tuple[float, float, bool]:
    """
    Predict equilibrium base temperature by fitting exponential to base temp history.
    
    The base temperature typically follows: T(n) = T_eq + (T_0 - T_eq) * exp(-n/tau)
    
    By fitting this exponential, we can predict T_eq without running to full equilibrium,
    which could take 50-100+ cycles.
    
    Args:
        base_temps: List of base temperatures at end of each period
        periods: Optional list of period indices (defaults to 0, 1, 2, ...)
        min_points: Minimum number of points needed for reliable fit
        
    Returns:
        Tuple of (predicted_T_eq, time_constant_tau, fit_reliable)
    """
    n_points = len(base_temps)
    
    if n_points < min_points:
        # Not enough points - return last value as best estimate
        return base_temps[-1] if base_temps else 293.15, float('inf'), False
    
    if periods is None:
        periods = list(range(n_points))
    
    # For exponential fit, we use linearization:
    # If T(n) = T_eq + A*exp(-n/tau), we can't directly linearize due to unknown T_eq
    # Use iterative approach or scipy if available
    
    try:
        from scipy.optimize import curve_fit
        import numpy as np
        
        def exp_decay(n, T_eq, A, tau):
            return T_eq + A * np.exp(-np.array(n) / tau)
        
        # Initial guess: T_eq = extrapolated from trend, A = first - last, tau = n_points/2
        T0 = base_temps[0]
        Tlast = base_temps[-1]
        # Better initial guess for T_eq
        T_eq_guess = Tlast - 0.5 * (Tlast - T0) if Tlast < T0 else Tlast + 0.5 * (Tlast - T0)
        p0 = [T_eq_guess, T0 - Tlast, n_points/2]
        
        # Bounds to ensure physical behavior (allow T_eq anywhere in reasonable range)
        bounds = (
            [200, -1e6, 0.1],  # Lower bounds (200K minimum for any temperature)
            [5000, 1e6, n_points * 10]  # Upper bounds
        )
        
        popt, pcov = curve_fit(
            exp_decay, 
            periods, 
            base_temps, 
            p0=p0,
            bounds=bounds,
            maxfev=1000
        )
        
        T_eq, A, tau = popt
        
        # Check fit quality - if tau is very large, we're already near equilibrium
        if tau > n_points * 5:
            return Tlast, tau, True  # Already at equilibrium
        
        # Verify the fit makes physical sense
        if T_eq < 200 or T_eq > 5000:  # Outside reasonable range
            return Tlast, tau, False
        
        return T_eq, tau, True
        
    except ImportError:
        # Scipy not available - use simple linear extrapolation on log of differences
        return _predict_equilibrium_simple(base_temps)
    except Exception as e:
        # Fit failed - use simple method
        print(f"  Warning: Exponential fit failed ({e}), using simple extrapolation")
        return _predict_equilibrium_simple(base_temps)


def _predict_equilibrium_simple(base_temps: List[float]) -> Tuple[float, float, bool]:
    """
    Simple equilibrium prediction when scipy is not available.
    
    Uses the rate of change to estimate equilibrium.
    """
    if len(base_temps) < 3:
        return base_temps[-1] if base_temps else 293.15, float('inf'), False
    
    # Calculate temperature changes between consecutive periods
    dT = [base_temps[i+1] - base_temps[i] for i in range(len(base_temps)-1)]
    
    # If changes are decreasing, fit exponential decay to changes
    # dT(n) ~ A * exp(-n/tau)
    # When dT -> 0, we're at equilibrium
    
    # Estimate equilibrium as current temp + sum of projected future changes
    # Assuming geometric decay of temperature changes
    
    if len(dT) >= 2 and dT[-2] != 0:
        ratio = dT[-1] / dT[-2] if dT[-2] != 0 else 0.5
        ratio = min(max(ratio, 0.1), 0.99)  # Bound ratio
        
        # Sum of geometric series: remaining change = dT[-1] * ratio / (1 - ratio)
        remaining_change = dT[-1] * ratio / (1 - ratio) if ratio < 1 else dT[-1] * 10
        T_eq = base_temps[-1] + remaining_change
        
        # Estimate tau from ratio: ratio = exp(-1/tau) -> tau = -1/ln(ratio)
        tau = -1 / math.log(ratio) if 0 < ratio < 1 else float('inf')
        
        return T_eq, tau, True
    
    # Fallback: use last value
    return base_temps[-1], float('inf'), False


def estimate_cycles_to_equilibrium(
    base_temps: List[float],
    tolerance: float = EQUILIBRIUM_TOLERANCE,
) -> Tuple[int, float]:
    """
    Estimate how many more cycles are needed to reach equilibrium.
    
    Args:
        base_temps: List of base temperatures recorded so far
        tolerance: Temperature tolerance for equilibrium
        
    Returns:
        Tuple of (estimated_cycles_remaining, predicted_equilibrium_temp)
    """
    T_eq, tau, reliable = predict_equilibrium_temperature(base_temps)
    
    if not reliable or tau == float('inf'):
        return N_PERIODS_MAX, base_temps[-1] if base_temps else T_TARGET
    
    # Number of cycles to get within tolerance of equilibrium
    # |T(n) - T_eq| < tolerance
    # |A * exp(-n/tau)| < tolerance
    # n > tau * ln(|A|/tolerance)
    
    A = abs(base_temps[0] - T_eq) if base_temps else 100
    
    if A < tolerance:
        return 0, T_eq  # Already at equilibrium
    
    n_needed = int(tau * math.log(A / tolerance)) + 1
    n_remaining = max(0, n_needed - len(base_temps))
    
    return n_remaining, T_eq


def binary_search_power(
    simulation_func,
    target_temp: float = T_TARGET,
    power_min: float = POWER_MIN,
    power_max: float = POWER_MAX,
    tolerance: float = POWER_TOLERANCE,
    max_iterations: int = 20,
    **sim_kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    Binary search to find power that achieves target max temperature.
    
    Args:
        simulation_func: Function that takes power and returns (T_max, results_dict)
        target_temp: Target maximum temperature in K
        power_min: Minimum power to search
        power_max: Maximum power to search
        tolerance: Relative tolerance for convergence
        max_iterations: Maximum search iterations
        **sim_kwargs: Additional arguments passed to simulation
        
    Returns:
        Tuple of (optimal_power, results_dict)
    """
    print(f"\n{'='*60}")
    print(f"Binary Search for Target Temperature: {target_temp:.1f} K ({target_temp-273.15:.1f} °C)")
    print(f"Power range: [{power_min:.1f}, {power_max:.1f}] W")
    print(f"{'='*60}")
    
    # Start with initial guess (geometric mean)
    power = math.sqrt(power_min * power_max)
    
    best_power = power
    best_T_max = 0
    best_results = {}
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}: Testing P = {power:.2f} W")
        
        # Run simulation at current power
        T_max, results = simulation_func(power=power, **sim_kwargs)
        
        print(f"  T_max = {T_max:.1f} K ({T_max-273.15:.1f} °C)")
        
        # Track best result (closest to target without exceeding too much)
        if abs(T_max - target_temp) < abs(best_T_max - target_temp):
            best_power = power
            best_T_max = T_max
            best_results = results
        
        # Check convergence
        temp_error = abs(T_max - target_temp)
        if temp_error < TEMP_TOLERANCE:
            print(f"  Converged! Error = {temp_error:.2f} K")
            return power, results
        
        # Update search range
        if T_max < target_temp:
            # Need more power
            power_min = power
        else:
            # Need less power
            power_max = power
        
        # Check if range is narrow enough
        power_range_ratio = power_max / power_min
        if power_range_ratio - 1.0 < tolerance:
            print(f"  Power range converged (ratio = {power_range_ratio:.4f})")
            break
        
        # New power guess
        power = math.sqrt(power_min * power_max)
    
    print(f"\nBest result: P = {best_power:.2f} W, T_max = {best_T_max:.1f} K")
    return best_power, best_results


def run_power_sweep(
    powers: List[float],
    simulation_func,
    **sim_kwargs
) -> List[Tuple[float, float, Dict]]:
    """
    Run simulations across a range of power values.
    
    Args:
        powers: List of power values to test
        simulation_func: Simulation function
        **sim_kwargs: Additional simulation arguments
        
    Returns:
        List of (power, T_max, results) tuples
    """
    results = []
    
    for power in powers:
        print(f"\nRunning at P = {power:.1f} W...")
        T_max, sim_results = simulation_func(power=power, **sim_kwargs)
        results.append((power, T_max, sim_results))
        print(f"  T_max = {T_max:.1f} K")
    
    return results


def run_parameter_sweep(
    simulation_func,
    anode_materials: Optional[List[str]] = None,
    substrate_materials: Optional[List[str]] = None,
    anode_thicknesses: Optional[List[float]] = None,
    voltages: Optional[List[float]] = None,
    include_degraded: bool = True,
    output_csv: str = 'power_loading_results.csv',
    output_json: str = 'power_loading_results.json',
) -> List[OptimizationResult]:
    """
    Run full parameter sweep across all combinations.
    
    Args:
        simulation_func: Function that runs simulation and returns (T_max, results)
        anode_materials: List of anode materials to test
        substrate_materials: List of substrate materials to test
        anode_thicknesses: List of anode thicknesses in meters
        voltages: List of voltages in kV
        include_degraded: If True, also test degraded anode condition
        output_csv: CSV file path for results
        output_json: JSON file path for results
        
    Returns:
        List of OptimizationResult objects
    """
    # Use defaults if not specified
    anode_materials = anode_materials or ANODE_MATERIALS
    substrate_materials = substrate_materials or SUBSTRATE_MATERIALS
    anode_thicknesses = anode_thicknesses or ANODE_THICKNESSES
    voltages = voltages or VOLTAGE_RANGE
    
    conditions = ['pristine']
    if include_degraded:
        conditions.append('degraded')
    
    # Calculate total combinations
    total = (len(anode_materials) * len(substrate_materials) * 
             len(anode_thicknesses) * len(voltages) * len(conditions))
    
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Anode materials: {anode_materials}")
    print(f"Substrate materials: {substrate_materials}")
    print(f"Anode thicknesses: {[t*1e6 for t in anode_thicknesses]} μm")
    print(f"Voltages: {voltages} kV")
    print(f"Conditions: {conditions}")
    print(f"Total combinations: {total}")
    print(f"{'='*60}\n")
    
    results: List[OptimizationResult] = []
    completed = 0
    
    start_time = time.time()
    
    for anode_mat in anode_materials:
        for substrate_mat in substrate_materials:
            for thickness in anode_thicknesses:
                for voltage in voltages:
                    for condition in conditions:
                        degraded = (condition == 'degraded')
                        
                        completed += 1
                        print(f"\n[{completed}/{total}] Running: "
                              f"{anode_mat}/{substrate_mat}, "
                              f"t={thickness*1e6:.0f}μm, "
                              f"V={voltage}kV, "
                              f"{condition}")
                        
                        iter_start = time.time()
                        
                        try:
                            # Run optimization for this configuration
                            optimal_power, sim_results = binary_search_power(
                                simulation_func,
                                target_temp=T_TARGET,
                                anode_material=anode_mat,
                                anode_thickness=thickness,
                                substrate_material=substrate_mat,
                                voltage_kV=voltage,
                                degraded=degraded,
                            )
                            
                            # Extract results
                            T_max = sim_results.get('T_max', 0)
                            T_base = sim_results.get('T_base', T_TARGET)
                            n_periods = sim_results.get('n_periods', N_PERIODS_MIN)
                            equilibrium = sim_results.get('equilibrium', False)
                            n_iters = sim_results.get('n_iterations', 0)
                            
                            result = OptimizationResult(
                                anode_material=anode_mat,
                                anode_thickness_um=thickness * 1e6,
                                substrate_material=substrate_mat,
                                voltage_kV=voltage,
                                degraded=degraded,
                                optimal_power_W=optimal_power,
                                max_temperature_K=T_max,
                                max_temperature_C=T_max - 273.15,
                                equilibrium_reached=equilibrium,
                                n_periods_to_equilibrium=n_periods,
                                base_temperature_K=T_base,
                                n_iterations=n_iters,
                                total_time_s=time.time() - iter_start,
                                timestamp=datetime.now().isoformat(),
                            )
                            results.append(result)
                            
                            print(f"  Result: P = {optimal_power:.1f} W, "
                                  f"T_max = {T_max:.0f} K")
                            
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            continue
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful runs: {len(results)}/{total}")
    print(f"{'='*60}")
    
    # Save results
    save_results(results, output_csv, output_json)
    
    return results


def save_results(
    results: List[OptimizationResult],
    csv_path: str,
    json_path: str
) -> None:
    """Save results to CSV and JSON files."""
    
    if not results:
        print("No results to save.")
        return
    
    # Save to CSV
    print(f"\nSaving results to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    
    # Save to JSON
    print(f"Saving results to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print("Results saved.")


def create_summary_table(results: List[OptimizationResult]) -> str:
    """Create a formatted summary table of results."""
    
    if not results:
        return "No results available."
    
    lines = []
    lines.append("=" * 100)
    lines.append("POWER LOADING OPTIMIZATION RESULTS")
    lines.append("=" * 100)
    lines.append(f"{'Anode':>8} {'Substrate':>10} {'Thick(μm)':>10} "
                 f"{'V(kV)':>7} {'Cond':>8} {'P(W)':>8} {'Tmax(°C)':>10}")
    lines.append("-" * 100)
    
    for r in results:
        cond = 'degraded' if r.degraded else 'pristine'
        lines.append(f"{r.anode_material:>8} {r.substrate_material:>10} "
                     f"{r.anode_thickness_um:>10.0f} {r.voltage_kV:>7.0f} "
                     f"{cond:>8} {r.optimal_power_W:>8.1f} "
                     f"{r.max_temperature_C:>10.0f}")
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


# Mock simulation function for testing without COMSOL
def mock_simulation(
    power: float,
    anode_material: str = 'Mo',
    anode_thickness: float = 20e-6,
    substrate_material: str = 'Mo',
    voltage_kV: float = 100,
    degraded: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Mock simulation function for testing the optimizer.
    
    Returns approximate temperature based on power and parameters.
    """
    # Very simplified thermal model for testing
    # T ~ T_ambient + power * thermal_resistance
    
    # Rough thermal resistance based on materials
    k_factors = {'Mo': 1.0, 'W': 0.8, 'Cu': 2.5, 'TZM': 1.1, 'Diamond': 10.0}
    k_anode = k_factors.get(anode_material, 1.0)
    k_substrate = k_factors.get(substrate_material, 1.0)
    
    # Thinner anode = less spreading = higher temperature
    thickness_factor = (20e-6 / anode_thickness) ** 0.3
    
    # Higher voltage = deeper penetration = lower peak temp
    voltage_factor = (100 / voltage_kV) ** 0.2
    
    # Degraded = higher temperature
    degraded_factor = 1.5 if degraded else 1.0
    
    # Combined thermal resistance
    R_thermal = (0.5 / k_anode + 0.5 / k_substrate) * thickness_factor * voltage_factor * degraded_factor
    
    # Temperature rise (very approximate: 10 K per Watt baseline)
    T_ambient = 293.15
    T_max = T_ambient + power * R_thermal * 10
    
    results = {
        'T_max': T_max,
        'T_base': T_ambient + power * R_thermal * 2,
        'equilibrium': True,
        'n_periods': 10,
        'n_iterations': 1,
    }
    
    return T_max, results


# Example usage
if __name__ == '__main__':
    print("Testing power optimizer with mock simulation...")
    
    # Test binary search
    power, results = binary_search_power(
        mock_simulation,
        target_temp=T_TARGET,
        anode_material='W',
        substrate_material='Diamond',
        anode_thickness=50e-6,
    )
    
    print(f"\nOptimal power: {power:.1f} W")
    print(f"Max temperature: {results['T_max']:.1f} K")
