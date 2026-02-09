"""
Module to load and interpolate Modified Gruen fit parameters from JSON.

The parameters correspond to the Modified Gruen depth profile:
D(z) = (z/alpha)^beta * exp(-z/alpha)

We need to provide alpha (in meters) and beta for any given voltage.
The JSON file contains data for discrete voltages (30, 40, ..., 160 kV).
We interpolate linearly between these points.
"""

import json
import os
import math
from typing import Tuple, Dict
import numpy as np

# Path to the JSON file (assumed to be in the same directory)
FIT_PARAMS_FILE = os.path.join(os.path.dirname(__file__), 'fit_parameters_Mo.json')

class FitParameterLoader:
    def __init__(self, filename: str = FIT_PARAMS_FILE):
        self.filename = filename
        self.data = self._load_data()
        self.voltages = sorted(self.data.keys())
        
    def _load_data(self) -> Dict[float, Dict[str, float]]:
        """Load JSON and organize by voltage."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Fit parameters file not found: {self.filename}")
            
        with open(self.filename, 'r') as f:
            content = json.load(f)
            
        results = content.get('results', [])
        data_by_voltage = {}
        
        for entry in results:
            v = float(entry['voltage_kV'])
            params = entry['depth']
            data_by_voltage[v] = {
                'alpha_um': params['alpha_um'],
                'beta': params['beta']
            }
            
        return data_by_voltage
        
    def get_params(self, voltage_kV: float) -> Tuple[float, float, float]:
        """
        Get interpolated alpha (m), beta, and normalization factor for a given voltage.
        
        Args:
            voltage_kV: Accelerating voltage in kV.
            
        Returns:
            Tuple (alpha_m, beta, norm_factor)
            
            norm_factor = 1.0 / (alpha_m * gamma(beta + 1))
            This ensures integral of D(z) from 0 to infinity is 1.
        """
        # Clamp voltage to range
        v_min = self.voltages[0]
        v_max = self.voltages[-1]
        
        if voltage_kV <= v_min:
            res = self.data[v_min]
            alpha_um = res['alpha_um']
            beta = res['beta']
        elif voltage_kV >= v_max:
            res = self.data[v_max]
            alpha_um = res['alpha_um']
            beta = res['beta']
        else:
            # Linear interpolation
            # Find neighbors
            idx = np.searchsorted(self.voltages, voltage_kV)
            v_high = self.voltages[idx]
            v_low = self.voltages[idx - 1]
            
            # Fraction
            f = (voltage_kV - v_low) / (v_high - v_low)
            
            p_low = self.data[v_low]
            p_high = self.data[v_high]
            
            alpha_um = p_low['alpha_um'] + f * (p_high['alpha_um'] - p_low['alpha_um'])
            beta = p_low['beta'] + f * (p_high['beta'] - p_low['beta'])
            
        # Convert alpha to meters
        alpha_m = alpha_um * 1e-6
        
        # Calculate normalization factor for (z/alpha)^beta * exp(-z/alpha)
        # Integral_0^inf (x/a)^b * exp(-x/a) dx = a * gamma(b+1)
        # We want integral to be 1, so norm = 1 / (a * gamma(b+1))
        
        norm = 1.0 / (alpha_m * math.gamma(beta + 1))
        
        return alpha_m, beta, norm

# Global instance for easy access
_loader = None

def get_fit_params(voltage_kV: float) -> Tuple[float, float, float]:
    """
    Get fit parameters for Mo at given voltage.
    Returns (alpha_m, beta, norm_factor).
    """
    global _loader
    if _loader is None:
        _loader = FitParameterLoader()
        
    return _loader.get_params(voltage_kV)

if __name__ == "__main__":
    # Test
    print("Testing fit parameter interpolation...")
    for v in [30, 35, 100, 160]:
        a, b, n = get_fit_params(v)
        print(f"Voltage {v} kV: alpha={a*1e6:.4f} um, beta={b:.4f}, norm={n:.4e}")
