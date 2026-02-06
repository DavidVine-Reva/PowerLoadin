# Rotating Anode X-Ray Source Power Loading Simulation

Python scripts to control COMSOL 6.4 via [MPH](https://mph.readthedocs.io/) for finding the maximum stable power loading in rotating anode X-ray sources.

## Overview

The simulation models a pulsed electron beam heating a layered target (anode + substrate) and finds the maximum power loading that achieves a target temperature of 1500°C at thermal equilibrium.

## Installation

1. **COMSOL**: Install COMSOL Multiphysics 6.4 on the target machine
2. **MPH Library**: 
   ```bash
   pip install mph
   ```

## Files

| File | Description |
|------|-------------|
| `power_loading_sim.py` | Main COMSOL simulation driver |
| `power_optimizer.py` | Binary search and parameter sweep |
| `run_sweep.py` | CLI for running optimization sweeps |
| `config.py` | Simulation parameters |
| `materials.py` | Temperature-dependent material properties |
| `gruen_function.py` | Electron energy deposition model |

## Quick Start

### Single Simulation
```bash
# Create model with Mo anode, 20μm thick, at 100kV
python power_loading_sim.py --anode-material Mo --anode-thickness 20 --voltage 100

# Test mode (create model only, no solve)
python power_loading_sim.py --test-mode
```

### Power Optimization
```bash
# Find optimal power for single configuration
python run_sweep.py --quick --use-mock  # Test without COMSOL

# Full parameter sweep
python run_sweep.py --output-dir ./results
```

### Custom Sweep
```bash
# Specific materials and voltages
python run_sweep.py \
    --materials W,Mo \
    --substrates Diamond,TZM \
    --voltages 70,100,130 \
    --thicknesses 20,50
```

## Physics Model

### Geometry
- **Substrate**: 10×10×1 mm block
- **Anode**: 10×10×t mm layer on top (t = 10, 50, or 100 μm)

### Heat Source
The electron beam is modeled as a volumetric heat source:

```
Q(x,y,z,t) = P₀ × Gxy(x - xbeam(t), y) × Gz(z) × pulse(t)
```

Where:
- **Gxy**: 2D Gaussian (70×700 μm FWHM spot)
- **Gz**: Gruen depth profile (voltage-dependent)
- **xbeam(t)**: Linear motion from -L/2 to L/2
- **pulse(t)**: On during beam sweep, off until next rotation

### Rotation Emulation
- Track length L = 3 mm
- Period T = 1 ms
- Beam resets to -L/2 after each period

### Boundary Conditions
- Sides and bottom: Fixed at room temperature (293.15 K)
- Top surface: Continuity

## Materials

Temperature-dependent properties for:
- **Anode**: Mo, W, W25Rh
- **Substrate**: Mo, Cu, TZM, Diamond

Degraded mode reduces thermal conductivity by 50%.

## Output

Results saved as CSV and JSON with columns:
- Optimal power (W)
- Max temperature (K, °C)
- Equilibrium status
- Configuration parameters

## Configuration

Edit `config.py` to modify:
- Geometry dimensions
- Beam parameters
- Target temperature
- Solver settings

## Requirements

- Python 3.8+
- COMSOL Multiphysics 6.4
- JPype1 (installed with mph)
