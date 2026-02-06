"""
COMSOL Rotating Anode X-Ray Source Power Loading Simulation

This script uses MPH to control COMSOL 6.4 for time-dependent heat transfer
simulation of a rotating anode X-ray source.

The simulation finds the maximum stable power loading for a given configuration.

Usage:
    python power_loading_sim.py [options]
    
Options:
    --anode-material   : Anode material (Mo, W, W25Rh) [default: Mo]
    --anode-thickness  : Anode thickness in μm [default: 20]
    --substrate        : Substrate material (Cu, TZM, Diamond) [default: Mo]
    --voltage          : Accelerating voltage in kV [default: 100]
    --degraded         : Use degraded anode (50% conductivity) [default: False]
    --power            : Fixed power in W (skips optimization) [default: None]
    --test-mode        : Create model only, don't solve [default: False]
    --save-model       : Save .mph file after creation [default: True]
"""

import argparse
import math
import sys
from typing import Optional, Tuple, Dict, Any

try:
    import mph
except ImportError:
    print("ERROR: mph library not found. Install with: pip install mph")
    print("Note: COMSOL Multiphysics must be installed on this machine.")
    sys.exit(1)

from config import (
    SUBSTRATE_SIZE, ANODE_THICKNESS, BEAM_SIGMA_X, BEAM_SIGMA_Y,
    TRACK_LENGTH, PERIOD, T_TARGET, T_AMBIENT, DEFAULT_VOLTAGE_KV,
    N_PERIODS_MIN, N_PERIODS_MAX, DT_BEAM_ON, DT_BEAM_OFF,
    MODEL_NAME
)
from materials import get_material_properties, get_comsol_material_definition
from gruen_function import gruen_range, get_simple_depth_expression


class RotatingAnodeSimulation:
    """
    COMSOL simulation for rotating anode heat transfer analysis.
    """
    
    def __init__(
        self,
        anode_material: str = 'Mo',
        anode_thickness: float = ANODE_THICKNESS,
        substrate_material: str = 'Mo',
        voltage_kV: float = DEFAULT_VOLTAGE_KV,
        degraded: bool = False,
        power: float = 100.0,
    ):
        """
        Initialize simulation parameters.
        
        Args:
            anode_material: Material for anode layer ('Mo', 'W', 'W25Rh')
            anode_thickness: Thickness of anode layer in meters
            substrate_material: Material for substrate ('Mo', 'Cu', 'TZM', 'Diamond')
            voltage_kV: Accelerating voltage in kV
            degraded: If True, use degraded thermal conductivity (50%)
            power: Initial beam power in Watts
        """
        self.anode_material = anode_material
        self.anode_thickness = anode_thickness
        self.substrate_material = substrate_material
        self.voltage_kV = voltage_kV
        self.degraded = degraded
        self.power = power
        
        # Calculate derived parameters
        self.beam_on_time = TRACK_LENGTH / self._calculate_velocity()
        
        # COMSOL objects
        self.client = None
        self.model = None
        self.java = None
        
    def _calculate_velocity(self) -> float:
        """Calculate beam velocity from track length and desired duty cycle."""
        # Assuming ~10% duty cycle (beam on for 10% of period)
        # Adjust this based on actual rotation speed
        duty_cycle = 0.1
        t_on = PERIOD * duty_cycle
        return TRACK_LENGTH / t_on
    
    def connect(self):
        """Connect to COMSOL server."""
        print("Starting COMSOL client...")
        self.client = mph.start()
        print(f"Connected to COMSOL {self.client.version}")
        
    def create_model(self) -> None:
        """Create the COMSOL model with all physics."""
        print("Creating model...")
        
        # Create new model
        pymodel = self.client.create(MODEL_NAME)
        self.model = pymodel
        self.java = pymodel.java
        
        # Set up model structure
        self._create_parameters()
        self._create_geometry()
        self._create_materials()
        self._create_physics()
        self._create_mesh()
        self._create_study()
        
        print("Model creation complete.")
        
    def _create_parameters(self) -> None:
        """Define global parameters."""
        print("  Defining parameters...")
        
        params = self.java.param()
        
        # Geometry parameters
        params.set("L_sub", f"{SUBSTRATE_SIZE[0]}[m]", "Substrate length (x)")
        params.set("W_sub", f"{SUBSTRATE_SIZE[1]}[m]", "Substrate width (y)")
        params.set("H_sub", f"{SUBSTRATE_SIZE[2]}[m]", "Substrate height (z)")
        params.set("t_anode", f"{self.anode_thickness}[m]", "Anode layer thickness")
        
        # Beam parameters
        params.set("sigma_x", f"{BEAM_SIGMA_X}[m]", "Beam sigma in x")
        params.set("sigma_y", f"{BEAM_SIGMA_Y}[m]", "Beam sigma in y")
        params.set("L_track", f"{TRACK_LENGTH}[m]", "Beam track length")
        params.set("T_period", f"{PERIOD}[s]", "Rotation period")
        
        # Calculate and set derived parameters
        t_on = self.beam_on_time
        velocity = TRACK_LENGTH / t_on
        params.set("t_on", f"{t_on}[s]", "Beam on time per period")
        params.set("v_beam", f"{velocity}[m/s]", "Beam velocity")
        
        # Power and thermal
        params.set("P0", f"{self.power}[W]", "Total beam power")
        params.set("T_ambient", f"{T_AMBIENT}[K]", "Ambient temperature")
        
        # Gruen function parameters
        gruen_params = get_simple_depth_expression(self.voltage_kV, self.anode_material)
        params.set("z_peak", f"{gruen_params['z_peak']}[m]", "Gruen peak depth")
        params.set("sigma_z", f"{gruen_params['sigma_z']}[m]", "Gruen depth sigma")
        params.set("R_gruen", f"{gruen_params['R_gruen']}[m]", "Gruen range")
        params.set("norm_z", f"{gruen_params['norm_z']}[1/m]", "Depth normalization")
        
    def _create_geometry(self) -> None:
        """Create 3D geometry: substrate block + anode layer."""
        print("  Creating geometry...")
        
        model = self.java
        
        # Create component
        model.component().create("comp1", True)
        
        # Create 3D geometry with union action (creates single domain from overlapping objects)
        geom = model.component("comp1").geom().create("geom1", 3)
        
        # Substrate block (bottom)
        blk1 = geom.create("substrate", "Block")
        blk1.set("size", ["L_sub", "W_sub", "H_sub"])
        blk1.set("pos", ["-L_sub/2", "-W_sub/2", "0"])
        blk1.label("Substrate")
        
        # Anode layer (on top of substrate)
        blk2 = geom.create("anode", "Block")
        blk2.set("size", ["L_sub", "W_sub", "t_anode"])
        blk2.set("pos", ["-L_sub/2", "-W_sub/2", "H_sub"])
        blk2.label("Anode")
        
        # Run and finalize geometry (auto-creates union/assembly)
        geom.run()
        
    def _create_materials(self) -> None:
        """Create materials with temperature-dependent properties."""
        print("  Creating materials...")
        
        model = self.java
        comp = model.component("comp1")
        
        # Create material container
        print("    Getting material container...")
        mat_cont = comp.material()
        
        # Substrate material
        print(f"    Creating substrate material ({self.substrate_material})...")
        sub_props = get_comsol_material_definition(self.substrate_material)
        print(f"      rho={sub_props['rho']}, k={sub_props['k']}, Cp={sub_props['Cp']}")
        mat_sub = mat_cont.create("mat_substrate", "Common")
        mat_sub.label("Substrate Material")
        mat_sub.propertyGroup("def").set("density", sub_props['rho'])
        mat_sub.propertyGroup("def").set("thermalconductivity", sub_props['k'])
        mat_sub.propertyGroup("def").set("heatcapacity", sub_props['Cp'])
        print("    Assigning substrate to domain 1...")
        mat_sub.selection().set([1])
        
        # Anode material
        print(f"    Creating anode material ({self.anode_material}, degraded={self.degraded})...")
        anode_props = get_comsol_material_definition(self.anode_material, self.degraded)
        print(f"      rho={anode_props['rho']}, k={anode_props['k']}, Cp={anode_props['Cp']}")
        mat_anode = mat_cont.create("mat_anode", "Common")
        mat_anode.label("Anode Material")
        mat_anode.propertyGroup("def").set("density", anode_props['rho'])
        mat_anode.propertyGroup("def").set("thermalconductivity", anode_props['k'])
        mat_anode.propertyGroup("def").set("heatcapacity", anode_props['Cp'])
        print("    Assigning anode to domain 2...")
        mat_anode.selection().set([2])
        print("    Materials created successfully.")
        
    def _create_physics(self) -> None:
        """Set up Heat Transfer physics interface."""
        print("  Setting up physics...")
        
        model = self.java
        comp = model.component("comp1")
        
        # Create Heat Transfer in Solids physics
        print("    Creating Heat Transfer physics...")
        physics = comp.physics().create("ht", "HeatTransfer", "geom1")
        physics.label("Heat Transfer")
        
        # Initial temperature
        print("    Setting initial temperature...")
        physics.feature("init1").set("Tinit", "T_ambient")
        
        # Fixed temperature boundary condition (sides and bottom)
        print("    Creating temperature boundary condition...")
        temp_bc = physics.create("temp1", "TemperatureBoundary", 2)
        temp_bc.label("Fixed Temperature BCs")
        temp_bc.set("T0", "T_ambient")
        print("    Setting boundary selection [1,2,3,4,5]...")
        temp_bc.selection().set([1, 2, 3, 4, 5])
        
        # Heat source in anode region
        print("    Creating heat source...")
        heat_src = physics.create("hs1", "HeatSource", 3)
        heat_src.label("Electron Beam Heat Source")
        print("    Setting heat source domain [2]...")
        heat_src.selection().set([2])
        
        # Define heat source expression
        print("    Building heat source expression...")
        heat_expr = self._build_heat_source_expression()
        print(f"    Heat source expression length: {len(heat_expr)} chars")
        heat_src.set("Q0", heat_expr)
        print("    Physics created successfully.")
        
    def _build_heat_source_expression(self) -> str:
        """
        Build the volumetric heat source expression for COMSOL.
        
        Q(x,y,z,t) = P0 * G_xy(x - x_beam(t), y) * G_z(z_local) * pulse(t)
        
        where:
        - G_xy is 2D Gaussian
        - G_z is Gruen depth profile
        - x_beam(t) moves from -L_track/2 to L_track/2
        - pulse(t) is 1 during beam-on, 0 during beam-off
        - z_local = (H_sub + t_anode) - z (depth from anode surface)
        """
        
        # Time within current period
        t_mod = "mod(t, T_period)"
        
        # Beam x position (linear motion during beam-on time)
        # x_beam = -L_track/2 + (t_mod / t_on) * L_track  for t_mod < t_on
        x_beam = f"(-L_track/2 + ({t_mod}/t_on)*L_track)"
        
        # Pulse function: 1 if t_mod < t_on, else 0
        pulse = f"({t_mod} < t_on)"
        
        # 2D Gaussian in x and y
        # G_xy = (1 / (2*pi*sigma_x*sigma_y)) * exp(-((x-x_beam)^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)))
        norm_xy = "(1/(2*pi*sigma_x*sigma_y))"
        gauss_xy = f"{norm_xy}*exp(-((x-{x_beam})^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)))"
        
        # Depth from anode surface (z=0 at bottom, so anode top is at H_sub + t_anode)
        z_local = "(H_sub + t_anode - z)"
        
        # Gruen depth profile (from parameters defined earlier)
        gauss_z = f"norm_z*({z_local}/z_peak)*exp(-(({z_local}-z_peak)^2)/(2*sigma_z^2))"
        valid_z = f"(({z_local} > 0) * ({z_local} < R_gruen))"
        
        # Complete heat source expression
        Q_expr = f"P0 * {gauss_xy} * {gauss_z} * {valid_z} * {pulse}"
        
        return Q_expr
        
    def _create_mesh(self) -> None:
        """Create adaptive mesh with refinement in anode region."""
        print("  Creating mesh...")
        
        model = self.java
        comp = model.component("comp1")
        
        print("    Creating mesh container...")
        mesh = comp.mesh().create("mesh1")
        
        # For debugging: use very coarse mesh everywhere
        print("    Setting global mesh size (coarse for debugging)...")
        mesh.feature("size").set("hauto", "7")  # Very coarse for fast debugging
        
        # Free tetrahedral mesh
        print("    Creating free tetrahedral mesh...")
        ftet = mesh.create("ftet1", "FreeTet")
        
        # Build mesh
        print("    Building mesh...")
        try:
            mesh.run()
            print("    Mesh built successfully.")
        except Exception as e:
            print(f"    Mesh build failed: {e}")
        
    def _create_study(self) -> None:
        """Create time-dependent study with events."""
        print("  Setting up study...")
        
        model = self.java
        
        # Create study
        study = model.study().create("std1")
        study.label("Time Dependent Study")
        
        # Time-dependent study step
        step = study.create("time1", "Transient")
        step.label("Time Dependent")
        
        # Time range: run for N_PERIODS_MIN periods
        total_time = N_PERIODS_MIN * PERIOD
        step.set("tlist", f"range(0, {DT_BEAM_ON}, {total_time})")
        
        # Set up solver configuration for events
        # Events will be added for beam on/off transitions
        sol = model.sol().create("sol1")
        sol.study("std1")
        sol.attach("std1")
        
        # Time-dependent solver
        sol.create("st1", "StudyStep")
        sol.feature("st1").set("study", "std1")
        sol.feature("st1").set("studystep", "time1")
        
        sol.create("v1", "Variables")
        sol.feature("v1").set("control", "time1")
        
        sol.create("t1", "Time")
        sol.feature("t1").set("control", "time1")
        sol.feature("t1").set("tlist", f"range(0, {DT_BEAM_ON}, {total_time})")
        
        # Add events for time step control
        self._add_time_step_events(sol)
        
        # Configure solver settings
        sol.feature("t1").set("maxorder", 2)
        sol.feature("t1").set("estrat", "exclude")
        sol.feature("t1").set("tstepsbdf", "free")
        
    def _add_time_step_events(self, sol) -> None:
        """
        Configure time stepping for the simulation.
        
        For now, use simple uniform time stepping for debugging.
        Complex event-based stepping can be added later.
        """
        print("    Configuring time stepping...")
        
        # For debugging: just use uniform coarse time steps
        # Skip complex event system until basic simulation works
        n_periods = N_PERIODS_MIN
        total_time = n_periods * PERIOD
        
        # Use simple uniform output times (coarse for debugging)
        dt_output = PERIOD / 10  # 10 points per period for debugging
        times_str = f"range(0, {dt_output}, {total_time})"
        sol.feature("t1").set("tlist", times_str)
        
        print(f"    Time stepping: {n_periods} periods, dt={dt_output*1e6:.1f} μs")
        
    def solve(self) -> None:
        """Run the simulation."""
        print("Running simulation...")
        self.model.solve()
        print("Simulation complete.")
        
    def get_results(self) -> Dict[str, Any]:
        """Extract results from the simulation."""
        print("Extracting results...")
        
        results = {}
        
        # Get global maximum temperature over time
        T_max_history = self.model.evaluate(
            'maxop1(T)',  # Maximum operator over domain
            'dataset', 'dset1',
            'refine', 2
        )
        results['T_max_history'] = T_max_history
        
        # Get temperature at specific probe points
        # (Additional evaluation for depth profile would go here)
        
        return results
        
    def save_model(self, filename: Optional[str] = None) -> None:
        """Save the COMSOL model to file."""
        if filename is None:
            filename = f"{MODEL_NAME}_{self.anode_material}_{int(self.anode_thickness*1e6)}um.mph"
        print(f"Saving model to {filename}...")
        self.model.save(filename)
        
    def close(self) -> None:
        """Clean up COMSOL resources."""
        if self.client:
            self.client.clear()
            print("COMSOL client closed.")


def run_single_simulation(
    anode_material: str = 'Mo',
    anode_thickness: float = ANODE_THICKNESS,
    substrate_material: str = 'Mo',
    voltage_kV: float = 100,
    degraded: bool = False,
    power: float = 100.0,
    save_model: bool = True,
    test_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run a single simulation with given parameters.
    
    Args:
        anode_material: Anode material name
        anode_thickness: Anode thickness in meters
        substrate_material: Substrate material name
        voltage_kV: Accelerating voltage
        degraded: Use degraded anode properties
        power: Beam power in Watts
        save_model: Save .mph file
        test_mode: Only create model, don't solve
        
    Returns:
        Results dictionary if solved, None if test_mode
    """
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
        
        if save_model:
            sim.save_model()
        
        if test_mode:
            print("Test mode: Model created successfully. Skipping solve.")
            return None
            
        sim.solve()
        results = sim.get_results()
        return results
        
    finally:
        sim.close()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="COMSOL Rotating Anode Power Loading Simulation"
    )
    
    parser.add_argument(
        '--anode-material', type=str, default='Mo',
        choices=['Mo', 'W', 'W25Rh'],
        help='Anode material (default: Mo)'
    )
    parser.add_argument(
        '--anode-thickness', type=float, default=20,
        help='Anode thickness in μm (default: 20)'
    )
    parser.add_argument(
        '--substrate', type=str, default='Mo',
        choices=['Mo', 'Cu', 'TZM', 'Diamond'],
        help='Substrate material (default: Mo)'
    )
    parser.add_argument(
        '--voltage', type=float, default=100,
        help='Accelerating voltage in kV (default: 100)'
    )
    parser.add_argument(
        '--degraded', action='store_true',
        help='Use degraded anode (50%% conductivity)'
    )
    parser.add_argument(
        '--power', type=float, default=100,
        help='Beam power in Watts (default: 100)'
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Create model only, do not solve'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save .mph file'
    )
    
    args = parser.parse_args()
    
    # Convert thickness from μm to m
    thickness_m = args.anode_thickness * 1e-6
    
    results = run_single_simulation(
        anode_material=args.anode_material,
        anode_thickness=thickness_m,
        substrate_material=args.substrate,
        voltage_kV=args.voltage,
        degraded=args.degraded,
        power=args.power,
        save_model=not args.no_save,
        test_mode=args.test_mode,
    )
    
    if results:
        print("\n=== Results ===")
        T_max = max(results.get('T_max_history', [0]))
        print(f"Maximum temperature: {T_max:.1f} K ({T_max - 273.15:.1f} °C)")


if __name__ == '__main__':
    main()
