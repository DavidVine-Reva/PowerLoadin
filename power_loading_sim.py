"""
COMSOL Rotating Anode X-Ray Source Power Loading Simulation

This script uses MPH to control COMSOL 6.4 for time-dependent heat transfer
simulation of a rotating anode X-ray source.

The simulation finds the maximum stable power loading for a given configuration.

Usage:
    python power_loading_sim.py [options]

Options:
    --anode-material   : Anode material (Mo, W, W25Rh) [default: Mo]
    --anode-thickness  : Anode thickness in um [default: 20]
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
import numpy as np
import matplotlib.pyplot as plt
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
    MODEL_NAME, VELOCITY, EQUILIBRIUM_TOLERANCE,
    SUPERGAUSS_ORDER, BEAM_TRACK_NSIGMA,
)
from materials import (get_material_properties, get_comsol_material_definition,
                       create_comsol_interpolation_functions,
                       get_interpolation_materials_used)
from gruen_function import gruen_range, get_simple_depth_expression
try:
    from fit_loader import get_fit_params
except ImportError:
    get_fit_params = None


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
        self.anode_material = anode_material
        self.anode_thickness = anode_thickness
        self.substrate_material = substrate_material
        self.voltage_kV = voltage_kV
        self.degraded = degraded
        self.power = power

        # Derived parameters
        self.beam_on_time = TRACK_LENGTH / VELOCITY

        # COMSOL objects
        self.client = None
        self.model = None
        self.java = None


    def connect(self):
        """Connect to COMSOL server."""
        print("Starting COMSOL client...")
        self.client = mph.start()
        print(f"Connected to COMSOL {self.client.version}")

    def create_model(self) -> None:
        """Create the COMSOL model with all physics."""
        print("Creating model...")

        pymodel = self.client.create(MODEL_NAME)
        self.model = pymodel
        self.java = pymodel.java

        self._create_parameters()
        self._create_geometry()
        self._create_selections()
        self._create_materials()
        self._create_physics()
        self._create_mesh()
        self._create_study()
        self._create_comsol_results()

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

        # Derived timing parameters
        t_on = self.beam_on_time
        params.set("t_on", f"{t_on}[s]", "Beam on time per period")
        params.set("v_beam", f"{VELOCITY}[m/s]", "Beam velocity")

        # Power and thermal
        params.set("P0", f"{self.power}[W]", "Total beam power")
        params.set("T_ambient", f"{T_AMBIENT}[K]", "Ambient temperature")

        # Super-Gaussian normalization
        # For 2D separable super-Gaussian of order n:
        #   G(x,y) = N * exp(-(x/sx)^n - (y/sy)^n)
        #   N = n^2 / (4 * sx * sy * Gamma(1/n)^2)
        # For n=4: N = 16 / (4 * sx * sy * Gamma(0.25)^2)
        #            = 4 / (sx * sy * Gamma(0.25)^2)
        gamma_quarter = math.gamma(0.25)  # ~3.6256
        n = SUPERGAUSS_ORDER
        norm_xy = (n * n) / (4.0 * BEAM_SIGMA_X * BEAM_SIGMA_Y
                             * math.gamma(1.0 / n) ** 2)
        params.set("norm_xy", f"{norm_xy}[1/m^2]",
                   f"2D super-Gaussian (order {n}) normalization")

        # Depth profile parameters (Gruen function / Fit)
        use_fit = (self.anode_material == 'Mo') and (get_fit_params is not None)

        if use_fit:
            print(f"  Using FIT parameters for {self.anode_material} "
                  f"at {self.voltage_kV} kV")
            alpha_m, beta, norm_fit = get_fit_params(self.voltage_kV)
            params.set("alpha_z", f"{alpha_m:.6e}[m]", "Modified Gruen alpha")
            params.set("beta_z", f"{beta:.6f}", "Modified Gruen beta")
            params.set("norm_z", f"{norm_fit:.6e}[1/m]", "Depth normalization")
            params.set("R_gruen", f"{10*alpha_m:.6e}[m]",
                       "Effective range cutoff")
            params.set("use_modified_gruen", "1",
                       "Flag for Modified Gruen profile")
        else:
            print(f"  Using STANDARD Gruen parameters for {self.anode_material}")
            gruen_params = get_simple_depth_expression(
                self.voltage_kV, self.anode_material)
            params.set("z_peak", f"{gruen_params['z_peak']}[m]",
                       "Gruen peak depth")
            params.set("sigma_z", f"{gruen_params['sigma_z']}[m]",
                       "Gruen depth sigma")
            params.set("R_gruen", f"{gruen_params['R_gruen']}[m]",
                       "Gruen range")
            params.set("norm_z", f"{gruen_params['norm_z']}[1/m]",
                       "Depth normalization")
            params.set("use_modified_gruen", "0",
                       "Flag for Modified Gruen profile")

    def _create_geometry(self) -> None:
        """Create 3D geometry: substrate + anode + beam track region."""
        print("  Creating geometry...")

        model = self.java

        # Create component
        model.component().create("comp1", True)

        # Create 3D geometry
        geom = model.component("comp1").geom().create("geom1", 3)

        # --- Cumulative selections for domain identification ---
        geom.selection().create("csel_substrate", "CumulativeSelection")
        geom.selection("csel_substrate").label("Substrate Domains")

        geom.selection().create("csel_beam_track", "CumulativeSelection")
        geom.selection("csel_beam_track").label("Beam Track Domains")

        geom.selection().create("csel_anode_all", "CumulativeSelection")
        geom.selection("csel_anode_all").label("All Anode Domains")

        # --- Beam track region size ---
        # For super-Gaussian order 4, amplitude at BEAM_TRACK_NSIGMA*sigma
        # is exp(-(NSIGMA)^4) which is negligible
        ns = BEAM_TRACK_NSIGMA
        track_half_x = f"(L_track/2 + {ns}*sigma_x)"
        track_half_y = f"{ns}*sigma_y"

        # --- Substrate block ---
        blk1 = geom.create("substrate", "Block")
        blk1.set("size", ["L_sub", "W_sub", "H_sub"])
        blk1.set("pos", ["-L_sub/2", "-W_sub/2", "0"])
        blk1.set("contributeto", "csel_substrate")
        blk1.label("Substrate")

        # --- Beam track region in anode (fine mesh region) ---
        blk_track = geom.create("beam_track", "Block")
        blk_track.set("size", [f"2*{track_half_x}",
                                f"2*{track_half_y}",
                                "t_anode"])
        blk_track.set("pos", [f"-{track_half_x}",
                               f"-{track_half_y}",
                               "H_sub"])
        blk_track.set("contributeto", "csel_beam_track")
        blk_track.label("Beam Track Region")

        # --- Full anode layer (on top of substrate) ---
        # After auto-partition, this creates the outer anode domain.
        # The overlap with beam_track creates the beam track domain.
        # csel_anode_all captures both the outer anode and the overlap.
        blk2 = geom.create("anode", "Block")
        blk2.set("size", ["L_sub", "W_sub", "t_anode"])
        blk2.set("pos", ["-L_sub/2", "-W_sub/2", "H_sub"])
        blk2.set("contributeto", "csel_anode_all")
        blk2.label("Anode")

        # Finalize geometry (auto form-union partitions overlapping blocks)
        geom.run()
        print("    Geometry built with cumulative selections")

    def _create_selections(self) -> None:
        """Create boundary selections for BCs using Box selections.

        Selects bottom face and four side faces of the assembly for
        fixed-temperature boundary conditions. The top face of the anode
        is left adiabatic (natural BC).
        """
        print("  Creating boundary selections...")

        comp = self.java.component("comp1")

        # Precompute geometric limits
        L = SUBSTRATE_SIZE[0]
        W = SUBSTRATE_SIZE[1]
        H = SUBSTRATE_SIZE[2]
        t_a = self.anode_thickness
        eps = 1e-9  # small tolerance for box boundary inclusion

        z_top = H + t_a  # top of anode

        # --- Bottom face (z = 0) ---
        sel = comp.selection().create("sel_bottom", "Box")
        sel.set("entitydim", 2)
        sel.set("condition", "inside")
        sel.set("xmin", -L/2 - eps)
        sel.set("xmax",  L/2 + eps)
        sel.set("ymin", -W/2 - eps)
        sel.set("ymax",  W/2 + eps)
        sel.set("zmin", -eps)
        sel.set("zmax",  eps)
        sel.label("Bottom Face")

        # --- Side face x = -L/2 ---
        sel = comp.selection().create("sel_xn", "Box")
        sel.set("entitydim", 2)
        sel.set("condition", "inside")
        sel.set("xmin", -L/2 - eps)
        sel.set("xmax", -L/2 + eps)
        sel.set("ymin", -W/2 - eps)
        sel.set("ymax",  W/2 + eps)
        sel.set("zmin", -eps)
        sel.set("zmax",  z_top + eps)
        sel.label("Side Face X-")

        # --- Side face x = +L/2 ---
        sel = comp.selection().create("sel_xp", "Box")
        sel.set("entitydim", 2)
        sel.set("condition", "inside")
        sel.set("xmin",  L/2 - eps)
        sel.set("xmax",  L/2 + eps)
        sel.set("ymin", -W/2 - eps)
        sel.set("ymax",  W/2 + eps)
        sel.set("zmin", -eps)
        sel.set("zmax",  z_top + eps)
        sel.label("Side Face X+")

        # --- Side face y = -W/2 ---
        sel = comp.selection().create("sel_yn", "Box")
        sel.set("entitydim", 2)
        sel.set("condition", "inside")
        sel.set("xmin", -L/2 - eps)
        sel.set("xmax",  L/2 + eps)
        sel.set("ymin", -W/2 - eps)
        sel.set("ymax", -W/2 + eps)
        sel.set("zmin", -eps)
        sel.set("zmax",  z_top + eps)
        sel.label("Side Face Y-")

        # --- Side face y = +W/2 ---
        sel = comp.selection().create("sel_yp", "Box")
        sel.set("entitydim", 2)
        sel.set("condition", "inside")
        sel.set("xmin", -L/2 - eps)
        sel.set("xmax",  L/2 + eps)
        sel.set("ymin",  W/2 - eps)
        sel.set("ymax",  W/2 + eps)
        sel.set("zmin", -eps)
        sel.set("zmax",  z_top + eps)
        sel.label("Side Face Y+")

        # --- Union of bottom + sides for fixed-temperature BC ---
        sel_bc = comp.selection().create("sel_fixed_temp", "Union")
        sel_bc.set("entitydim", 2)
        sel_bc.set("input", ["sel_bottom", "sel_xn", "sel_xp",
                              "sel_yn", "sel_yp"])
        sel_bc.label("Fixed Temperature Boundaries")

        print("    Boundary selections created (bottom + 4 sides)")

    def _create_materials(self) -> None:
        """Create materials with temperature-dependent properties.

        For materials with tabulated data (Mo, W, TZM, Diamond), creates
        COMSOL interpolation functions first, then references them.
        """
        print("  Creating materials...")

        model = self.java
        comp = model.component("comp1")

        # Create interpolation functions for materials that need them
        interp_mats = get_interpolation_materials_used(
            self.anode_material, self.substrate_material)
        for mat in interp_mats:
            create_comsol_interpolation_functions(model, mat)
            print(f"    Created interpolation functions for {mat}")

        mat_cont = comp.material()

        # Substrate material
        sub_props = get_comsol_material_definition(self.substrate_material)
        mat_sub = mat_cont.create("mat_substrate", "Common")
        mat_sub.label("Substrate Material")
        mat_sub.propertyGroup("def").set("density", sub_props['rho'])
        mat_sub.propertyGroup("def").set("thermalconductivity", sub_props['k'])
        mat_sub.propertyGroup("def").set("heatcapacity", sub_props['Cp'])
        mat_sub.selection().named("geom1_csel_substrate")
        print(f"    Substrate ({self.substrate_material}) -> csel_substrate")

        # Anode material (covers all anode domains including beam track)
        anode_props = get_comsol_material_definition(
            self.anode_material, self.degraded)
        mat_anode = mat_cont.create("mat_anode", "Common")
        mat_anode.label("Anode Material")
        mat_anode.propertyGroup("def").set("density", anode_props['rho'])
        mat_anode.propertyGroup("def").set("thermalconductivity",
                                           anode_props['k'])
        mat_anode.propertyGroup("def").set("heatcapacity", anode_props['Cp'])
        mat_anode.selection().named("geom1_csel_anode_all")
        print(f"    Anode ({self.anode_material}, degraded={self.degraded})"
              f" -> csel_anode_all")

    def _create_physics(self) -> None:
        """Set up Heat Transfer physics interface."""
        print("  Setting up physics...")

        model = self.java
        comp = model.component("comp1")

        # Create Heat Transfer in Solids
        physics = comp.physics().create("ht", "HeatTransfer", "geom1")
        physics.label("Heat Transfer")

        # Initial temperature
        physics.feature("init1").set("Tinit", "T_ambient")

        # Fixed temperature BC on bottom + sides (using named selection)
        temp_bc = physics.create("temp1", "TemperatureBoundary", 2)
        temp_bc.label("Fixed Temperature BCs")
        temp_bc.set("T0", "T_ambient")
        temp_bc.selection().named("comp1_sel_fixed_temp")
        print("    Temperature BC -> sel_fixed_temp (bottom + sides)")

        # Volumetric heat source in all anode domains
        heat_src = physics.create("hs1", "HeatSource", 3)
        heat_src.label("Electron Beam Heat Source")
        heat_src.selection().named("geom1_csel_anode_all")

        heat_expr = self._build_heat_source_expression()
        heat_src.set("Q0", heat_expr)
        print(f"    Heat source -> csel_anode_all "
              f"(expression: {len(heat_expr)} chars)")

    def _build_heat_source_expression(self) -> str:
        """
        Build the volumetric heat source expression for COMSOL.

        Q(x,y,z,t) = P0 * G_xy(x - x_beam(t), y) * G_z(z_local) * beam_state

        where:
        - G_xy is 2D super-Gaussian of order 4
        - G_z is depth profile (Modified Gruen or Standard Gruen)
        - x_beam(t) moves linearly during beam-on
        - beam_state is discrete state (1 = on, 0 = off) via Events
        - z_local = (H_sub + t_anode) - z  (depth from anode surface)
        """
        # --- Beam x position (sawtooth centered at each period) ---
        # x_beam = 0 at t = k*T, sweeps from -L/2 to +L/2 during beam-on
        term_floor = "floor(t/T_period + 0.5)"
        x_beam = f"v_beam * (t - {term_floor}*T_period)"

        # --- 2D Super-Gaussian lateral profile (order 4) ---
        # G_xy = norm_xy * exp(-((x-xb)^2/sx^2)^2 - (y^2/sy^2)^2)
        # Using (a^2)^2 form to avoid issues with negative bases
        dx = f"(x-({x_beam}))"
        gauss_xy = (f"norm_xy*exp("
                    f"-({dx}^2/sigma_x^2)^2"
                    f" - (y^2/sigma_y^2)^2)")

        # --- Depth from anode surface ---
        z_local = "(H_sub + t_anode - z)"

        # --- Depth profile (substitute z_local everywhere) ---
        # Standard Gruen: (z/z_peak) * exp(-(z-z_peak)^2 / (2*sigma_z^2))
        gauss_z_std = (
            f"norm_z*({z_local}/z_peak)"
            f"*exp(-(({z_local}-z_peak)^2)/(2*sigma_z^2))"
        )

        # Modified Gruen: (z/alpha)^beta * exp(-z/alpha)
        gauss_z_mod = (
            f"norm_z*({z_local}/alpha_z)^beta_z"
            f"*exp(-{z_local}/alpha_z)"
        )

        # Select depth profile based on flag (arithmetic switch)
        gauss_z = (f"((use_modified_gruen)*({gauss_z_mod})"
                   f" + (1-use_modified_gruen)*({gauss_z_std}))")

        # Valid depth range: z_local in (0, R_gruen)
        valid_z = f"(({z_local} > 0) * ({z_local} < R_gruen))"

        # Complete expression (beam_state from Events interface)
        Q_expr = f"P0 * {gauss_xy} * {gauss_z} * {valid_z} * beam_state"

        return Q_expr

    def _create_mesh(self) -> None:
        """Create mesh with swept mesh in beam track and free tet elsewhere.

        Strategy:
        - Beam track domain: swept mesh through anode thickness with fine
          face mesh to resolve beam spot profile
        - Outer anode + substrate: free tetrahedral with graded sizes
        """
        print("  Creating mesh...")

        model = self.java
        comp = model.component("comp1")
        mesh = comp.mesh().create("mesh1")

        # --- Global mesh size (moderate, for substrate bulk) ---
        mesh.feature("size").set("custom", "on")
        mesh.feature("size").set("hmaxactive", "on")
        mesh.feature("size").set("hmax", f"{SUBSTRATE_SIZE[2] / 5}")
        mesh.feature("size").set("hminactive", "on")
        mesh.feature("size").set("hmin", f"{self.anode_thickness / 2}")
        print(f"    Global hmax = {SUBSTRATE_SIZE[2]/5*1e6:.0f} um")

        # --- Fine size for beam track (controls swept mesh face) ---
        face_hmax = BEAM_SIGMA_X / 3.0
        size_bt = mesh.create("size_bt", "Size")
        size_bt.selection().named("geom1_csel_beam_track")
        size_bt.set("custom", "on")
        size_bt.set("hmaxactive", "on")
        size_bt.set("hmax", f"{face_hmax}")
        size_bt.set("hminactive", "on")
        size_bt.set("hmin", f"{face_hmax / 5}")
        print(f"    Beam track face hmax = {face_hmax*1e6:.1f} um")

        # --- Swept mesh for beam track (through anode thickness) ---
        n_layers = max(10, int(self.anode_thickness / 2e-6))
        swe = mesh.create("swe1", "Sweep")
        swe.selection().named("geom1_csel_beam_track")
        dist = swe.create("dis1", "Distribution")
        dist.set("type", "predefined")
        dist.set("numelem", str(n_layers))
        print(f"    Swept mesh: {n_layers} layers through "
              f"{self.anode_thickness*1e6:.0f} um anode")

        # --- Free tetrahedral for remaining domains ---
        ftet = mesh.create("ftet1", "FreeTet")
        # Auto-excludes already-meshed beam track domain

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
        n_periods = N_PERIODS_MIN
        total_time = n_periods * PERIOD

        step = study.create("time1", "Transient")
        step.label("Time Dependent")
        step.set("tlist", f"range(0, {DT_BEAM_ON}, {total_time})")

        # Solver configuration
        sol = model.sol().create("sol1")
        sol.study("std1")
        sol.attach("std1")

        sol.create("st1", "StudyStep")
        sol.feature("st1").set("study", "std1")
        sol.feature("st1").set("studystep", "time1")

        sol.create("v1", "Variables")
        sol.feature("v1").set("control", "time1")

        sol.create("t1", "Time")
        sol.feature("t1").set("control", "time1")

        # Add events and configure time stepping
        self._add_time_step_events(sol)

        # Solver settings
        sol.feature("t1").set("maxorder", "2")
        sol.feature("t1").set("estrat", "exclude")
        sol.feature("t1").set("tstepsbdf", "free")

    def _add_time_step_events(self, sol) -> None:
        """
        Configure time stepping with periodic COMSOL discrete events.

        Uses 2 periodic events (beam ON and beam OFF) instead of creating
        individual events for each cycle. The Events interface forces the
        solver to restart at each event time, ensuring beam transitions
        are never skipped by adaptive time stepping.
        """
        print("    Configuring events and time stepping...")

        model = self.java
        comp = model.component("comp1")

        # --- Events interface for beam state control ---
        ev = comp.physics().create("ev", "Events", "geom1")
        ev.label("Beam State Control")

        # Discrete state: beam_state (0=off, 1=on)
        ds = ev.feature().create("ds1", "DiscreteStates")
        ds.label("Beam State")
        ds.set("dim", "beam_state")
        ds.set("dimInit", "1")  # Beam ON at t=0

        t_on = self.beam_on_time
        t_period = PERIOD
        n_periods = N_PERIODS_MIN

        print(f"      Period: {t_period*1e6:.1f} us")
        print(f"      Beam on time: {t_on*1e6:.1f} us")
        print(f"      Beam off time: {(t_period - t_on)*1e6:.1f} us")
        print(f"      Number of periods: {n_periods}")

        # --- Periodic event: beam OFF ---
        # Fires at t = 0.5*t_on, repeats every T_period
        # At these times, beam has reached +L/2, turn off
        ev.feature().create("evt_off", "ExplicitEvent")
        f_off = ev.feature("evt_off")
        f_off.label("Beam OFF (periodic)")
        f_off.set("start", f"{0.5 * t_on:.9g}")
        f_off.set("period", f"{t_period:.9g}")
        f_off.set("reInitName", "beam_state")
        f_off.set("reInitValue", "0")

        # --- Periodic event: beam ON ---
        # Fires at t = T - 0.5*t_on, repeats every T_period
        # At these times, beam enters at -L/2, turn on
        ev.feature().create("evt_on", "ExplicitEvent")
        f_on = ev.feature("evt_on")
        f_on.label("Beam ON (periodic)")
        f_on.set("start", f"{t_period - 0.5 * t_on:.9g}")
        f_on.set("period", f"{t_period:.9g}")
        f_on.set("reInitName", "beam_state")
        f_on.set("reInitValue", "1")

        print("      Created 2 periodic events (ON/OFF)")

        # --- Output time list with fine steps around transitions ---
        dt_on = max(t_on / 20.0, 1e-9)
        dt_off = max((t_period - t_on) / 10.0, 1e-9)

        parts = []
        for n in range(n_periods):
            t_start = n * t_period
            t_mid_off = t_start + 0.5 * t_on
            t_mid_on = (n + 1) * t_period - 0.5 * t_on
            t_end = (n + 1) * t_period

            # Segment 1 (ON): beam center to edge
            parts.append(f"range({t_start:.9g},{dt_on:.9g},{t_mid_off:.9g})")
            # Segment 2 (OFF): beam off period
            parts.append(f"range({t_mid_off:.9g},{dt_off:.9g},{t_mid_on:.9g})")
            # Segment 3 (ON): beam edge to center
            parts.append(f"range({t_mid_on:.9g},{dt_on:.9g},{t_end:.9g})")

        tlist_str = " ".join(parts)
        sol.feature("t1").set("tlist", tlist_str)

        print(f"    dt_on = {dt_on*1e9:.1f} ns, dt_off = {dt_off*1e6:.2f} us")
        print(f"    Total simulation time: {n_periods * t_period * 1e3:.2f} ms")

    def _create_comsol_results(self) -> None:
        """Create COMSOL result objects for post-processing and export.

        Creates:
        - Maximum coupling operator on beam track domain
        - Plot groups for temperature visualization
        - Export configurations
        """
        print("  Creating result objects...")

        model = self.java
        comp = model.component("comp1")

        # --- Maximum operator on beam track for T_max extraction ---
        maxop = comp.cpl().create("maxop_bt", "Maximum")
        maxop.selection().named("geom1_csel_beam_track")
        maxop.label("Max over Beam Track")
        print("    Created maxop_bt (Maximum over beam track)")

        # --- Minimum operator on beam track ---
        minop = comp.cpl().create("minop_bt", "Minimum")
        minop.selection().named("geom1_csel_beam_track")
        minop.label("Min over Beam Track")

        # --- 3D Temperature plot group ---
        pg3d = model.result().create("pg_temp3d", "PlotGroup3D")
        pg3d.label("Temperature 3D")
        pg3d.set("data", "dset1")
        surf = pg3d.create("surf1", "Surface")
        surf.set("expr", "T")
        surf.set("unit", "degC")
        surf.set("colortable", "ThermalLight")

        # --- XZ cross-section (cut plane at y=0) ---
        pg_xz = model.result().create("pg_temp_xz", "PlotGroup3D")
        pg_xz.label("Temperature XZ Cross-Section")
        pg_xz.set("data", "dset1")

        # Create multislice in XZ plane
        mslice = pg_xz.create("mslc1", "Multislice")
        mslice.set("expr", "T")
        mslice.set("unit", "degC")
        mslice.set("colortable", "ThermalLight")
        mslice.set("xnumber", "0")
        mslice.set("ynumber", "1")  # single slice at y=0
        mslice.set("znumber", "0")

        print("    Created COMSOL plot groups (3D, XZ cross-section)")

    def solve(self) -> None:
        """Run the simulation."""
        print("Running simulation...")
        import time
        t0 = time.time()
        self.model.solve()
        dt = time.time() - t0
        print(f"Simulation complete in {dt:.1f} seconds.")

    def get_results(self) -> Dict[str, Any]:
        """Extract results from the simulation.

        Returns dict with:
        - times: array of time points
        - T_max_history: max T over all domains at each time
        - T_max_bt_history: max T in beam track domain at each time
        - T_min_bt_history: min T in beam track domain at each time
        - T_max: overall maximum temperature
        """
        print("Extracting results...")

        results = {}

        try:
            model = self.java
            sol = model.sol("sol1")

            # Get solution times
            times_java = sol.getPVals()
            if hasattr(times_java, '__iter__'):
                times = np.array([float(t) for t in times_java])
            else:
                times = np.array([float(times_java)])

            print(f"  Number of time points: {len(times)}")
            print(f"  Time range: {times[0]*1e6:.2f} to {times[-1]*1e6:.2f} us")

            # --- Max T in beam track at each time step ---
            # Use the maxop_bt coupling operator
            print("  Evaluating max T in beam track...")
            try:
                T_max_bt = self.model.evaluate('maxop_bt(T)')
                if isinstance(T_max_bt, np.ndarray):
                    T_max_bt_history = T_max_bt.flatten().tolist()
                else:
                    T_max_bt_history = [float(T_max_bt)]
            except Exception as e:
                print(f"  Warning: maxop_bt evaluation failed ({e}), "
                      "falling back to global max")
                T_max_bt_history = None

            # --- Max T over all domains ---
            print("  Evaluating global max T...")
            T_all = self.model.evaluate('T', unit='K')

            T_max_history = []
            if hasattr(T_all, 'shape') and len(T_all.shape) > 1:
                n_times = len(times)
                if T_all.shape[-1] == n_times:
                    for i in range(n_times):
                        T_max_history.append(float(np.max(T_all[..., i])))
                elif T_all.shape[0] == n_times:
                    for i in range(n_times):
                        T_max_history.append(float(np.max(T_all[i, ...])))
                else:
                    steps = T_all.shape[-1]
                    for i in range(steps):
                        T_max_history.append(float(np.max(T_all[..., i])))
            else:
                T_max_history = [float(np.max(T_all))]

            # Align lengths
            min_len = min(len(times), len(T_max_history))
            times = times[:min_len]
            T_max_history = T_max_history[:min_len]

            if T_max_bt_history is not None:
                T_max_bt_history = T_max_bt_history[:min_len]
            else:
                T_max_bt_history = T_max_history  # fallback

            # --- Min T in beam track (for base temp tracking) ---
            try:
                T_min_bt = self.model.evaluate('minop_bt(T)')
                if isinstance(T_min_bt, np.ndarray):
                    T_min_bt_history = T_min_bt.flatten().tolist()[:min_len]
                else:
                    T_min_bt_history = [float(T_min_bt)]
            except Exception:
                T_min_bt_history = [T_AMBIENT] * min_len

            results['times'] = np.array(times)
            results['T_max_history'] = np.array(T_max_history)
            results['T_max_bt_history'] = np.array(T_max_bt_history)
            results['T_min_bt_history'] = np.array(T_min_bt_history)
            results['T_max'] = max(T_max_bt_history)

            print(f"  Max temperature (beam track): "
                  f"{results['T_max']:.1f} K "
                  f"({results['T_max'] - 273.15:.1f} C)")

        except Exception as e:
            import traceback
            print(f"  Warning: Could not extract full results: {e}")
            traceback.print_exc()
            results['T_max'] = 0
            results['times'] = np.array([])
            results['T_max_history'] = np.array([])
            results['T_max_bt_history'] = np.array([])
            results['T_min_bt_history'] = np.array([])

        return results

    def export_comsol_plots(self, output_dir: str = '.') -> None:
        """Export COMSOL plot groups as image files."""
        print("Exporting COMSOL plots...")
        model = self.java

        import os
        os.makedirs(output_dir, exist_ok=True)

        plot_configs = [
            ("pg_temp3d", "temperature_3d.png"),
            ("pg_temp_xz", "temperature_xz_section.png"),
        ]

        for pg_tag, filename in plot_configs:
            try:
                filepath = os.path.join(output_dir, filename)
                exp_tag = f"exp_{pg_tag}"
                exp = model.result().export().create(exp_tag, "Image")
                exp.set("sourceobject", pg_tag)
                exp.set("filename", filepath)
                exp.set("size", "manualweb")
                exp.set("unit", "px")
                exp.set("height", "800")
                exp.set("width", "1200")
                exp.run()
                print(f"  Saved {filepath}")
            except Exception as e:
                print(f"  Warning: Could not export {filename}: {e}")

    def save_model(self, filename: Optional[str] = None) -> None:
        """Save the COMSOL model to file."""
        if filename is None:
            filename = (f"{MODEL_NAME}_{self.anode_material}"
                        f"_{int(self.anode_thickness*1e6)}um.mph")
        print(f"Saving model to {filename}...")
        self.model.save(filename)

    def close(self) -> None:
        """Clean up COMSOL resources."""
        if self.client:
            self.client.clear()
            print("COMSOL client closed.")

    def check_equilibrium(self, times: np.ndarray,
                          T_max_history: np.ndarray) -> bool:
        """
        Check if thermal equilibrium has been reached.

        Looks at the peak temperature of the last 5 periods.
        Equilibrium is reached when the range is within tolerance.
        """
        if len(times) == 0 or len(T_max_history) == 0:
            return False

        t_period = PERIOD
        t_max = times[-1]
        n_periods = int(t_max / t_period)

        if n_periods < 5:
            print("  Not enough periods to check equilibrium (< 5).")
            return False

        period_peaks = []
        for n in range(n_periods):
            t_start = n * t_period
            t_end = (n + 1) * t_period
            indices = np.where((times >= t_start) & (times < t_end))[0]
            if len(indices) > 0:
                period_peaks.append(np.max(T_max_history[indices]))

        if len(period_peaks) < 5:
            return False

        last_5 = np.array(period_peaks[-5:])
        peak_range = np.max(last_5) - np.min(last_5)

        print(f"  Equilibrium check (last 5 peaks): {last_5}")
        print(f"  Range: {peak_range:.2f} K "
              f"(Tolerance: {EQUILIBRIUM_TOLERANCE} K)")

        if peak_range < EQUILIBRIUM_TOLERANCE:
            print("  Equilibrium REACHED.")
            return True
        else:
            print("  Equilibrium NOT reached.")
            return False


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
    """Run a single simulation with given parameters."""
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

        if results:
            is_equil = sim.check_equilibrium(
                results.get('times', np.array([])),
                results.get('T_max_bt_history', np.array([])))
            results['equilibrium_reached'] = is_equil

            # Export COMSOL plots
            sim.export_comsol_plots()

        return results

    finally:
        sim.close()


def save_results_csv(results: Dict[str, Any], filename: str) -> None:
    """Save temperature history to CSV file."""
    times = results.get('times', np.array([]))
    T_max = results.get('T_max_history', np.array([]))
    T_max_bt = results.get('T_max_bt_history', np.array([]))

    if len(times) == 0 or len(T_max) == 0:
        print(f"No data to save to {filename}")
        return

    with open(filename, 'w') as f:
        f.write("time_s,time_us,T_max_global_K,T_max_beamtrack_K,"
                "T_max_global_C,T_max_beamtrack_C\n")
        for i, t in enumerate(times):
            Tg = T_max[i] if i < len(T_max) else 0
            Tb = T_max_bt[i] if i < len(T_max_bt) else 0
            f.write(f"{t:.12e},{t*1e6:.6f},"
                    f"{Tg:.2f},{Tb:.2f},"
                    f"{Tg-273.15:.2f},{Tb-273.15:.2f}\n")
    print(f"Saved results to {filename}")


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
        help='Anode thickness in um (default: 20)'
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
    parser.add_argument(
        '--output-prefix', type=str, default='sim_results',
        help='Prefix for output files (default: sim_results)'
    )

    args = parser.parse_args()

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
        T_max = results.get('T_max', 0)
        print(f"Maximum temperature: {T_max:.1f} K ({T_max - 273.15:.1f} C)")

        # Save CSV
        csv_filename = (f"{args.output_prefix}_{args.anode_material}"
                        f"_{int(args.anode_thickness)}um.csv")
        save_results_csv(results, csv_filename)

        # Report equilibrium
        if results.get('equilibrium_reached'):
            print(">> THERMAL EQUILIBRIUM ACHIEVED <<")
        else:
            print(">> WARNING: Thermal equilibrium NOT achieved <<")

        # Generate Python-side visualizations
        try:
            from visualization import generate_all_plots
            generate_all_plots(results, args, args.output_prefix)
        except ImportError as e:
            print(f"Visualization module not available: {e}")
        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
