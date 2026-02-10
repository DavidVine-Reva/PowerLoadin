"""
Visualization module for rotating anode X-ray source simulation.

Provides analytical heat source plots (no COMSOL required) and
temperature result plots from simulation data.

All plots are saved to files for easy viewing.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from config import (
    SUBSTRATE_SIZE, ANODE_THICKNESS, BEAM_SIGMA_X, BEAM_SIGMA_Y,
    TRACK_LENGTH, PERIOD, T_AMBIENT, DEFAULT_VOLTAGE_KV,
    N_PERIODS_MIN, VELOCITY, SUPERGAUSS_ORDER, BEAM_TRACK_NSIGMA,
)

try:
    from fit_loader import get_fit_params
except ImportError:
    get_fit_params = None


# ---------------------------------------------------------------------------
# Depth profile helpers
# ---------------------------------------------------------------------------

def _depth_profile_modified_gruen(z, alpha, beta):
    """Normalized Modified Gruen depth profile: (z/a)^b * exp(-z/a)."""
    norm = 1.0 / (alpha * math.gamma(beta + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = norm * np.where(
            z > 0,
            (z / alpha) ** beta * np.exp(-z / alpha),
            0.0,
        )
    return profile


def _depth_profile_standard_gruen(z, z_peak, sigma_z, R_gruen):
    """Normalized Standard Gruen depth profile."""
    norm = 1.0 / (sigma_z * math.sqrt(2 * math.pi))
    profile = np.where(
        (z > 0) & (z < R_gruen),
        norm * (z / z_peak) * np.exp(-((z - z_peak) ** 2) / (2 * sigma_z ** 2)),
        0.0,
    )
    return profile


def _get_depth_profile_func(voltage_kV, anode_material='Mo'):
    """Return (profile_func, R_gruen) for the given parameters.

    profile_func(z) returns the normalized depth profile at depth z (meters).
    """
    use_fit = (anode_material == 'Mo') and (get_fit_params is not None)

    if use_fit:
        alpha_m, beta, _ = get_fit_params(voltage_kV)
        R_gruen = 10 * alpha_m

        def profile(z):
            return _depth_profile_modified_gruen(z, alpha_m, beta)

        return profile, R_gruen, {'type': 'modified_gruen',
                                   'alpha': alpha_m, 'beta': beta}
    else:
        from gruen_function import get_simple_depth_expression
        params = get_simple_depth_expression(voltage_kV, anode_material)

        def profile(z):
            return _depth_profile_standard_gruen(
                z, params['z_peak'], params['sigma_z'], params['R_gruen'])

        return profile, params['R_gruen'], {'type': 'standard_gruen',
                                             **params}


# ---------------------------------------------------------------------------
# Super-Gaussian helpers
# ---------------------------------------------------------------------------

def _supergauss_1d(x, sigma, n=SUPERGAUSS_ORDER):
    """1D super-Gaussian: exp(-(|x|/sigma)^n), NOT normalized."""
    return np.exp(-np.abs(x / sigma) ** n)


def _supergauss_1d_norm(sigma, n=SUPERGAUSS_ORDER):
    """Normalization for 1D super-Gaussian so integral = 1."""
    return n / (2.0 * sigma * math.gamma(1.0 / n))


def _supergauss_2d_norm(sigma_x, sigma_y, n=SUPERGAUSS_ORDER):
    """Normalization for separable 2D super-Gaussian."""
    return (_supergauss_1d_norm(sigma_x, n) *
            _supergauss_1d_norm(sigma_y, n))


# ---------------------------------------------------------------------------
# Heat source projection plots at t=0
# ---------------------------------------------------------------------------

def plot_heat_source_projections(
    voltage_kV=DEFAULT_VOLTAGE_KV,
    anode_material='Mo',
    anode_thickness=ANODE_THICKNESS,
    power=100.0,
    output_prefix='heat_source',
):
    """
    Plot XY, XZ, and YZ cross-sections of the heat source at t=0.

    At t=0, beam_state=1 and x_beam=0 (beam at center of track).

    Saves to: {output_prefix}_projections.png
    """
    sigma_x = BEAM_SIGMA_X
    sigma_y = BEAM_SIGMA_Y
    n = SUPERGAUSS_ORDER
    H_sub = SUBSTRATE_SIZE[2]
    t_anode = anode_thickness

    depth_func, R_gruen, depth_info = _get_depth_profile_func(
        voltage_kV, anode_material)

    norm_xy = _supergauss_2d_norm(sigma_x, sigma_y, n)

    # --- Coordinate grids ---
    # XY: surface plane (z = H_sub + t_anode, z_local = 0 -> peak near surface)
    x_range = 3 * sigma_x
    y_range = 3 * sigma_y
    nx, ny = 200, 200
    x_1d = np.linspace(-x_range, x_range, nx)
    y_1d = np.linspace(-y_range, y_range, ny)
    X, Y = np.meshgrid(x_1d, y_1d)

    # Depth range (z_local from 0 to min(anode_thickness, R_gruen))
    z_max_depth = min(t_anode, R_gruen * 1.5)
    nz = 200
    z_local_1d = np.linspace(0, z_max_depth, nz)

    # --- XY projection (integrate over depth) ---
    # Since depth profile is normalized, integral over z = 1
    # Q_xy(x,y) = P0 * norm_xy * exp(-(x/sx)^n - (y/sy)^n) * 1
    Q_xy = power * norm_xy * _supergauss_1d(X, sigma_x, n) * \
           _supergauss_1d(Y, sigma_y, n)

    # --- XZ cross-section at y=0 ---
    X_xz, Z_xz = np.meshgrid(x_1d, z_local_1d)
    G_x = _supergauss_1d_norm(sigma_x, n) * _supergauss_1d(X_xz, sigma_x, n)
    G_z = depth_func(Z_xz)
    Q_xz = power * G_x * G_z

    # --- YZ cross-section at x=0 ---
    Y_yz, Z_yz = np.meshgrid(y_1d, z_local_1d)
    G_y = _supergauss_1d_norm(sigma_y, n) * _supergauss_1d(Y_yz, sigma_y, n)
    G_z_yz = depth_func(Z_yz)
    Q_yz = power * G_y * G_z_yz

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # XY projection
    ax = axes[0]
    im = ax.pcolormesh(x_1d * 1e6, y_1d * 1e6, Q_xy,
                       shading='auto', cmap='hot')
    cb = fig.colorbar(im, ax=ax, label='Surface power density (W/m$^2$)')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title(f'XY Projection (t=0)\nIntegrated over depth')
    ax.set_aspect('auto')

    # XZ cross-section at y=0
    ax = axes[1]
    # Invert z_local axis so surface is at top
    im = ax.pcolormesh(x_1d * 1e6, z_local_1d * 1e6, Q_xz,
                       shading='auto', cmap='hot')
    cb = fig.colorbar(im, ax=ax, label='Q (W/m$^3$)')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('Depth from surface (um)')
    ax.set_title('XZ Cross-section (y=0, t=0)')
    ax.invert_yaxis()

    # YZ cross-section at x=0
    ax = axes[2]
    im = ax.pcolormesh(y_1d * 1e6, z_local_1d * 1e6, Q_yz,
                       shading='auto', cmap='hot')
    cb = fig.colorbar(im, ax=ax, label='Q (W/m$^3$)')
    ax.set_xlabel('y (um)')
    ax.set_ylabel('Depth from surface (um)')
    ax.set_title('YZ Cross-section (x=0, t=0)')
    ax.invert_yaxis()

    fig.suptitle(
        f'Heat Source Distribution at t=0 | {anode_material}, '
        f'{voltage_kV} kV, P={power} W\n'
        f'Super-Gaussian order {n}, '
        f'sx={sigma_x*1e6:.1f} um, sy={sigma_y*1e6:.1f} um',
        fontsize=11, y=1.02)
    plt.tight_layout()

    outfile = f"{output_prefix}_projections.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Depth profile comparison plot
# ---------------------------------------------------------------------------

def plot_depth_profile(
    voltage_kV=DEFAULT_VOLTAGE_KV,
    anode_material='Mo',
    anode_thickness=ANODE_THICKNESS,
    output_prefix='heat_source',
):
    """Plot the depth energy deposition profile.

    Saves to: {output_prefix}_depth_profile.png
    """
    depth_func, R_gruen, info = _get_depth_profile_func(
        voltage_kV, anode_material)

    z_max = min(anode_thickness, R_gruen * 1.5)
    z = np.linspace(0, z_max, 500)
    D = depth_func(z)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z * 1e6, D * 1e-6, 'b-', linewidth=2)
    ax.axvline(anode_thickness * 1e6, color='r', linestyle='--',
               label=f'Anode thickness ({anode_thickness*1e6:.0f} um)')
    if R_gruen < anode_thickness * 2:
        ax.axvline(R_gruen * 1e6, color='gray', linestyle=':',
                   label=f'Gruen range ({R_gruen*1e6:.1f} um)')

    ax.set_xlabel('Depth from surface (um)')
    ax.set_ylabel('Normalized energy deposition (1/um)')
    ax.set_title(
        f'Depth Profile | {anode_material}, {voltage_kV} kV\n'
        f'Type: {info["type"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, z_max * 1e6)

    plt.tight_layout()
    outfile = f"{output_prefix}_depth_profile.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Beam position vs time
# ---------------------------------------------------------------------------

def plot_beam_position_vs_time(
    n_periods=3,
    output_prefix='heat_source',
):
    """
    Plot beam center position and beam_state vs time.

    Verifies that the heat source traverses the focal track correctly
    during each beam-on cycle.

    Saves to: {output_prefix}_beam_position.png
    """
    t_on = TRACK_LENGTH / VELOCITY
    t_period = PERIOD

    # Fine time array
    dt = t_on / 100
    t = np.arange(0, n_periods * t_period, dt)

    # Beam position: x_beam = v * (t - floor(t/T + 0.5) * T)
    x_beam = VELOCITY * (t - np.floor(t / t_period + 0.5) * t_period)

    # Beam state (analytical recreation)
    t_mod = np.mod(t, t_period)
    beam_on = ((t_mod < 0.5 * t_on) | (t_mod > t_period - 0.5 * t_on))
    beam_state = beam_on.astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Position plot
    # Color segments by beam state
    for i in range(len(t) - 1):
        color = '#d62728' if beam_state[i] > 0.5 else '#aaaaaa'
        lw = 2.0 if beam_state[i] > 0.5 else 0.5
        ax1.plot(t[i:i+2] * 1e6, x_beam[i:i+2] * 1e3,
                 color=color, linewidth=lw)

    ax1.axhline(TRACK_LENGTH / 2 * 1e3, color='blue', linestyle='--',
                alpha=0.5, label=f'+L/2 = {TRACK_LENGTH/2*1e3:.1f} mm')
    ax1.axhline(-TRACK_LENGTH / 2 * 1e3, color='blue', linestyle='--',
                alpha=0.5, label=f'-L/2 = {-TRACK_LENGTH/2*1e3:.1f} mm')
    ax1.set_ylabel('Beam x position (mm)')
    ax1.set_title(
        f'Beam Position & State vs Time\n'
        f'v = {VELOCITY} m/s, L = {TRACK_LENGTH*1e3:.1f} mm, '
        f't_on = {t_on*1e6:.1f} us, T = {t_period*1e6:.0f} us')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add beam-on shading
    for n in range(n_periods):
        tc = n * t_period
        # ON: tc - 0.5*t_on to tc + 0.5*t_on
        # For n=0: 0 to 0.5*t_on (partial)
        on_start = max(0, tc - 0.5 * t_on)
        on_end = tc + 0.5 * t_on
        ax1.axvspan(on_start * 1e6, on_end * 1e6,
                    alpha=0.1, color='red')

    # State plot
    ax2.fill_between(t * 1e6, beam_state, step='post',
                     color='red', alpha=0.4, label='beam ON')
    ax2.set_xlabel('Time (us)')
    ax2.set_ylabel('beam_state')
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f"{output_prefix}_beam_position.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Peak heat source location vs time
# ---------------------------------------------------------------------------

def plot_peak_heat_location_vs_time(
    n_periods=5,
    output_prefix='heat_source',
):
    """
    Plot the (x, y, z_local) location of peak volumetric heat source vs time.

    Peak is always at (x_beam, 0, z_peak) where z_peak depends on the
    depth profile. This confirms the heat source moves correctly.

    Saves to: {output_prefix}_peak_location.png
    """
    t_on = TRACK_LENGTH / VELOCITY
    t_period = PERIOD

    dt = t_on / 50
    t = np.arange(0, n_periods * t_period, dt)

    # Beam position
    x_beam = VELOCITY * (t - np.floor(t / t_period + 0.5) * t_period)

    # Beam state
    t_mod = np.mod(t, t_period)
    beam_on = ((t_mod < 0.5 * t_on) | (t_mod > t_period - 0.5 * t_on))

    # Peak depth (from depth profile)
    depth_func, R_gruen, info = _get_depth_profile_func(
        DEFAULT_VOLTAGE_KV, 'Mo')

    # Find peak depth numerically
    z_test = np.linspace(0, min(ANODE_THICKNESS, R_gruen), 1000)
    D_test = depth_func(z_test)
    z_peak = z_test[np.argmax(D_test)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # X position of peak
    ax = axes[0]
    ax.scatter(t[beam_on] * 1e6, x_beam[beam_on] * 1e3,
               c='red', s=2, label='Beam ON')
    ax.scatter(t[~beam_on] * 1e6, x_beam[~beam_on] * 1e3,
               c='gray', s=1, alpha=0.3, label='Beam OFF')
    ax.axhline(TRACK_LENGTH / 2 * 1e3, color='blue', linestyle='--',
               alpha=0.5)
    ax.axhline(-TRACK_LENGTH / 2 * 1e3, color='blue', linestyle='--',
               alpha=0.5)
    ax.set_ylabel('x_peak (mm)')
    ax.set_title('Peak Heat Source Location vs Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y position (always 0)
    ax = axes[1]
    ax.axhline(0, color='red', linewidth=2)
    ax.set_ylabel('y_peak (mm)')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    # Z depth of peak
    ax = axes[2]
    z_plot = np.where(beam_on, z_peak, np.nan)
    ax.scatter(t[beam_on] * 1e6,
               np.full(np.sum(beam_on), z_peak * 1e6),
               c='red', s=2)
    ax.axhline(z_peak * 1e6, color='red', linestyle='-', alpha=0.5,
               label=f'z_peak = {z_peak*1e6:.2f} um')
    ax.axhline(ANODE_THICKNESS * 1e6, color='black', linestyle='--',
               alpha=0.5, label=f'Anode thickness = {ANODE_THICKNESS*1e6:.0f} um')
    ax.set_ylabel('z_peak depth (um)')
    ax.set_xlabel('Time (us)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f"{output_prefix}_peak_location.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Temperature history plots (from simulation results)
# ---------------------------------------------------------------------------

def plot_temperature_history(
    results,
    output_prefix='sim_results',
):
    """
    Plot max temperature in the beam track domain vs time.

    Shows beam-on/off shading, global max T, and beam track max T.

    Saves to: {output_prefix}_temperature_history.png
    """
    times = results.get('times', np.array([]))
    T_max_bt = results.get('T_max_bt_history', np.array([]))
    T_max_global = results.get('T_max_history', np.array([]))

    if len(times) == 0:
        print("  No data for temperature history plot")
        return

    t_on = TRACK_LENGTH / VELOCITY
    t_period = PERIOD

    fig, ax = plt.subplots(figsize=(14, 6))

    # Beam-on shading
    n_periods = int(times[-1] / t_period) + 1
    for n in range(n_periods):
        tc = n * t_period
        on_start = max(0, tc - 0.5 * t_on)
        on_end = tc + 0.5 * t_on
        if on_start < times[-1]:
            ax.axvspan(on_start * 1e6, min(on_end, times[-1]) * 1e6,
                       alpha=0.08, color='red')

    # Temperature curves
    if len(T_max_bt) == len(times):
        ax.plot(times * 1e6, T_max_bt - 273.15, 'r-', linewidth=1.5,
                label='Max T (beam track)', zorder=3)

    if len(T_max_global) == len(times) and not np.array_equal(
            T_max_global, T_max_bt):
        ax.plot(times * 1e6, T_max_global - 273.15, 'b--', linewidth=1,
                alpha=0.7, label='Max T (global)', zorder=2)

    ax.set_xlabel('Time (us)', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.set_title('Maximum Temperature vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Info box
    T_peak = np.max(T_max_bt) - 273.15 if len(T_max_bt) > 0 else 0
    T_final = T_max_bt[-1] - 273.15 if len(T_max_bt) > 0 else 0
    info = (f'T_peak = {T_peak:.1f} C\n'
            f'T_final = {T_final:.1f} C')
    ax.text(0.98, 0.98, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    outfile = f"{output_prefix}_temperature_history.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


def plot_equilibrium_trend(
    results,
    output_prefix='sim_results',
):
    """
    Plot per-cycle peak and base temperatures to visualize approach
    to thermal equilibrium.

    Saves to: {output_prefix}_equilibrium_trend.png
    """
    times = results.get('times', np.array([]))
    T_max_bt = results.get('T_max_bt_history', np.array([]))
    T_min_bt = results.get('T_min_bt_history', np.array([]))

    if len(times) == 0:
        print("  No data for equilibrium trend plot")
        return

    t_period = PERIOD
    n_periods = int(times[-1] / t_period)

    if n_periods < 2:
        print("  Not enough periods for equilibrium trend plot")
        return

    cycle_nums = []
    peak_temps = []
    base_temps = []

    for n in range(n_periods):
        t_start = n * t_period
        t_end = (n + 1) * t_period
        idx = np.where((times >= t_start) & (times < t_end))[0]

        if len(idx) > 0:
            cycle_nums.append(n + 1)
            peak_temps.append(np.max(T_max_bt[idx]) - 273.15)
            base_temps.append(np.min(T_min_bt[idx]) - 273.15)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cycle_nums, peak_temps, 'ro-', markersize=4,
            label='Peak T per cycle')
    ax.plot(cycle_nums, base_temps, 'bs-', markersize=4,
            label='Base T per cycle')

    # Trend line for last half
    if len(cycle_nums) > 6:
        mid = len(cycle_nums) // 2
        ax.axhline(np.mean(peak_temps[mid:]), color='red',
                   linestyle=':', alpha=0.5,
                   label=f'Mean peak (last half) = '
                         f'{np.mean(peak_temps[mid:]):.1f} C')
        ax.axhline(np.mean(base_temps[mid:]), color='blue',
                   linestyle=':', alpha=0.5,
                   label=f'Mean base (last half) = '
                         f'{np.mean(base_temps[mid:]):.1f} C')

    ax.set_xlabel('Cycle number')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Thermal Equilibrium Convergence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = f"{output_prefix}_equilibrium_trend.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


def plot_temperature_zoom_last_cycles(
    results,
    n_cycles=3,
    output_prefix='sim_results',
):
    """
    Zoom into the last few cycles to show the periodic temperature response.

    Saves to: {output_prefix}_temperature_zoom.png
    """
    times = results.get('times', np.array([]))
    T_max_bt = results.get('T_max_bt_history', np.array([]))

    if len(times) == 0:
        print("  No data for zoom plot")
        return

    t_period = PERIOD
    t_on = TRACK_LENGTH / VELOCITY

    # Last n_cycles
    t_start = times[-1] - n_cycles * t_period
    t_start = max(0, t_start)
    idx = times >= t_start

    fig, ax = plt.subplots(figsize=(14, 5))

    # Beam-on shading
    n_full = int(times[-1] / t_period) + 1
    for n in range(n_full):
        tc = n * t_period
        on_start = max(t_start, tc - 0.5 * t_on)
        on_end = tc + 0.5 * t_on
        if on_end > t_start and on_start < times[-1]:
            ax.axvspan(on_start * 1e6, min(on_end, times[-1]) * 1e6,
                       alpha=0.15, color='red', label='Beam ON' if n == 0 else '')

    ax.plot(times[idx] * 1e6, T_max_bt[idx] - 273.15, 'r-',
            linewidth=2, label='Max T (beam track)')

    ax.set_xlabel('Time (us)', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.set_title(f'Temperature Detail - Last {n_cycles} Cycles', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    outfile = f"{output_prefix}_temperature_zoom.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outfile}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(results=None, args=None, output_prefix='sim_results'):
    """Generate all visualization plots.

    Args:
        results: Simulation results dict (None for analytical-only plots)
        args: CLI args namespace with anode_material, voltage, etc.
        output_prefix: Prefix for output filenames
    """
    print("\nGenerating plots...")

    # Extract parameters from args or use defaults
    if args is not None:
        voltage = getattr(args, 'voltage', DEFAULT_VOLTAGE_KV)
        material = getattr(args, 'anode_material', 'Mo')
        thickness = getattr(args, 'anode_thickness', 20) * 1e-6
        power = getattr(args, 'power', 100)
    else:
        voltage = DEFAULT_VOLTAGE_KV
        material = 'Mo'
        thickness = ANODE_THICKNESS
        power = 100.0

    # --- Analytical plots (no COMSOL needed) ---
    plot_heat_source_projections(
        voltage_kV=voltage,
        anode_material=material,
        anode_thickness=thickness,
        power=power,
        output_prefix=output_prefix,
    )

    plot_depth_profile(
        voltage_kV=voltage,
        anode_material=material,
        anode_thickness=thickness,
        output_prefix=output_prefix,
    )

    plot_beam_position_vs_time(
        n_periods=3,
        output_prefix=output_prefix,
    )

    plot_peak_heat_location_vs_time(
        n_periods=5,
        output_prefix=output_prefix,
    )

    # --- Simulation result plots (require results data) ---
    if results is not None and len(results.get('times', [])) > 0:
        plot_temperature_history(results, output_prefix)
        plot_equilibrium_trend(results, output_prefix)
        plot_temperature_zoom_last_cycles(results, n_cycles=3,
                                          output_prefix=output_prefix)

    print("All plots generated.\n")


if __name__ == '__main__':
    # Generate analytical plots only (no COMSOL required)
    generate_all_plots(results=None, output_prefix='heat_source_check')
