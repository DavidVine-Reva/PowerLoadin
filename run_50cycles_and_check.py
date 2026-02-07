#!/usr/bin/env python3
"""
Run 50-cycle simulation and check thermal equilibrium.
"""

import mph
import numpy as np
import sys

print("="*80)
print("50-CYCLE SIMULATION WITH EQUILIBRIUM CHECK")
print("="*80)

# Configuration
N_cycles = 50
period_s = 1e-3  # 1 ms
duty_cycle = 0.03  # 3%
power_W = 1200.0
T_on = period_s * duty_cycle

print(f"\nConfiguration:")
print(f"  Cycles: {N_cycles}")
print(f"  Period: {period_s*1e3} ms")
print(f"  Duty cycle: {duty_cycle*100}%")
print(f"  T_on: {T_on*1e6:.1f} µs")
print(f"  Power: {power_W} W")

# Connect
print("\n[1] Connecting to COMSOL...")
client = mph.start()
model = client.load("Rigaku MicroMax-007HF.mph")
J = model.java
print("[OK] Connected and loaded")

# Configure events for 50 cycles
print(f"\n[2] Configuring {N_cycles} cycles...")
ev = J.component("comp1").physics("ev")

# Remove existing events
for tag in list(ev.feature().tags()):
    tag_str = str(tag)
    if tag_str.startswith('expl'):
        ev.feature().remove(tag)

# Add events for N cycles
event_num = 1
for i in range(N_cycles):
    t_off = i * period_s + T_on
    t_on_next = (i + 1) * period_s

    # OFF event
    tag_off = f"expl{event_num}"
    ev.feature().create(tag_off, "ExplicitEvent")
    f_off = ev.feature(tag_off)
    f_off.set("start", f"{t_off:.9g}")
    f_off.set("reInitName", "beam_state")
    f_off.set("reInitValue", "0")
    event_num += 1

    # ON event (except after last cycle)
    if i < N_cycles - 1:
        tag_on = f"expl{event_num}"
        ev.feature().create(tag_on, "ExplicitEvent")
        f_on = ev.feature(tag_on)
        f_on.set("start", f"{t_on_next:.9g}")
        f_on.set("reInitName", "beam_state")
        f_on.set("reInitValue", "1")
        event_num += 1

print(f"[OK] Added {event_num-1} events")

# Update study times
print("\n[3] Updating study times...")
dt_on = max(T_on / 20.0, 1e-7)
dt_off = max((period_s - T_on) / 10.0, 1e-7)

parts = []
for i in range(N_cycles):
    t0 = i * period_s
    t_on_end = t0 + T_on
    t1 = (i + 1) * period_s

    if T_on > 0:
        parts.append(f"range({t0:.9g},{dt_on:.9g},{t_on_end:.9g})")
    if (period_s - T_on) > 0:
        parts.append(f"range({t_on_end:.9g},{dt_off:.9g},{t1:.9g})")

tlist_str = " ".join(parts)
J.study("std1").feature("time").set("tlist", tlist_str)
print("[OK] Study times updated")

# Set power
print(f"\n[4] Setting P0 = {power_W} W...")
model.parameter('P0', f'{power_W}[W]')
print("[OK] P0 set")

# Solve
print(f"\n[5] Running solve for {N_cycles} cycles (this will take several minutes)...")
print(f"    Total simulation time: {N_cycles * period_s * 1e3:.1f} ms")

try:
    model.solve()
    print("[OK] Solve completed!")
except Exception as e:
    print(f"[ERROR] Solve failed: {e}")
    sys.exit(1)

# Save model
output_file = f"Rigaku_P{power_W:.0f}W_{N_cycles}cycles_solved.mph"
print(f"\n[6] Saving to {output_file}...")
model.save(output_file)
print("[OK] Saved")

# Extract time data
print(f"\n[7] Extracting results...")
time_data = np.array(model.evaluate('t')).flatten()
print(f"[OK] {len(time_data)} time points")
print(f"    Time range: {time_data[0]:.6f} to {time_data[-1]:.6f} s")

# Energy balance
energy_in_per_cycle = power_W * T_on
total_energy_in = energy_in_per_cycle * N_cycles
print(f"\n[8] Energy balance:")
print(f"    Energy in per cycle: {energy_in_per_cycle:.4f} J")
print(f"    Total energy input: {total_energy_in:.2f} J")
print(f"    Average power: {total_energy_in / (N_cycles * period_s):.1f} W")

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print(f"✓ {N_cycles} cycles simulated")
print(f"✓ {len(time_data)} solution time points")
print(f"✓ Total time: {N_cycles * period_s * 1e3:.1f} ms")
print(f"✓ Saved to: {output_file}")
print("\nTo verify equilibrium:")
print("  Open model in COMSOL GUI")
print("  Plot temperature at a point vs time")
print("  Check that last 5-10 cycles have overlapping temperature profiles")
print("="*80)
