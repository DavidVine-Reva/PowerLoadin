#!/usr/bin/env python3
"""
Inspect the COMSOL model to learn how events are configured.
"""
import mph

print("Loading COMSOL model...")
client = mph.start()
model = client.load("rotating_anode_heat_transfer_Mo_99um.mph")
J = model.java

print("\n=== INSPECTING EVENTS INTERFACE ===")
comp = J.component("comp1")
physics = comp.physics()

# List all physics interfaces
print("\nPhysics interfaces:")
for tag in physics.tags():
    phy = physics.feature(str(tag))
    print(f"  {tag}: {phy.getType()}")

# Check if events interface exists
try:
    ev = comp.physics("ev")
    print(f"\nEvents interface found: tag='ev'")
    print(f"  Type: {ev.getType()}")
    print(f"  Label: {ev.label()}")
    
    # List all features in events
    print("\n  Features in Events interface:")
    for tag in ev.feature().tags():
        feat = ev.feature(str(tag))
        print(f"    {tag}:")
        print(f"      Type: {feat.getType()}")
        print(f"      Label: {feat.label()}")
        
        # Get all properties
        try:
            props = feat.properties()
            print(f"      Properties: {list(props)}")
            
            # Print values for key properties
            for prop in props:
                try:
                    val = feat.getString(str(prop))
                    if val:
                        print(f"        {prop} = {val}")
                except:
                    try:
                        val = feat.getStringArray(str(prop))
                        if val and len(val) > 0:
                            print(f"        {prop} = {list(val)}")
                    except:
                        pass
        except Exception as e:
            print(f"      Error getting properties: {e}")
            
except Exception as e:
    print(f"Events interface not found: {e}")

# Also check global definitions for any relevant settings
print("\n=== CHECKING PARAMETERS FOR EVENTS ===")
params = J.param()
for name in params.varnames():
    if 'beam' in str(name).lower() or 'state' in str(name).lower() or 'event' in str(name).lower():
        val = params.get(str(name))
        print(f"  {name} = {val}")

print("\nDone!")
client.clear()
