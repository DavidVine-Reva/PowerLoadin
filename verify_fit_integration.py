
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Mock mph module
from unittest.mock import MagicMock
sys.modules['mph'] = MagicMock()

from fit_loader import get_fit_params
from power_loading_sim import RotatingAnodeSimulation

def verify_fit_loader():
    print("Verifying fit_loader...")
    
    # Test cases
    voltages = [30, 100, 160, 55] # 55 is interpolated
    
    for v in voltages:
        try:
            alpha, beta, norm = get_fit_params(v)
            print(f"  Voltage {v} kV: alpha={alpha:.4e} m, beta={beta:.4f}, norm={norm:.4e}")
            
            # Check basic validity
            if alpha <= 0 or beta <= 0 or norm <= 0:
                print(f"  ERROR: Invalid parameters for {v} kV")
                return False
        except Exception as e:
            print(f"  ERROR: Failed to get params for {v} kV: {e}")
            return False
            
    print("fit_loader verification PASSED.")
    return True

def verify_simulation_logic():
    print("\nVerifying simulation logic (mocking COMSOL)...")
    
    # Mock MPH client to avoid error during init
    # Actually, RotatingAnodeSimulation only instantiates client in connect(), not init.
    # But it does import mph. If mph is installed, we are good.
    
    try:
        # Create simulation instance (Mo anode)
        sim = RotatingAnodeSimulation(anode_material='Mo', voltage_kV=100)
        
        # Check if _build_heat_source_expression contains the new logic
        # We can't easily check _create_parameters output without a model, 
        # but we can check the method code or try to run what we can.
        
        # Let's inspect the heat source expression method logic directly 
        # by calling it? No, it relies on self.model which is None.
        
        # Instead, let's just create a dummy object that behaves enough like sim
        # to test the expression string generation if we were to refactor it to not depend on self.params?
        # Actually _build_heat_source_expression doesn't use self.params, it assumes params are set in COMSOL.
        # It DOES uses self.model to get the component... wait.
        # In the code:
        # def _build_heat_source_expression(self) -> str:
        #    ...
        #    return Q_expr
        
        # It looks like _build_heat_source_expression DOES NOT depend on self.model or self.client!
        # It relies on calculated strings.
        # Let's verify this by checking the file again mentally.
        # Yes, it uses f-strings with variables like x_beam, gauss_xy which are local strings.
        # It DOES use "T_period", "sigma_x", etc which are COMSOL parameter names (strings).
        
        expr = sim._build_heat_source_expression()
        print("\nGenerated Heat Source Expression:")
        print(expr)
        
        # Check for presence of Modified Gruen terms
        if "alpha_z" in expr and "beta_z" in expr and "use_modified_gruen" in expr:
            print("\n  SUCCESS: Expression contains Modified Gruen terms.")
        else:
            print("\n  FAILURE: Expression missing Modified Gruen terms.")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ERROR during simulation verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    v1 = verify_fit_loader()
    v2 = verify_simulation_logic()
    
    if v1 and v2:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nCHECKS FAILED")
        sys.exit(1)
