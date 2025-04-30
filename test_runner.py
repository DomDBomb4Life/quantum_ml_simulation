# quantum_ml_simulation/test_runner.py
# Enhanced temporary script to thoroughly test the quantum simulation components

import os
import numpy as np
import time
import traceback # Import traceback for detailed error printing
import itertools

# Import the simulation classes and config using EXPLICIT relative imports
from .simulations.ising_model import IsingModelSimulation
# Assuming TestSimulation hasn't been updated for the new features (like get_hamiltonian_operator),
# we will focus on testing IsingModelSimulation for now.
# from .simulations.test_simulation import TestSimulation
from .config import simulation_params as cfg

# --- Dependency Check ---
try:
    import scipy
    print("[INFO] Found SciPy (needed for exact diagonalization validation).")
except ImportError:
    if cfg.VALIDATE_TROTTER:
        print("\n[WARNING] SciPy not found, but VALIDATE_TROTTER is True in config.")
        print("           Exact diagonalization validation will fail or be skipped.")
        print("           Install SciPy: pip install scipy")
    else:
        print("[INFO] SciPy not found, but validation is disabled. Proceeding.")
    # Allow continuing without scipy if validation is off
    pass


def run_simulation_test(simulation_class, simulation_name):
    """
    Runs thorough tests on a given simulation class.
    """
    print(f"\n{'='*40}")
    print(f"  Testing Simulation: {simulation_name}  ")
    print(f"{'='*40}")

    # --- 1. Configuration Overview ---
    print("[INFO] Using Configuration:")
    print(f"  N_Qubits: {cfg.N_QUBITS}")
    print(f"  DELTA_T: {cfg.DELTA_T}")
    print(f"  N_Steps Range: {min(cfg.N_STEPS_RANGE)}-{max(cfg.N_STEPS_RANGE)} ({len(cfg.N_STEPS_RANGE)} values)")
    print(f"  J Range: {min(cfg.J_RANGE)}-{max(cfg.J_RANGE)} ({len(cfg.J_RANGE)} values)")
    print(f"  B Range: {min(cfg.B_RANGE)}-{max(cfg.B_RANGE)} ({len(cfg.B_RANGE)} values)")
    print(f"  Initial State: {cfg.INITIAL_STATE_TYPE}")
    print(f"  Simulation Mode: {cfg.SIMULATION_MODE}" + (f", Shots: {cfg.N_SHOTS}" if cfg.SIMULATION_MODE == 'shots' else ""))
    print(f"  Validate Trotter (<= {cfg.VALIDATION_MAX_QUBITS} qubits): {cfg.VALIDATE_TROTTER}")
    print(f"  Compute Entanglement (Partition: {cfg.ENTANGLEMENT_PARTITION}): {cfg.COMPUTE_ENTANGLEMENT}")
    print("-" * 40)

    sim = None
    try:
        # --- 2. Instantiation Test ---
        print("[CHECK] Instantiating simulation class...")
        sim = simulation_class()
        print(f"[SUCCESS] Simulation object created for {simulation_name}.")
        print(f"  Object uses N_Qubits={sim.n_qubits}, Mode='{sim.q_simulator.simulation_mode}', Validate={sim.validate_trotter}, Entangle={sim.compute_entanglement}")
        print("-" * 40)

        # --- 3. Parameter Space Test ---
        print("[CHECK] Generating parameter space...")
        param_space = sim.get_parameter_space()
        expected_params = len(cfg.N_STEPS_RANGE) * len(cfg.J_RANGE) * len(cfg.B_RANGE)
        print(f"  Generated {len(param_space)} parameter combinations.")
        if len(param_space) == expected_params:
            print(f"[SUCCESS] Parameter space size matches configuration ({expected_params}).")
        else:
            print(f"[WARNING] Parameter space size ({len(param_space)}) does NOT match expected ({expected_params})!")
        print("-" * 40)

        # --- 4. Spot Check `run_single_simulation` ---
        if not param_space:
             print("[SKIP] Skipping single simulation checks as parameter space is empty.")
        else:
            print("[CHECK] Running spot checks on `run_single_simulation`...")
            # Select a few parameters: min/mid/max n_steps, J, B
            n_steps_samples = [min(cfg.N_STEPS_RANGE), (min(cfg.N_STEPS_RANGE)+max(cfg.N_STEPS_RANGE))//2 , max(cfg.N_STEPS_RANGE)]
            j_samples = [min(cfg.J_RANGE), cfg.J_RANGE[len(cfg.J_RANGE)//2], max(cfg.J_RANGE)]
            b_samples = [min(cfg.B_RANGE), cfg.B_RANGE[len(cfg.B_RANGE)//2], max(cfg.B_RANGE)]
            # Use unique parameters only
            spot_check_params = list(set(itertools.product(n_steps_samples, j_samples, b_samples)))
            print(f"  Testing {len(spot_check_params)} specific parameter sets:")

            all_spot_checks_passed = True
            for i, params in enumerate(spot_check_params):
                print(f"\n  Test {i+1}/{len(spot_check_params)}: PARAMS = {params}")
                start_single = time.time()
                result_dict = sim.run_single_simulation(params)
                end_single = time.time()
                print(f"    -> Completed in {end_single - start_single:.3f} seconds.")
                print(f"    -> RESULT DICT:")
                for key, value in result_dict.items():
                     if isinstance(value, float):
                         print(f"      {key:<20}: {value:.6f}" if not np.isnan(value) else f"      {key:<20}: NaN")
                     else:
                         print(f"      {key:<20}: {value}")

                # Basic checks on the results
                exp_val = result_dict.get('expectation_value', np.nan)
                val_diff = result_dict.get('validation_diff', np.nan)
                ent_val = result_dict.get('entanglement_entropy', np.nan)

                if np.isnan(exp_val):
                    print("    [ERROR] Expectation value is NaN!")
                    all_spot_checks_passed = False
                if sim.validate_trotter and np.isnan(val_diff):
                     # Only expect non-NaN if validation should have run
                     if cfg.N_QUBITS <= cfg.VALIDATION_MAX_QUBITS:
                         print("    [WARNING] Validation difference is NaN when validation should have run.")
                         # Don't mark as error yet, could be SciPy issue or ED failure
                     else:
                          print("    [INFO] Validation difference is NaN (expected as N_Qubits > validation max).")
                elif sim.validate_trotter and not np.isnan(val_diff):
                      print(f"    [INFO] Validation difference computed: {val_diff:.4e}")

                if sim.compute_entanglement and np.isnan(ent_val):
                     if cfg.SIMULATION_MODE == 'statevector':
                         print("    [WARNING] Entanglement entropy is NaN when computation should have run.")
                     else:
                          print("    [INFO] Entanglement entropy is NaN (expected in shots mode).")
                elif sim.compute_entanglement and not np.isnan(ent_val):
                     print(f"    [INFO] Entanglement entropy computed: {ent_val:.4f}")

            if all_spot_checks_passed:
                 print("\n[SUCCESS] All spot checks completed without producing NaN expectation values.")
            else:
                 print("\n[FAILURE] One or more spot checks resulted in NaN expectation value. Check error logs above.")
            print("-" * 40)


        # --- 5. Full Dataset Generation Test ---
        print("[CHECK] Starting full `generate_dataset` run...")
        start_full = time.time()
        # generate_dataset now prints its own summary stats at the end
        full_dataset = sim.generate_dataset()
        end_full = time.time()
        print(f"[INFO] `generate_dataset` finished in {end_full - start_full:.2f} seconds.")

        if not full_dataset:
            print("[ERROR] Full dataset generation resulted in an empty list!")
            return # Stop here if dataset is empty

        num_generated = len(full_dataset)
        num_valid_exp_val = sum(1 for d in full_dataset if 'output' in d and not np.isnan(d['output']))

        if num_valid_exp_val == num_generated:
            print(f"[SUCCESS] All {num_generated} points in the dataset have valid expectation values.")
        else:
            print(f"[WARNING] {num_generated - num_valid_exp_val}/{num_generated} points have NaN expectation values.")

        # --- 6. Analyze Optional Feature Results Across Dataset ---
        if sim.validate_trotter and cfg.N_QUBITS <= cfg.VALIDATION_MAX_QUBITS :
             valid_diffs = [d.get('validation_diff', np.nan) for d in full_dataset]
             valid_diffs = [v for v in valid_diffs if not np.isnan(v)]
             if valid_diffs:
                  print(f"[INFO] Validation run summary ({len(valid_diffs)} points):")
                  print(f"  Avg Abs Diff : {np.mean(np.abs(valid_diffs)):.4e}")
                  print(f"  Max Abs Diff : {np.max(np.abs(valid_diffs)):.4e}")
                  print(f"  Min Diff     : {np.min(valid_diffs):.4e}")
                  print(f"  Max Diff     : {np.max(valid_diffs):.4e}")
             else:
                   print("[WARNING] Validation was enabled, but no valid difference values found in dataset.")

        if sim.compute_entanglement and cfg.SIMULATION_MODE == 'statevector':
             valid_ents = [d.get('entanglement', np.nan) for d in full_dataset]
             valid_ents = [v for v in valid_ents if not np.isnan(v)]
             if valid_ents:
                  print(f"[INFO] Entanglement run summary ({len(valid_ents)} points):")
                  print(f"  Avg Entropy  : {np.mean(valid_ents):.4f}")
                  print(f"  Max Entropy  : {np.max(valid_ents):.4f}")
                  print(f"  Min Entropy  : {np.min(valid_ents):.4f}")
             else:
                  print("[WARNING] Entanglement computation was enabled (statevector mode), but no valid entropy values found.")


    except Exception as e:
        print(f"\n{'!'*40}")
        print(f"AN UNEXPECTED ERROR OCCURRED during test for {simulation_name}:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc() # Print detailed traceback
        print(f"{'!'*40}")
    finally:
        print(f"\n{'='*40}")
        print(f"  Finished Testing: {simulation_name}  ")
        print(f"{'='*40}")


# --- Main execution block for the test ---
if __name__ == "__main__":
    print("Starting Enhanced Quantum Simulation Component Test...")

    # Check Qiskit Aer availability
    try:
        from qiskit_aer import AerSimulator
        print("[INFO] Qiskit Aer simulator found.")
    except ImportError:
        print("\n[ERROR] Qiskit Aer not found. Please install it ('pip install qiskit-aer').")
        exit()

    # Ensure output directories exist
    try:
        os.makedirs(cfg.DATA_PATH, exist_ok=True)
        os.makedirs(cfg.ML_RESULTS_PATH, exist_ok=True)
        print(f"[INFO] Output directories checked/created: {cfg.BASE_SAVE_PATH}")
    except OSError as e:
        print(f"[ERROR] Failed to create output directories: {e}")
        exit()

    # --- Run Tests ---
    # Only testing IsingModelSimulation as it's the one significantly updated
    run_simulation_test(IsingModelSimulation, "IsingModel")

    # If TestSimulation was also updated to the new BaseSimulation requirements
    # (e.g., adding get_hamiltonian_operator), uncomment below:
    # run_simulation_test(TestSimulation, "TestSim")

    print("\nQuantum Simulation Component Test Finished.")