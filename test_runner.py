# quantum_ml_simulation/test_runner.py
# THOROUGH temporary script to test the quantum simulation data generation part
# This version checks for validation, entanglement, and runs in different modes.

import os
import numpy as np
import time
import traceback
import sys

# Add the parent directory to the Python path to allow running this script directly
# while still resolving package imports. This is an alternative to `python -m`.
# However, running with `python -m` from the project root is still the recommended
# way for package execution. This block is mainly for convenience if running
# `python test_runner.py` from inside `quantum_ml_simulation`.
# For consistent package behavior, run from project root:
# python -m quantum_ml_simulation.test_runner
try:
    from .simulations.test_simulation import TestSimulation
    from .simulations.ising_model import IsingModelSimulation
    from .config import simulation_params as cfg
    from .quantum_runner.simulator import QuantumSimulator # Import directly for mode testing
    import qiskit
    from qiskit.quantum_info import Statevector
except ImportError:
    # Fallback imports if running directly from the file's directory
    # This makes the script runnable as `python test_runner.py` from inside
    # quantum_ml_simulation/, but it's less robust than `python -m`.
    # If using `python -m`, this block is skipped.
    print("Running with fallback imports. Please use `python -m quantum_ml_simulation.test_runner` from project root for best results.")
    import sys
    from pathlib import Path
    # Add the directory containing 'quantum_ml_simulation' to the path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from quantum_ml_simulation.simulations.test_simulation import TestSimulation
    from quantum_ml_simulation.simulations.ising_model import IsingModelSimulation
    from quantum_ml_simulation.config import simulation_params as cfg
    from quantum_ml_simulation.quantum_runner.simulator import QuantumSimulator
    import qiskit
    from qiskit.quantum_info import Statevector


# --- Helper Functions for Specific Checks ---

def check_initial_state(simulation_class, n_qubits: int, expected_type: str):
    """Tests if the initial state is prepared correctly."""
    print(f"\n--- Testing Initial State: {expected_type} ({n_qubits} qubits) ---")
    original_n_steps_range = cfg.N_STEPS_RANGE
    original_init_state = cfg.INITIAL_STATE_TYPE

    try:
        # Temporarily set config for test
        cfg.INITIAL_STATE_TYPE = expected_type
        cfg.N_STEPS_RANGE = [0] # Test only the initial circuit with 0 steps

        sim = simulation_class()
        params_space = sim.get_parameter_space()

        if not params_space:
            print("  SKIP: No parameter space found for initial state test (e.g., ranges empty).")
            return

        # Pick one parameter set with n_steps = 0
        test_params = None
        for p in params_space:
            if p[0] == 0: # Assumes n_steps is the first parameter
                test_params = p
                break

        if test_params is None:
            print("  SKIP: Could not find a parameter set with n_steps = 0.")
            return

        print(f"  Building circuit for params with n=0: {test_params}")
        circuit = sim.build_circuit_for_params(test_params)
        print(f"  Circuit depth: {circuit.depth()}, size: {circuit.size()}")

        # Simulate to get the statevector
        # Use a temporary simulator instance in statevector mode
        temp_simulator = QuantumSimulator(simulation_mode="statevector", n_shots=1) # Shots irrelevant here
        statevector = temp_simulator.run_circuit_and_get_statevector(circuit)

        if statevector is None:
            print("  FAIL: Could not obtain statevector.")
            return

        # Define expected statevector
        if expected_type == 'zero':
            expected_sv = Statevector([1.0] + [0.0] * (2**n_qubits - 1))
        elif expected_type == 'superposition':
            expected_sv = Statevector([1.0/np.sqrt(2**n_qubits)] * (2**n_qubits))
        else:
            print(f"  SKIP: Unexpected initial state type '{expected_type}'.")
            return

        # Compare statevectors
        is_correct = statevector.is_approximately_equal(expected_sv, rtol=1e-8, atol=1e-8)

        if is_correct:
            print(f"  PASS: Initial state correctly prepared as {expected_type}.")
        else:
            print(f"  FAIL: Initial state mismatch for {expected_type}.")
            print("    Actual statevector sample:", statevector.data[:min(8, 2**n_qubits)]) # Print first few elements
            print("    Expected statevector sample:", expected_sv.data[:min(8, 2**n_qubits)])


    except Exception as e:
        print(f"  ERROR during initial state test: {e}")
        traceback.print_exc()
    finally:
        # Restore original config
        cfg.N_STEPS_RANGE = original_n_steps_range
        cfg.INITIAL_STATE_TYPE = original_init_state
        print("-" * 30)


def analyze_simulation_results(dataset, simulation_name, sim):
    """Analyzes the dataset produced by generate_dataset."""
    print(f"\n--- Analyzing Dataset for: {simulation_name} ---")

    if not dataset:
        print("  No data points to analyze.")
        return

    # Basic check: All data points have expected keys
    expected_keys = ['input', 'output', 'validation_diff', 'entanglement'] # Based on BaseSimulation result dict
    if not all(all(key in d for key in expected_keys) for d in dataset):
         print("  WARNING: Not all data points contain the expected keys.")
         # Attempt to proceed with available data

    # Check expectation values
    outputs = np.array([d['output'] for d in dataset])
    valid_outputs = outputs[~np.isnan(outputs)]
    print(f"  Valid Expectation Values: {len(valid_outputs)}/{len(dataset)}")
    if len(valid_outputs) > 0:
        print(f"    Min/Max: {np.min(valid_outputs):.4f} / {np.max(valid_outputs):.4f}")
        print(f"    Mean:    {np.mean(valid_outputs):.4f}")
        # Check if values are within [-1, 1] range for Pauli expectation values
        if np.any(valid_outputs < -1.01) or np.any(valid_outputs > 1.01): # Add small tolerance
             print("    WARNING: Some expectation values outside [-1, 1] range.")
        else:
             print("    Expectation values within expected [-1, 1] range.")
    else:
        print("    No valid expectation values found.")

    # Check Validation Difference
    if 'validation_diff' in dataset[0]: # Check if validation was attempted
        diffs = np.array([d['validation_diff'] for d in dataset])
        valid_diffs = diffs[~np.isnan(diffs)]
        print(f"  Trotter Validation Differences: {len(valid_diffs)}/{len(dataset)}")
        if len(valid_diffs) > 0:
            avg_abs_diff = np.mean(np.abs(valid_diffs))
            max_abs_diff = np.max(np.abs(valid_diffs))
            print(f"    Avg Abs Diff: {avg_abs_diff:.4e}, Max Abs Diff: {max_abs_diff:.4e}")
            points_above_tolerance = sum(1 for d in dataset if not np.isnan(d['validation_diff']) and abs(d['validation_diff']) > cfg.VALIDATION_TOLERANCE)
            print(f"    Points > Tolerance ({cfg.VALIDATION_TOLERANCE:.4e}): {points_above_tolerance}/{len(valid_diffs)}")

            # Check hypothesis: Error increases with n_steps
            try:
                n_step_index = dataset[0]['input'].index(next(f for f in sim.get_ml_input_feature_names() if f == 'n_steps')) # Hacky way to find n_steps index
                n_steps_values = np.array([d['input'][n_step_index] for d in dataset])

                # Group diffs by n_steps and check mean absolute difference trend
                error_by_n = {}
                for i in range(len(dataset)):
                    n = n_steps_values[i]
                    diff = dataset[i]['validation_diff']
                    if not np.isnan(diff):
                        if n not in error_by_n:
                            error_by_n[n] = []
                        error_by_n[n].append(abs(diff))

                if error_by_n:
                    n_values = sorted(error_by_n.keys())
                    mean_abs_errors = [np.mean(error_by_n[n]) for n in n_values]

                    print("    Mean Absolute Validation Diff vs n_steps:")
                    for n, err in zip(n_values, mean_abs_errors):
                        print(f"      n={int(n)}: {err:.4e}")

                    # Simple check if error generally increases
                    # Check if mean error at max n_steps is significantly higher than at min n_steps
                    if len(n_values) >= 2:
                        min_n_err = mean_abs_errors[0]
                        max_n_err = mean_abs_errors[-1]
                        print(f"    Trend Check: Mean Error at n={int(n_values[0])} is {min_n_err:.4e}, at n={int(n_values[-1])} is {max_n_err:.4e}")
                        if max_n_err > min_n_err * 2: # Simple heuristic for "significantly higher"
                             print("    Observation: Error generally increases with n_steps (supports hypothesis).")
                        else:
                             print("    Observation: Error trend with n_steps is not strongly increasing based on simple check.")
            except (ValueError, StopIteration, IndexError, KeyError) as e:
                 print(f"    Could not analyze error vs n_steps: {e}")


    # Check Entanglement Entropy
    if 'entanglement' in dataset[0]: # Check if entanglement was attempted
         entropies = np.array([d['entanglement'] for d in dataset])
         valid_entropies = entropies[~np.isnan(entropies)]
         print(f"  Entanglement Entropies: {len(valid_entropies)}/{len(dataset)}")
         if len(valid_entropies) > 0:
             print(f"    Min/Max: {np.min(valid_entropies):.4f} / {np.max(valid_entropies):.4f}")
             print(f"    Mean:    {np.mean(valid_entropies):.4f}")
             # Max possible entanglement for a bipartite cut of size k is log2(min(2^k, 2^(N-k)))
             max_possible_ent = np.log2(min(2**cfg.ENTANGLEMENT_PARTITION, 2**(cfg.N_QUBITS - cfg.ENTANGLEMENT_PARTITION)))
             print(f"    Max possible for partition size {cfg.ENTANGLEMENT_PARTITION}: {max_possible_ent:.4f}")
             # Check if entanglement is non-zero (indicative of non-trivial dynamics/superposition state working)
             if np.mean(valid_entropies) > 0.01: # Heuristic
                  print("    Observation: Non-zero entanglement observed (indicates complex dynamics).")
             else:
                  print("    Observation: Entanglement values are close to zero.")

         else:
              print("    No valid entanglement entropies found.")

    print("-" * 30)


# --- Main Test Runner Logic ---
def run_full_simulation_test_suite():
    """Runs multiple tests on the simulation components."""
    print("Starting Comprehensive Quantum Simulation Component Test Suite...")

    # --- Pre-Checks ---
    try:
        from qiskit_aer import AerSimulator
        print("Qiskit Aer simulator found.")
    except ImportError:
        print("\nERROR: Qiskit Aer not found. Please install it ('pip install qiskit-aer').")
        print("Cannot run quantum simulations without it. Exiting.")
        sys.exit(1) # Exit with error code

    try:
        # Check for SciPy needed for ED validation
        import scipy.linalg
        print("SciPy found (needed for exact diagonalization validation).")
    except ImportError:
        print("\nWARNING: SciPy not found. Exact diagonalization validation will be skipped.")
        cfg.VALIDATE_TROTTER = False # Disable validation if scipy is missing

    # Ensure output directories exist (they should be created by config on import)
    os.makedirs(cfg.DATA_PATH, exist_ok=True)
    os.makedirs(cfg.ML_RESULTS_PATH, exist_ok=True)
    print(f"Output directories checked/created relative to current directory: {cfg.BASE_SAVE_PATH}")
    print("="*40)


    # --- Configure Simulations to Test ---
    simulations_to_test = [
        (TestSimulation, "TestSim"),
        (IsingModelSimulation, "IsingModel"),
        # Add other simulations here as they are implemented
        # (SpinChainPotentialSimulation, "SpinChainPotential"),
        # (DimerizedHeisenbergSimulation, "DimerizedHeisenberg"),
    ]

    # --- Test Initial States ---
    # Temporarily save original config values
    original_n_qubits = cfg.N_QUBITS
    original_initial_state_type = cfg.INITIAL_STATE_TYPE

    # Test |0...0> initial state (default for Qiskit)
    # For TestSim (1 qubit)
    cfg.N_QUBITS = cfg.TEST_SIM_PARAMS["n_qubits"]
    cfg.INITIAL_STATE_TYPE = 'zero' # Explicitly set for the test function
    check_initial_state(TestSimulation, cfg.N_QUBITS, 'zero')

    # For IsingModel (cfg.N_QUBITS)
    cfg.N_QUBITS = original_n_qubits # Restore N_QUBITS for Ising
    cfg.INITIAL_STATE_TYPE = 'zero'
    check_initial_state(IsingModelSimulation, cfg.N_QUBITS, 'zero')


    # Test |+...+\> initial state
    # For TestSim (1 qubit)
    cfg.N_QUBITS = cfg.TEST_SIM_PARAMS["n_qubits"]
    cfg.INITIAL_STATE_TYPE = 'superposition'
    check_initial_state(TestSimulation, cfg.N_QUBITS, 'superposition')

     # For IsingModel (cfg.N_QUBITS)
    cfg.N_QUBITS = original_n_qubits # Restore N_QUBITS for Ising
    cfg.INITIAL_STATE_TYPE = 'superposition'
    check_initial_state(IsingModelSimulation, cfg.N_QUBITS, 'superposition')

    # Restore original initial state config for main simulation tests
    cfg.INITIAL_STATE_TYPE = original_initial_state_type
    cfg.N_QUBITS = original_n_qubits
    print("="*40)


    # --- Test Simulation Data Generation in Different Modes ---
    simulation_modes_to_test = ["statevector", "shots"] # Test both

    for sim_class, sim_name in simulations_to_test:
        for mode in simulation_modes_to_test:
            print(f"\n{'*'*50}")
            print(f"Running {sim_name} in {mode.upper()} mode")
            print(f"{'*'*50}")

            # Temporarily set the simulation mode in config
            original_sim_mode = cfg.SIMULATION_MODE
            cfg.SIMULATION_MODE = mode

            # Re-instantiate simulator in case backend needs changing
            # This is handled within BaseSimulation.__init__ now.
            # Need to re-instantiate the *simulation* object to pick up the new mode config.
            # sim = sim_class() # Instantiated inside run_simulation_and_analyze

            # Run the data generation for this simulation and mode
            # The generate_dataset method prints progress and basic stats
            # The run_single_simulation method within generate_dataset handles errors and NaNs
            # DataHandler will be used later for saving, just generating dataset here.

            # Call a helper to run generation and then analyze the resulting dataset structure/content
            dataset = []
            sim_instance = None
            try:
                 sim_instance = sim_class() # Instantiate *after* setting mode
                 dataset = sim_instance.generate_dataset()
            except Exception as e:
                 print(f"\nFATAL ERROR during {sim_name} data generation in {mode} mode: {e}")
                 traceback.print_exc()


            # Analyze the generated dataset structure and contents
            analyze_simulation_results(dataset, f"{sim_name} ({mode.upper()} Mode)", sim_instance)

            # Restore original simulation mode config
            cfg.SIMULATION_MODE = original_sim_mode
            print(f"{'*'*50}")

    print("\nComprehensive Quantum Simulation Component Test Suite Finished.")

# --- Main Execution ---
if __name__ == "__main__":
    run_full_simulation_test_suite()