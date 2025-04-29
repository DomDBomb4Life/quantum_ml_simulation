# quantum_ml_simulation/test_runner.py
# Temporary script to test the quantum simulation data generation part

import os
import numpy as np
import time
import traceback # Import traceback for detailed error printing

# Import the simulation classes and config using EXPLICIT relative imports
# When running with `python -m quantum_ml_simulation.test_runner` from the parent directory,
# the '.' refers to the 'quantum_ml_simulation' package root.
from .simulations.test_simulation import TestSimulation
from .simulations.ising_model import IsingModelSimulation
from .config import simulation_params as cfg

def run_simulation_test(simulation_class, simulation_name):
    """
    Runs the generate_dataset method for a given simulation class
    and prints basic information about the result.
    """
    print(f"\n{'='*30}")
    print(f"Testing Simulation: {simulation_name}")
    print(f"{'='*30}")

    sim = None
    try:
        # 1. Instantiate the simulation
        sim = simulation_class()
        print(f"Simulation object created for {simulation_name}.")

        # 2. Generate the dataset
        start_time = time.time()
        dataset = sim.generate_dataset()
        end_time = time.time()
        print(f"Dataset generation for {simulation_name} finished in {end_time - start_time:.2f} seconds.")

        # 3. Report results
        if not dataset:
            print(f"Generated an EMPTY dataset for {simulation_name}.")
            return

        print(f"Successfully generated {len(dataset)} data points.")

        # Count successful vs failed points (NaN outputs)
        successful_points = sum(1 for d in dataset if not np.isnan(d['output']))
        failed_points = len(dataset) - successful_points
        print(f"  Successful data points: {successful_points}")
        if failed_points > 0:
            print(f"  Failed data points (NaN output): {failed_points}")

        # Print the first few data points
        print("\nSample data points:")
        num_samples = min(5, len(dataset))
        for i in range(num_samples):
            print(f"  Point {i+1}:")
            print(f"    Raw Params: {dataset[i]['raw_params']}")
            print(f"    ML Input:   {dataset[i]['input']}")
            print(f"    Output:     {dataset[i]['output']:.6f}" if not np.isnan(dataset[i]['output']) else "    Output:     NaN")

        # Verify input feature count matches expected
        expected_feature_count = len(sim.get_ml_input_feature_names())
        if successful_points > 0 and len(dataset[0]['input']) != expected_feature_count:
             print(f"WARNING: ML input feature count ({len(dataset[0]['input'])}) does not match expected count ({expected_feature_count}) from get_ml_input_feature_names!")
        elif successful_points > 0:
             print(f"\nML input feature names: {sim.get_ml_input_feature_names()}")
             print(f"ML input feature count: {len(dataset[0]['input'])} (Matches get_ml_input_feature_names)")


        print(f"{'-'*30}")

    except Exception as e:
        print(f"\n{'!'*30}")
        print(f"AN ERROR OCCURRED during test for {simulation_name}:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc() # Print detailed traceback
        print(f"{'!'*30}")

# --- Main execution block for the test ---
if __name__ == "__main__":
    print("Starting Quantum Simulation Component Test (using explicit relative imports)...")

    # Check if Qiskit Aer is available early
    try:
        from qiskit_aer import AerSimulator
        print("Qiskit Aer simulator found.")
    except ImportError:
        print("\nERROR: Qiskit Aer not found. Please install it ('pip install qiskit-aer').")
        print("Cannot run quantum simulations without it.")
        exit() # Exit if the core requirement is missing

    # Ensure output directories exist (they should be created by config)
    # Note: This will create ./project_results_mvp relative to the *parent* directory,
    # which is where you should be running the script from.
    os.makedirs(cfg.DATA_PATH, exist_ok=True)
    os.makedirs(cfg.ML_RESULTS_PATH, exist_ok=True)
    print(f"Output directories checked/created relative to current directory: {cfg.BASE_SAVE_PATH}")


    # Run the test for TestSimulation
    run_simulation_test(TestSimulation, "TestSim")

    # Run the test for IsingModelSimulation
    run_simulation_test(IsingModelSimulation, "IsingModel")

    print("\nQuantum Simulation Component Test Finished.")