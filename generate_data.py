# quantum_ml_simulation/generate_data.py
# Orchestrates step-by-step data generation

import argparse
import os
import sys
import time
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use relative imports for cleaner structure
from .config import simulation_params as cfg
from .data_management.data_handler import DataHandler

def analyze_data(dataset_path: str, data_handler: DataHandler):
    """Loads data and performs basic analysis (Now uses analyze_dataset.py)."""
    print(f"\n--- Analyzing Dataset: {os.path.basename(dataset_path)} ---")
    print("Please use the dedicated 'analyze_dataset.py' script for detailed analysis.")
    # Minimal check here:
    X_df, y_df, metadata = data_handler.load_simulation_data_and_metadata(
        dataset_path,
        # No single target column now, need to load all output columns
        # load_simulation_data_and_metadata needs adjustment if we want y_df here
        # For now, just load features and metadata
        load_target_vector=False # Hypothetical flag to just get X and meta
    )

    if metadata:
        print("\n[Metadata Snippet]")
        print(f"  Simulation Name: {metadata.get('simulation_name', 'N/A')}")
        print(f"  N Qubits: {metadata.get('n_qubits', 'N/A')}")
        print(f"  Delta T: {metadata.get('delta_t', 'N/A')}")
        print(f"  Recorded Steps Range: {metadata.get('n_steps_range', 'N/A')}")
        print(f"  ML Input Features: {metadata.get('expected_ml_features', 'N/A')}")
        print(f"  ML Output Observables: {metadata.get('output_observable_names', 'N/A')}")
    else:
        print("\nNo metadata found.")

    if X_df is None:
        print("\nCould not load feature data for basic check.")
    else:
        print(f"\n[Data Summary]")
        print(f"  Shape (Features X): {X_df.shape}")
        print(f"  Feature Columns: {X_df.columns.tolist()}")
        print(f"\nRun 'analyze_dataset.py --dataset_id {os.path.basename(dataset_path)}' for full analysis.")


def main():
    parser = argparse.ArgumentParser(description="Quantum Simulation Time-Series Data Generation")
    parser.add_argument("--simulation", required=True, choices=cfg.SIMULATION_CONFIGS.keys(),
                        help="Name of the simulation type to run (e.g., IsingModel).")
    parser.add_argument("--n_qubits", type=int, default=None, help="Override default N_QUBITS.")
    parser.add_argument("--delta_t", type=float, default=None, help="Override default DELTA_T.")
    parser.add_argument("--max_n_steps", type=int, default=None, help="Maximum number of steps (n) to simulate and record.")
    parser.add_argument("--initial_state", choices=['zero', 'superposition'], default=None, help="Override default INITIAL_STATE_TYPE.")
    # Add arguments for J_RANGE, B_RANGE etc. if needed
    parser.add_argument("--force", action="store_true", help="Force regeneration even if data exists.")
    # Removed --analyze, use separate script

    args = parser.parse_args()

    # --- Determine Parameters ---
    sim_config = cfg.SIMULATION_CONFIGS[args.simulation]
    # Use CLI args > sim_config defaults > global defaults
    n_qubits = args.n_qubits if args.n_qubits is not None else sim_config["default_ranges"].get("n_qubits", cfg.N_QUBITS)
    delta_t = args.delta_t if args.delta_t is not None else sim_config.get("delta_t", cfg.DELTA_T) # Allow dt per sim?
    n_steps_record_points = list(range(1, args.max_n_steps + 1)) if args.max_n_steps is not None \
                        else sim_config["default_ranges"].get("n_steps", cfg.N_STEPS_RANGE)
    n_steps_max = n_steps_record_points[-1]
    initial_state_type = args.initial_state if args.initial_state is not None else cfg.INITIAL_STATE_TYPE
    simulation_mode = cfg.SIMULATION_MODE # Keep global for now

    # Parameter ranges specific to the simulation (use defaults from sim_config)
    # The `params` key in SIMULATION_CONFIGS lists *all* params (incl n_steps)
    # We need the ranges for params *other than* n_steps
    varying_param_names = [p for p in sim_config["params"] if p != 'n_steps']
    current_param_ranges = {}
    for p_name in varying_param_names:
         if p_name in sim_config["default_ranges"]:
              current_param_ranges[p_name] = sim_config["default_ranges"][p_name]
         else:
              # Try finding in global ranges (e.g., J_RANGE, B_RANGE) - less ideal
              # Best practice is to define all needed ranges within sim_config["default_ranges"]
              print(f"Warning: Range for parameter '{p_name}' not found in SIMULATION_CONFIGS[{args.simulation}]['default_ranges'].")
              # Attempt to find a global range - THIS IS BRITTLE
              global_range_name = f"{p_name.upper()}_RANGE"
              if hasattr(cfg, global_range_name):
                    current_param_ranges[p_name] = getattr(cfg, global_range_name)
                    print(f"  -> Found global range '{global_range_name}'.")
              else:
                    print(f"  -> ERROR: Cannot find range for '{p_name}'. Exiting.")
                    sys.exit(1)


    # Construct dataset path based on *actual* parameters being used
    dataset_path = cfg.get_dataset_path(args.simulation, n_qubits, delta_t, n_steps_max)
    print(f"Target Dataset Path: {dataset_path}")

    data_handler = DataHandler()

    # --- Check if Data Exists ---
    if not args.force and data_handler.check_data_exists(dataset_path):
        print("Data already exists. Use --force to regenerate.")
        # Optionally run analysis here or just exit
        # analyze_data(dataset_path, data_handler) # Call basic check
        return

    # --- Data Generation Mode ---
    print(f"\n--- Generating Time-Series Data for: {args.simulation} ---")
    print(f"Parameters: N={n_qubits}, dt={delta_t}, Record Steps={n_steps_record_points[0]}-{n_steps_max}")
    print(f"Initial State: {initial_state_type}, Sim Mode: {simulation_mode}")
    print(f"Parameter Ranges: {current_param_ranges}")


    # --- Dynamically Import and Instantiate Simulation Class ---
    try:
        module_path, class_name = sim_config["class_path"].rsplit('.', 1)
        SimModule = importlib.import_module(module_path)
        SimClass = getattr(SimModule, class_name)
    except Exception as e:
        print(f"Error importing simulation class {sim_config['class_path']}: {e}")
        sys.exit(1)

    # Instantiate the simulation, passing ALL required parameters
    try:
        simulation = SimClass(
            n_qubits=n_qubits,
            # measurement_operator=sim_config["measurement_operator"], # Removed
            delta_t=delta_t,
            n_steps_range=n_steps_record_points, # Pass the points to record
            initial_state_type=initial_state_type,
            simulation_mode=simulation_mode,
            param_ranges=current_param_ranges, # Pass the ranges dict
            **sim_config.get("extra_args", {}) # Pass extra args like fixed_J
        )
    except Exception as e:
        print(f"Error instantiating simulation class {class_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nStarting quantum simulation trajectories...")
    start_time = time.time()
    # generate_dataset now runs the step-by-step simulation internally
    dataset = simulation.generate_dataset()
    end_time = time.time()
    print(f"Simulation trajectories completed in {end_time - start_time:.2f} seconds.")

    if not dataset:
        print("Error: No data generated by the simulation.")
        return

    # --- Prepare for Saving ---
    # Get expected feature names and output observable names from the instance
    expected_ml_features = simulation.get_ml_input_feature_names()
    output_observable_names = simulation.get_output_observable_names()

    # Construct metadata dictionary
    metadata = {
        "simulation_name": args.simulation,
        "n_qubits": n_qubits,
        "delta_t": delta_t,
        "n_steps_range": n_steps_record_points, # Record the actual points used
        # "measurement_operator": sim_config["measurement_operator"], # Maybe keep for context?
        "initial_state_type": initial_state_type,
        "simulation_mode": simulation_mode,
        "param_ranges_used": current_param_ranges, # Record ranges used
        "fixed_params": sim_config.get("extra_args", {}), # Record fixed params
        "expected_ml_features": expected_ml_features,
        "output_observable_names": output_observable_names, # CRUCIAL for loading later
        "generation_script": os.path.basename(__file__),
        # generation_timestamp added by data_handler
    }

    # Save data and metadata
    print("\nSaving data and metadata...")
    data_handler.save_simulation_data_with_metadata(
        dataset=dataset, # The list of dicts from generate_dataset
        metadata=metadata,
        dataset_path=dataset_path,
        # Feature names passed here are used to structure the CSV/DF correctly
        feature_names=expected_ml_features,
        # Output observable names are saved in metadata, used during loading
        # Need to modify save_simulation_data... to use output_observable_names for cols
        output_observable_names=output_observable_names
    )

    print("\nData generation complete.")
    # print(f"\nRun analysis using: python -m quantum_ml_simulation.analyze_dataset --dataset_id {os.path.basename(dataset_path)}")

if __name__ == "__main__":
    main()