# quantum_ml_simulation/generate_data.py
# Orchestrates step-by-step data generation with sampling options

import argparse
import os
import sys
import time
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json # <--------------------- ADD THIS IMPORT

# Use relative imports for cleaner structure
from .config import simulation_params as cfg
from .data_management.data_handler import DataHandler

# Placeholder for analyze_data - use dedicated script instead
def analyze_data(dataset_path: str, data_handler: DataHandler):
    print(f"\n--- Basic Dataset Check: {os.path.basename(dataset_path)} ---")
    print("Run 'analyze_dataset.py' script for detailed analysis and plots.")
    exists = data_handler.check_data_exists(dataset_path)
    print(f"Dataset CSV file exists: {exists}")
    if exists:
         # Try loading metadata for basic info
         sim_data_dir = data_handler.get_sim_data_path(dataset_path)
         metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
         if os.path.exists(metadata_filepath):
             try:
                 # *** This line needs the json import ***
                 with open(metadata_filepath, 'r') as f: metadata = json.load(f)
                 print(f"  Metadata Check: Found simulation '{metadata.get('simulation_name', 'N/A')}' "
                       f"with {metadata.get('n_qubits', 'N/A')} qubits, "
                       f"{len(metadata.get('expected_ml_features',[]))} features, "
                       f"{len(metadata.get('output_observable_names',[]))} outputs.")
             except Exception as e: print(f"  Metadata Check: Error loading metadata - {e}")
         else: print("  Metadata Check: metadata.json not found.")

# --- rest of the main() function remains the same ---
def main():
    parser = argparse.ArgumentParser(description="Quantum Simulation Time-Series Data Generation")
    parser.add_argument("--simulation", required=True, choices=cfg.SIMULATION_CONFIGS.keys(),
                        help="Name of the simulation type to run.")
    # Core parameter overrides
    parser.add_argument("--n_qubits", type=int, default=None, help="Override N_QUBITS.")
    parser.add_argument("--delta_t", type=float, default=None, help="Override DELTA_T.")
    parser.add_argument("--max_n_steps", type=int, default=None, help="Override max n_steps to record.")
    # Sampling strategy overrides
    parser.add_argument("--sampling_method", choices=['grid', 'random'], default=cfg.SAMPLING_METHOD,
                        help=f"Parameter sampling method (default: {cfg.SAMPLING_METHOD}).")
    parser.add_argument("--num_random_sets", type=int, default=cfg.DEFAULT_NUM_RANDOM_PARAMETER_SETS,
                        help="Number of parameter sets for 'random' sampling.")
    # Initial state override
    parser.add_argument("--initial_state_type", choices=['zero', 'superposition', 'random_rotations'], default=None,
                        help="Override initial state type generation.")
    # Other controls
    parser.add_argument("--force", action="store_true", help="Force regeneration even if data exists.")

    args = parser.parse_args()

    # --- Determine Parameters ---
    sim_config = cfg.SIMULATION_CONFIGS[args.simulation]

    # Get base parameters, allowing overrides
    n_qubits = args.n_qubits if args.n_qubits is not None else sim_config["default_sampling_ranges"]["n_qubits"]
    delta_t = args.delta_t if args.delta_t is not None else cfg.DELTA_T
    n_steps_record_points = list(range(1, args.max_n_steps + 1)) if args.max_n_steps is not None \
                        else sim_config["default_sampling_ranges"].get("n_steps", cfg.N_STEPS_RECORD_POINTS)
    n_steps_max = n_steps_record_points[-1]
    simulation_mode = cfg.SIMULATION_MODE # Keep global for now

    # Determine initial state config
    initial_state_config = cfg.INITIAL_STATE_CONFIG.copy()
    if args.initial_state_type is not None:
        initial_state_config['type'] = args.initial_state_type

    # Determine sampling method and number of sets
    sampling_method = args.sampling_method
    num_parameter_sets = args.num_random_sets if sampling_method == 'random' else None # None for grid, calculated later

    # Get the simulation-specific parameter names and their sampling ranges/values
    system_param_names = sim_config["params"] # e.g., ['J', 'B']
    param_config = sim_config["default_sampling_ranges"]

    # Construct dataset path
    dataset_path = cfg.get_dataset_path(args.simulation, n_qubits, delta_t, n_steps_max)
    # Optional: Add more identifiers to path based on sampling/initial state?
    # dataset_path += f"_{sampling_method}_{initial_state_config['type']}"
    print(f"Target Dataset Path: {dataset_path}")

    data_handler = DataHandler()

    # --- Check if Data Exists ---
    if not args.force and data_handler.check_data_exists(dataset_path):
        print("Data already exists. Use --force to regenerate.")
        analyze_data(dataset_path, data_handler) # Basic check
        return

    # --- Data Generation ---
    print(f"\n--- Generating Time-Series Data for: {args.simulation} ---")
    print(f"Parameters: N={n_qubits}, dt={delta_t}, Record Steps={n_steps_record_points[0]}-{n_steps_max}")
    print(f"Initial State: {initial_state_config['type']}, Sim Mode: {simulation_mode}")
    print(f"Sampling Method: {sampling_method}" + (f", Num Sets: {num_parameter_sets}" if sampling_method == 'random' else ""))
    # Print parameter ranges/values being used
    print("Parameter Configuration:")
    for p_name in system_param_names:
         print(f"  - {p_name}: {param_config.get(p_name, 'N/A')}")


    # --- Dynamically Import and Instantiate Simulation Class ---
    try:
        module_path, class_name = sim_config["class_path"].rsplit('.', 1)
        SimModule = importlib.import_module(module_path)
        SimClass = getattr(SimModule, class_name)
    except Exception as e:
        print(f"Error importing simulation class {sim_config['class_path']}: {e}")
        sys.exit(1)

    # Instantiate the simulation, passing configuration
    try:
        simulation = SimClass(
            n_qubits=n_qubits,
            delta_t=delta_t,
            n_steps_range=n_steps_record_points, # The points to record
            initial_state_config=initial_state_config, # Pass the whole dict
            simulation_mode=simulation_mode,
            sampling_method=sampling_method, # Pass sampling method
            num_parameter_sets=num_parameter_sets, # Pass num sets (relevant for random)
            parameter_config=param_config, # Pass the dict with sampling ranges/values
            **sim_config.get("extra_args", {}) # Pass fixed args like fixed_J
        )
    except Exception as e:
        print(f"Error instantiating simulation class {class_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nStarting quantum simulation trajectories...")
    start_time = time.time()
    dataset = simulation.generate_dataset() # generate_dataset uses the sampling config internally
    end_time = time.time()
    print(f"Simulation trajectories completed in {end_time - start_time:.2f} seconds.")

    if not dataset:
        print("Error: No data generated by the simulation.")
        return

    # --- Prepare for Saving ---
    # Get names needed for metadata and saving from the instance
    expected_ml_features = simulation.get_ml_input_feature_names()
    output_observable_names = simulation.get_output_observable_names()

    # Construct metadata dictionary
    metadata = {
        "simulation_name": args.simulation,
        "n_qubits": n_qubits,
        "delta_t": delta_t,
        "n_steps_range": n_steps_record_points,
        "initial_state_config": initial_state_config, # Save initial state config used
        "simulation_mode": simulation_mode,
        "sampling_method": sampling_method, # Save sampling method used
        "num_parameter_sets_requested": num_parameter_sets, # Num requested for random
        "param_config_used": param_config, # Save ranges/values used
        "fixed_params": sim_config.get("extra_args", {}),
        "expected_ml_features": expected_ml_features,
        "output_observable_names": output_observable_names,
        "generation_script": os.path.basename(__file__),
        # generation_timestamp added by data_handler
    }

    # Save data and metadata
    print("\nSaving data and metadata...")
    data_handler.save_simulation_data_with_metadata(
        dataset=dataset,
        metadata=metadata,
        dataset_path=dataset_path,
        feature_names=expected_ml_features,
        output_observable_names=output_observable_names
    )

    print("\nData generation complete.")
    analyze_data(dataset_path, data_handler) # Basic check after saving

if __name__ == "__main__":
    main()