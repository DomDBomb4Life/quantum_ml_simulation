# quantum_ml_simulation/generate_data.py
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
    """Loads data and performs basic analysis."""
    print(f"\n--- Analyzing Dataset: {os.path.basename(dataset_path)} ---")
    X_df, y_series, metadata = data_handler.load_simulation_data_and_metadata(dataset_path)

    if metadata:
        print("\n[Metadata]")
        for key, value in metadata.items():
            # Truncate long lists for display
            display_value = value
            if isinstance(value, list) and len(value) > 10:
                 display_value = f"[{value[0]}, ..., {value[-1]}] (Length: {len(value)})"
            print(f"  {key}: {display_value}")
    else:
        print("\nNo metadata found.")
        # Attempt to infer parameters from path if possible
        try:
            parts = os.path.basename(dataset_path).split('_')
            sim_name = parts[0]
            n_q = int(parts[1][1:])
            dt = float(parts[2][2:])
            n_max = int(parts[3][4:])
            print(f"Inferred from path: Sim={sim_name}, N={n_q}, dt={dt}, n_max={n_max}")
        except Exception:
             print("Could not infer parameters from path.")


    if X_df is None or y_series is None:
        print("\n[Data Analysis]")
        print("Could not load data for analysis.")
        return

    print("\n[Data Summary]")
    print(f"  Shape (Features X): {X_df.shape}")
    print(f"  Target (y) Length: {len(y_series)}")
    print(f"  Feature Columns: {X_df.columns.tolist()}")

    print("\n[Target Variable Statistics (output)]")
    print(y_series.describe().to_string())

    print("\n[Feature Statistics]")
    print(X_df.describe().to_string())

    print("\n[Missing Values Check]")
    print("Features (X):")
    print(X_df.isnull().sum().to_string())
    print("\nTarget (y):")
    print(f"  NaN count: {y_series.isnull().sum()}") # Should be 0 after filtering in load

    # --- Basic Plots (Optional) ---
    # Example: Histogram of the target variable
    plt.figure(figsize=(8, 5))
    plt.hist(y_series, bins=30, edgecolor='k', alpha=0.7)
    plt.title(f"Histogram of Target Variable ('{y_series.name}')")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    analysis_plot_path = os.path.join(data_handler.get_sim_data_path(dataset_path), "target_histogram.png")
    try:
         plt.savefig(analysis_plot_path)
         print(f"\nTarget histogram saved to: {os.path.relpath(analysis_plot_path)}")
         plt.close()
    except Exception as e:
         print(f"Could not save histogram: {e}")
         # plt.show() # Optionally show if saving fails


def main():
    parser = argparse.ArgumentParser(description="Quantum Simulation Data Generation and Analysis")
    parser.add_argument("--simulation", required=True, choices=cfg.SIMULATION_CONFIGS.keys(),
                        help="Name of the simulation type to run (e.g., IsingModel).")
    parser.add_argument("--n_qubits", type=int, default=None, help="Override default N_QUBITS.")
    parser.add_argument("--delta_t", type=float, default=None, help="Override default DELTA_T.")
    parser.add_argument("--max_n_steps", type=int, default=None, help="Override default max n_steps.")
    # Add arguments for J_RANGE, B_RANGE if needed, but gets complex. Better to modify config.
    parser.add_argument("--force", action="store_true", help="Force regeneration even if data exists.")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing data instead of generating.")

    args = parser.parse_args()

    # --- Determine Parameters ---
    sim_config = cfg.SIMULATION_CONFIGS[args.simulation]
    n_qubits = args.n_qubits if args.n_qubits is not None else sim_config["default_ranges"]["n_qubits"]
    delta_t = args.delta_t if args.delta_t is not None else cfg.DELTA_T # Use global default dt
    n_steps_range = list(range(1, args.max_n_steps + 1)) if args.max_n_steps is not None \
                     else sim_config["default_ranges"]["n_steps"]
    n_steps_max = n_steps_range[-1]

    # Construct dataset path based on *actual* parameters being used
    dataset_path = cfg.get_dataset_path(args.simulation, n_qubits, delta_t, n_steps_max)
    print(f"Target Dataset Path: {dataset_path}")

    data_handler = DataHandler()

    # --- Mode Selection ---
    if args.analyze:
        if data_handler.check_data_exists(dataset_path):
            analyze_data(dataset_path, data_handler)
        else:
            print(f"Error: Cannot analyze. Data not found at expected location: {data_handler.get_sim_data_path(dataset_path)}")
        return # Exit after analysis

    # --- Data Generation Mode ---
    print(f"\n--- Generating Data for: {args.simulation} ---")
    print(f"Parameters: N={n_qubits}, dt={delta_t}, n_steps={n_steps_range[0]}-{n_steps_max}")

    if not args.force and data_handler.check_data_exists(dataset_path):
        print("Data already exists. Use --force to regenerate or --analyze to analyze.")
        return # Exit if data exists and not forcing

    # --- Dynamically Import and Instantiate Simulation Class ---
    try:
        module_path, class_name = sim_config["class_path"].rsplit('.', 1)
        SimModule = importlib.import_module(module_path)
        SimClass = getattr(SimModule, class_name)
    except Exception as e:
        print(f"Error importing simulation class {sim_config['class_path']}: {e}")
        sys.exit(1)

    # Prepare metadata BEFORE instantiation (to pass correct params)
    metadata = {
        "simulation_name": args.simulation,
        "n_qubits": n_qubits,
        "delta_t": delta_t,
        "n_steps_range": n_steps_range,
        "measurement_operator": sim_config["measurement_operator"],
        "initial_state_type": cfg.INITIAL_STATE_TYPE, # Add other relevant global params
        "simulation_mode": cfg.SIMULATION_MODE,
        # Add parameter ranges used (J, B, etc.)
        **{k: v for k, v in sim_config["default_ranges"].items() if k not in ['n_qubits', 'n_steps']},
        **sim_config["extra_args"],
    }

    current_param_ranges = {
        key: sim_config["default_ranges"][key]
        for key in sim_config["params"] if key not in ['n_qubits', 'n_steps'] and key in sim_config["default_ranges"]
    }
    # Update with J_RANGE and B_RANGE specifically if defined globally/overridden
    if 'J' in sim_config["params"]:
        current_param_ranges['J'] = sim_config["default_ranges"].get("J", []) # Or read override
    if 'B' in sim_config["params"]:
        current_param_ranges['B'] = sim_config["default_ranges"].get("B", []) # Or read override


    # Instantiate the simulation, passing required parameters
    try:
        simulation = SimClass(
            n_qubits=n_qubits,
            measurement_operator=sim_config["measurement_operator"],
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_type=cfg.INITIAL_STATE_TYPE, # Read from global config
            simulation_mode=cfg.SIMULATION_MODE,       # Read from global config
            param_ranges=current_param_ranges       # Pass the dict of ranges
            # Add **sim_config["extra_args"] if needed
        )

        # Optional: Verify simulation object has the intended parameters
        # (This check is less critical now as parameters are passed directly)
        # if simulation.n_qubits != n_qubits or simulation.delta_t != delta_t or simulation.n_steps_range != n_steps_range:
        #      print("Warning: Simulation object parameters mismatch after instantiation.")

    except Exception as e:
        print(f"Error instantiating simulation class {class_name}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging instantiation
        sys.exit(1)

    print("Starting quantum simulation runs...")
    start_time = time.time()
    dataset = simulation.generate_dataset() # Generate list of dicts
    end_time = time.time()
    print(f"Simulation runs completed in {end_time - start_time:.2f} seconds.")

    if not dataset:
        print("Error: No data generated by the simulation.")
        return

    # Get expected feature names from the instance
    expected_feature_names = simulation.get_ml_input_feature_names()

    # Save data and metadata
    print("Saving data and metadata...")
    data_handler.save_simulation_data_with_metadata(
        dataset,
        metadata,
        dataset_path,
        expected_feature_names
    )

    print("Data generation complete.")

if __name__ == "__main__":
    main()