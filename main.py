# quantum_ml_simulation/main.py
# Main script to orchestrate the quantum simulation and ML pipeline

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings

# Use relative imports for cleaner structure
from .simulations.ising_model import IsingModelSimulation
# from .simulations.test_simulation import TestSimulation # Can uncomment if using
# Import other simulations here when ready
# from .simulations.spin_chain_potential import SpinChainPotentialSimulation
# from .simulations.dimerized_heisenberg import DimerizedHeisenbergSimulation

from .ml_model.trainer import ModelTrainer
from .ml_model.evaluator import ModelEvaluator
from .data_management.data_handler import DataHandler
from .config import simulation_params as cfg

# --- Helper Function ---
def filter_valid_data(X_df: pd.DataFrame | None, y_series: pd.Series | None) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Filters out rows where the target 'y' is NaN."""
    if y_series is None or X_df is None:
        print("Warning: Input data for filtering is None.")
        return X_df, y_series

    initial_count = len(y_series)
    valid_indices = y_series.notna()

    y_filtered = y_series[valid_indices]
    X_filtered = X_df[valid_indices] # Apply the same index filter to features

    dropped_count = initial_count - len(y_filtered)
    if dropped_count > 0:
        print(f"Filtered out {dropped_count} rows with NaN target values.")

    if len(y_filtered) == 0:
        print("Warning: All data points filtered out.")

    return X_filtered, y_filtered

# --- Core Experiment Function ---
def run_experiment(simulation_class, simulation_name: str, force_generate_data: bool = False):
    """
    Runs the full workflow for a given simulation type.

    Args:
        simulation_class: The class of the simulation to run (e.g., IsingModelSimulation).
        simulation_name: A string name for the simulation (used for saving).
        force_generate_data: If True, always regenerate quantum data, ignoring existing files.
    """
    print(f"\n{'='*25} Starting Experiment: {simulation_name} {'='*25}")
    start_experiment_time = time.time()

    # 1. Initialization
    print("\n[1] Initializing Components...")
    try:
        simulation = simulation_class()
        data_handler = DataHandler()
        expected_feature_names = simulation.get_ml_input_feature_names() # These are the ONLY features the ML model should use
        print(f"  Simulation Type: {type(simulation).__name__}")
        print(f"  Expected ML Input Features: {expected_feature_names}") # Clarify these are INPUT features
    except Exception as e:
         print(f"Error during initialization: {e}")
         # import traceback; traceback.print_exc() # Uncomment for debugging
         return # Cannot proceed

    # 2. Data Generation or Loading
    print("\n[2] Loading/Generating Simulation Data...")
    X_df, y_series = None, None # We will get the processed X_df here
    loaded_column_names = None
    target_column = 'output'

    data_filepath_csv = os.path.join(cfg.DATA_PATH, f"{simulation_name}_dataset.csv")

    if not force_generate_data and os.path.exists(data_filepath_csv):
        print(f"Attempting to load existing data from: {data_filepath_csv}")
        # load_simulation_data returns ALL columns except target in X_df_loaded
        X_df_loaded, y_series, loaded_column_names = data_handler.load_simulation_data(simulation_name, target_column=target_column)

        if X_df_loaded is not None and y_series is not None:
            # Check if all EXPECTED features are PRESENT in the loaded columns
            missing_features = [f for f in expected_feature_names if f not in loaded_column_names]
            if missing_features:
                print(f"Error: Expected ML input features missing from loaded CSV: {missing_features}")
                print(f"  Available columns in CSV (excluding target): {loaded_column_names}")
                print("Cannot proceed with loaded data. Forcing regeneration.")
                force_generate_data = True # Force regeneration if expected features are missing
                X_df_loaded, y_series, loaded_column_names = None, None, None # Clear loaded data
            else:
                print(f"Successfully loaded {len(X_df_loaded)} data points.")
                # Explicitly select ONLY the expected features for ML input from the loaded DataFrame
                X_df = X_df_loaded[expected_feature_names].copy()
                # feature_names is already set to expected_feature_names
                print(f"  Selected features for ML: {expected_feature_names}")
                # Data is loaded, proceed to filtering and splitting

        else:
            print("Failed to load valid data. Will attempt generation.")
            force_generate_data = True

    if force_generate_data or X_df is None:
        if force_generate_data and X_df is not None:
             print("Forcing data regeneration as requested (despite previous successful load).")

        print("Generating new quantum simulation data...")
        start_gen_time = time.time()
        dataset = simulation.generate_dataset() # Returns list of dicts
        print(f"Data generation took {time.time() - start_gen_time:.2f}s")

        if not dataset:
            print("FATAL: No data was generated by the simulation. Exiting experiment.")
            return

        # Save the newly generated data
        # Pass expected_feature_names to ensure they are recorded correctly
        data_handler.save_simulation_data(dataset, simulation_name, expected_feature_names)

        # Reload from the saved file to ensure consistency and get DataFrame format
        print("Reloading generated data from CSV...")
        # This reload should now successfully load the CSV with the features we just saved
        X_df_loaded, y_series, loaded_column_names = data_handler.load_simulation_data(simulation_name, target_column=target_column)

        if X_df_loaded is None or y_series is None:
            print("FATAL: Failed to load the newly generated data from CSV. Check saving/loading logic.")
            return

        # Check if expected features are present after reloading the generated data
        missing_features = [f for f in expected_feature_names if f not in loaded_column_names]
        if missing_features:
             print(f"FATAL Error: Expected ML input features missing even after regeneration and reload: {missing_features}")
             print(f"  Columns found in regenerated CSV (excluding target): {loaded_column_names}")
             return
        X_df = X_df_loaded[expected_feature_names].copy() # Select only ML input features
        # feature_names is already set to expected_feature_names
        print(f"  Selected features for ML after regeneration: {expected_feature_names}")


    # At this point, X_df should be a DataFrame containing ONLY the expected_feature_names,
    # and y_series should be the corresponding Series.

    # Filter out any rows with NaN target values BEFORE splitting
    X_df, y_series = filter_valid_data(X_df, y_series)
    if X_df is None or X_df.empty:
         print("FATAL: No valid data points remain after filtering NaNs. Cannot proceed.")
         return

    # Set the feature_names variable based on the columns present in the final X_df
    feature_names = X_df.columns.tolist()
    input_dim = len(feature_names)

    # 3. Data Preparation for ML (Split into Train/Validation/Test)
    print("\n[3] Preparing Data for Machine Learning...")

    # --- FIX START ---
    # Define split sizes based on config
    test_split_size = cfg.ML_MODEL_PARAMS.get('test_split', 0.2)
    val_split_size = cfg.ML_MODEL_PARAMS.get('validation_split', 0.2)

    if (test_split_size + val_split_size) >= 1.0:
         raise ValueError("Sum of test_split and validation_split must be less than 1.0")
    if test_split_size < 0 or val_split_size < 0:
         raise ValueError("Split sizes must be non-negative.")

    # Convert to NumPy arrays for scikit-learn splitting
    X_np = X_df.values
    y_np = y_series.values
    # --- FIX END ---

    # Split 1: Separate Test Set
    X_train_val, X_test, y_train_val, y_test = None, None, None, None
    if test_split_size > 0 and len(y_np) > 1: # Need at least 2 samples to split into test
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_np, y_np, test_size=test_split_size, random_state=42, shuffle=True
            )
        except ValueError as e:
             print(f"Warning: Not enough samples for test split ({e}). Using all data for train/val.")
             X_train_val, y_train_val = X_np, y_np
             X_test, y_test = None, None # Explicitly set test to None
             test_split_size = 0.0 # Adjust size for next split calculation
    else:
         print("Skipping test split (test_split_size is 0 or not enough samples).")
         X_train_val, y_train_val = X_np, y_np
         X_test, y_test = None, None


    # Split 2: Separate Validation Set from remaining Train/Val data
    X_train, X_val, y_train, y_val = None, None, None, None
    if val_split_size > 0 and len(y_train_val) > 1: # Need at least 2 samples in train_val to split into validation
         # Calculate validation proportion relative to the train_val set size
         # Ensure denominator is not zero
         remaining_proportion = (1.0 - test_split_size)
         relative_val_split = val_split_size / remaining_proportion if remaining_proportion > 1e-9 else 0.5 # Use tiny number to avoid division by zero
         if relative_val_split >= 1.0: relative_val_split = 0.5 # Safety clamp


         try:
              X_train, X_val, y_train, y_val = train_test_split(
                   X_train_val, y_train_val, test_size=relative_val_split, random_state=42, shuffle=True
              )
         except ValueError as e:
              print(f"Warning: Not enough samples for validation split ({e}). Using all remaining data for training.")
              X_train, y_train = X_train_val, y_train_val
              X_val, y_val = None, None # Explicitly set validation to None
    else:
         print("Skipping validation split (val_split_size is 0 or not enough samples).")
         X_train, y_train = X_train_val, y_train_val
         X_val, y_val = None, None


    print("Data Split Summary:")
    print(f"  Training samples:   {len(y_train) if y_train is not None else 0}")
    print(f"  Validation samples: {len(y_val) if y_val is not None else 0}")
    print(f"  Test samples:       {len(y_test) if y_test is not None else 0}")

    # Check if we have training data
    if X_train is None or y_train is None or len(y_train) == 0:
         print("FATAL: Not enough training data available after splitting. Exiting.")
         return

    # Ensure validation data exists for training, if not, use training data (not ideal)
    if X_val is None or y_val is None or len(y_val) == 0:
         print("Warning: No validation data available. Using training data for validation during fit. This will not provide independent validation metrics.")
         X_val, y_val = X_train, y_train


    # 4. Initialize and Train ML Model
    print("\n[4] Training Machine Learning Model...")
    training_history = None
    try:
        # input_dim is correctly based on the columns in the final X_df
        trainer = ModelTrainer(input_dim=input_dim)
        training_history = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)
    except Exception as e:
         print(f"FATAL: Error during model training: {e}")
         import traceback
         traceback.print_exc()
         return # Cannot proceed without a trained model

    # 5. Evaluate Model
    print("\n[5] Evaluating Model Performance...")
    eval_results = None
    # Only evaluate on the TEST set if it exists and has samples
    if X_test is not None and y_test is not None and len(y_test) > 0:
        try:
            evaluator = ModelEvaluator(trainer)
            # Convert X_test back to DataFrame WITH CORRECT COLUMNS for evaluator
            X_test_df = pd.DataFrame(X_test, columns=feature_names) # Use the final selected feature names
            y_test_series = pd.Series(y_test, name=target_column)
            # evaluate now gets df with only ML input features
            eval_results = evaluator.evaluate(X_test_df, y_test_series)

            # Calculate specific distance metric (e.g., MSE) on test set
            _ = evaluator.calculate_distance(eval_results["y_test"], eval_results["y_pred"], metric='mse')

        except Exception as e:
            print(f"Error during model evaluation: {e}")
            # import traceback; traceback.print_exc() # Uncomment for debugging
            # Proceed without evaluation results if it fails
    else:
        print("Skipping evaluation on test set as it's not available or empty.")


    # 6. Perform Analysis and Visualization (using Test Set results if available)
    print("\n[6] Generating Analysis Plots...")
    results_dir = os.path.join(cfg.ML_RESULTS_PATH, simulation_name)
    plot_save_path_pvsa = os.path.join(results_dir, f"{simulation_name}_pred_vs_actual.png")
    plot_save_path_evsn = os.path.join(results_dir, f"{simulation_name}_error_vs_nsteps.png")

    if eval_results and 'df_results' in eval_results:
        try:
            # df_results in eval_results should have columns = expected_feature_names + y_actual + y_predicted + errors
            # The df_results already contains the correct features selected earlier
            evaluator.plot_predictions_vs_actual(eval_results["df_results"],
                                                title=f"{simulation_name}: Predictions vs Actual (Test Set)",
                                                save_path=plot_save_path_pvsa)

            # df_results correctly contains 'n_steps' because it was selected earlier
            evaluator.plot_error_vs_n_steps(eval_results["df_results"],
                                            title=f"{simulation_name}: Mean Abs Error vs n_steps (Test Set)",
                                            save_path=plot_save_path_evsn)
        except Exception as e:
             print(f"Error generating plots: {e}")
             # import traceback; traceback.print_exc() # Uncomment for debugging
    else:
        print("Skipping plot generation as evaluation results are not available (no test set).")


    # 7. Save All Results
    print("\n[7] Saving All Results...")
    if eval_results:
        # Save full results including metrics, predictions, and the df_results
        data_handler.save_ml_results(eval_results, training_history, simulation_name, trainer)
    else:
        # Save at least the model and history if evaluation failed/skipped
        print("Saving model and training history only (evaluation results missing).")
        minimal_results = {} # Empty dict for metrics part
        data_handler.save_ml_results(minimal_results, training_history, simulation_name, trainer)


    total_time = time.time() - start_experiment_time
    print(f"\n{'='*25} Experiment for {simulation_name} Finished ({total_time:.2f}s) {'='*25}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Suppress specific warnings if needed (e.g., TensorFlow future warnings)
    # warnings.filterwarnings("ignore", category=FutureWarning)
    # tf.get_logger().setLevel('ERROR') # Suppress TensorFlow INFO messages

    # Define which simulations to run
    # Add other simulations to this dictionary when they are implemented
    simulations_to_run = {
        "IsingModel": IsingModelSimulation,
        # "TestSim": TestSimulation, # Uncomment if TestSimulation is complete and configured
        # "SpinChainPotential": SpinChainPotentialSimulation,
        # "DimerizedHeisenberg": DimerizedHeisenbergSimulation,
    }

    # Control whether to force regeneration of quantum data
    # Set to True to always re-run quantum sims, False to load if possible
    FORCE_DATA_GENERATION = False

    # Run experiments sequentially
    for name, sim_class in simulations_to_run.items():
        run_experiment(sim_class, name, force_generate_data=FORCE_DATA_GENERATION)

    print("\nAll specified experiments have been run.")