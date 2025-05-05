# quantum_ml_simulation/train_evaluate_ml.py
# Handles ML training/evaluation for time-series vector output

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt

# Use relative imports
from .config import simulation_params as cfg
from .data_management.data_handler import DataHandler
from .ml_model.trainer import ModelTrainer
from .ml_model.evaluator import ModelEvaluator # Evaluator needs updates too
from sklearn.model_selection import train_test_split

def run_standard_training(dataset_path: str, data_handler: DataHandler, run_name_suffix: str = ""):
    """Runs ML training with default hyperparameters for vector output."""
    print("\n--- Running Standard ML Training (Vector Output Mode) ---")
    run_name = f"standard_run_{time.strftime('%Y%m%d_%H%M%S')}{run_name_suffix}"
    print(f"ML Run Name: {run_name}")

    # 1. Load Data and Metadata
    print("Loading dataset...")
    # --- MODIFIED: Expects y_df (DataFrame) ---
    X_df, y_df, metadata = data_handler.load_simulation_data_and_metadata(
        dataset_path, load_target_vector=True
    )
    if X_df is None or y_df is None:
        print(f"Error: Could not load data (X or Y vector) from {dataset_path}. Exiting.")
        return None
    if metadata is None:
        print("Warning: Metadata not found. Critical info like output names might be missing.")
        # Attempt to proceed, but it's risky
        metadata = {} # Use empty dict

    simulation_name = metadata.get("simulation_name", "UnknownSim")
    feature_names = metadata.get("expected_ml_features")
    output_observable_names = metadata.get("output_observable_names")

    # --- Determine input/output dimensions ---
    if feature_names is None: feature_names = X_df.columns.tolist(); print("Warning: Inferring feature names from loaded X_df.")
    if output_observable_names is None: output_observable_names = y_df.columns.tolist(); print("Warning: Inferring output names from loaded y_df.")

    input_dim = X_df.shape[1]
    output_dim = y_df.shape[1]
    print(f"Data Loaded: Input Dim={input_dim}, Output Dim={output_dim}")
    if input_dim != len(feature_names): print(f"Warning: X_df columns ({X_df.shape[1]}) != feature_names ({len(feature_names)})")
    if output_dim != len(output_observable_names): print(f"Warning: y_df columns ({y_df.shape[1]}) != output_names ({len(output_observable_names)})")


    # 2. Prepare Data (Split)
    print("Splitting data...")
    test_split = cfg.DEFAULT_ML_PARAMS['test_split']
    val_split = cfg.DEFAULT_ML_PARAMS['validation_split']
    if (test_split + val_split) >= 1.0:
        print("Error: Invalid split ratios in config. Sum must be < 1."); return None

    # Convert to NumPy arrays BEFORE splitting
    X_np = X_df.values
    y_np = y_df.values # y_np is now 2D: (num_samples, output_dim)

    # Split 1: Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_np, y_np, test_size=test_split, random_state=42, shuffle=True
    )
    # Split 2: Validation set
    relative_val_split = val_split / (1.0 - test_split) if (1.0 - test_split) > 1e-9 else 0.5
    if relative_val_split >= 1.0 : relative_val_split = 0.5
    if len(X_train_val) > 1:
         X_train, X_val, y_train, y_val = train_test_split(
              X_train_val, y_train_val, test_size=relative_val_split, random_state=42, shuffle=True
         )
    else:
         print("Warning: Not enough samples for validation set. Using training set for validation.")
         X_train, y_train = X_train_val, y_train_val
         X_val, y_val = X_train, y_train

    print(f"Data Split: Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}")
    print(f"  Shapes: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_val={X_val.shape}, y_val={y_val.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")
    if len(y_train) == 0: print("Error: No training data after split."); return None


    # 3. Initialize and Train Model
    print("Initializing and training model...")
    run_config = cfg.DEFAULT_ML_PARAMS.copy()
    run_config['mode'] = 'standard_vector'
    run_config['input_dim'] = input_dim # Add actual dims to config log
    run_config['output_dim'] = output_dim

    # --- MODIFIED: Pass explicit input_dim and output_dim ---
    try:
        trainer = ModelTrainer(input_dim=input_dim, output_dim=output_dim)
    except Exception as e:
         print(f"Error initializing ModelTrainer: {e}"); return None

    # Train using 2D y arrays
    training_history = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)


    # 4. Evaluate Model
    print("Evaluating model...")
    eval_results = None
    evaluator = None
    if X_test is not None and y_test is not None and len(y_test) > 0:
         try:
             evaluator = ModelEvaluator(trainer) # Evaluator now needs updating too!
             # Pass features as DataFrame, targets as DataFrame/Series or 2D numpy array
             # Need to reconstruct X_test_df to include feature names for evaluator plots
             X_test_df = pd.DataFrame(X_test, columns=feature_names)
             # Pass y_test as 2D numpy array
             eval_results = evaluator.evaluate(X_test_df, y_test) # Pass 2D y_test
         except Exception as e:
              print(f"Error during evaluation: {e}")
              import traceback; traceback.print_exc()
    else:
         print("Skipping evaluation on test set (not available or empty).")


    # 5. Generate Plots (Requires Evaluator Update for Vector Output)
    print("Generating plots...")
    ml_results_path = data_handler.get_ml_results_path(dataset_path)
    run_dir = os.path.join(ml_results_path, run_name)
    # Define output observable names from metadata if available, otherwise create generic names
    output_obs_names = metadata.get("output_observable_names", [f"Observable_{i}" for i in range(output_dim)])
    obs0_name = output_obs_names[0] if output_observable_names else "Observable_0"

    # Path for prediction vs actual plot (only for the first observable)
    plot_save_path_pvsa = os.path.join(run_dir, f"{simulation_name}_pred_vs_actual_{obs0_name}.png")
    # Path for L2 norm error plot
    plot_save_path_l2_evsn = os.path.join(run_dir, f"{simulation_name}_l2_error_vs_nsteps.png")
    # Path for training history plot
    plot_save_path_history = os.path.join(run_dir, f"{simulation_name}_training_history.png")

    if evaluator and eval_results and 'df_results' in eval_results:
        print("Generating evaluation plots...") # More specific message
        try:
            # Plot Pred vs Actual for first observable
            evaluator.plot_predictions_vs_actual(
                    eval_results["y_test_np"], # Pass numpy arrays
                    eval_results["y_pred_np"], # Pass numpy arrays
                    observable_index=0,
                    observable_name=obs0_name, # Use the actual name
                    title=f"{simulation_name}: Pred vs Actual ({obs0_name}, Test Set)", # More specific title
                    save_path=plot_save_path_pvsa)

            # Plot L2 Norm Error vs n_steps (most important plot)
            # --- MODIFIED LINE ---
            evaluator.plot_error_vs_n_steps(eval_results["df_results"],
                    error_column='l2_norm_error', # Changed 'error_type' to 'error_column'
                    title=f"{simulation_name}: Mean L2 Norm Error vs n_steps (Test Set)",
                    save_path=plot_save_path_l2_evsn)
            # --- END MODIFIED LINE ---

        except Exception as e: print(f"Error generating evaluation plots: {e}")

    if evaluator and training_history:
        try: evaluator.plot_training_history(training_history, save_path=plot_save_path_history)
        except Exception as e: print(f"Error generating training history plot: {e}")


    # 6. Save Results
    print("Saving ML run results...")
    # Need to ensure eval_results format is handled correctly by save_ml_run_results
    if eval_results is None: eval_results = {} # Ensure eval_results is a dict
    data_handler.save_ml_run_results(
        run_name=run_name,
        dataset_path=dataset_path,
        run_config=run_config,
        eval_results=eval_results,
        training_history=training_history,
        model_trainer=trainer
    )
    print("Standard ML training complete.")
    return run_dir


# --- run_hp_search function needs similar adaptation ---
# It should also determine output_dim, pass it to trainer,
# handle 2D y_np, and potentially adapt its result summary/comparison logic.
# Skipping full refactor of hp_search for brevity now, focus on standard run.
def run_hp_search(dataset_path: str, data_handler: DataHandler):
     print("\n--- Hyperparameter Search (Vector Output Mode) ---")
     print("WARNING: HP Search for vector output needs adaptation similar to standard run.")
     print("         Skipping HP search for now.")
     # TODO: Implement HP search for vector output if needed later.
     # Key steps: Load X_df, y_df; get input/output dims; loop through HPs;
     # instantiate trainer with trial HPs and correct output_dim; train;
     # evaluate using vector metrics; save results.
     pass


def main():
    parser = argparse.ArgumentParser(description="Quantum ML Model Training and Evaluation")
    parser.add_argument("--dataset_id", required=True,
                        help="Identifier of the dataset directory (e.g., IsingModel_N3_dt0.100_nmax10).")
    parser.add_argument("--mode", choices=['standard', 'hp_search'], default='standard',
                        help="Operation mode: 'standard' training or 'hp_search'.")
    parser.add_argument("--run_suffix", default="", help="Optional suffix for standard run name.")

    args = parser.parse_args()

    dataset_path = os.path.join(cfg.BASE_PROJECT_PATH, args.dataset_id)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}"); sys.exit(1)

    data_handler = DataHandler()

    if args.mode == 'standard':
        run_standard_training(dataset_path, data_handler, args.run_suffix)
    elif args.mode == 'hp_search':
        run_hp_search(dataset_path, data_handler) # Currently prints warning and exits

if __name__ == "__main__":
    main()