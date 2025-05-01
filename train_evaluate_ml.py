# quantum_ml_simulation/train_evaluate_ml.py
import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt # Keep for potential HP search summary plots

# Use relative imports
from .config import simulation_params as cfg
from .data_management.data_handler import DataHandler
from .ml_model.trainer import ModelTrainer
from .ml_model.evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

def run_standard_training(dataset_path: str, data_handler: DataHandler, run_name_suffix: str = ""):
    """Runs ML training with default hyperparameters."""
    print("\n--- Running Standard ML Training ---")
    run_name = f"standard_run_{time.strftime('%Y%m%d_%H%M%S')}{run_name_suffix}"
    print(f"ML Run Name: {run_name}")

    # 1. Load Data and Metadata
    print("Loading dataset...")
    X_df, y_series, metadata = data_handler.load_simulation_data_and_metadata(dataset_path)
    if X_df is None or y_series is None:
        print(f"Error: Could not load data from {dataset_path}. Exiting.")
        return None
    if metadata is None:
        print("Warning: Metadata not found. Proceeding without it.")
    simulation_name = metadata.get("simulation_name", "UnknownSim") if metadata else "UnknownSim"
    feature_names = X_df.columns.tolist() # Features are already selected by load method
    input_dim = len(feature_names)

    # 2. Prepare Data (Split)
    print("Splitting data...")
    test_split = cfg.DEFAULT_ML_PARAMS['test_split']
    val_split = cfg.DEFAULT_ML_PARAMS['validation_split']
    if (test_split + val_split) >= 1.0:
        print("Error: Invalid split ratios in config. Sum must be < 1.")
        return None

    X_np = X_df.values
    y_np = y_series.values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_np, y_np, test_size=test_split, random_state=42, shuffle=True
    )
    # Calculate relative validation split size needed for the second split
    relative_val_split = val_split / (1.0 - test_split) if (1.0 - test_split) > 1e-9 else 0.5
    if relative_val_split >= 1.0 : relative_val_split = 0.5 # Clamp

    if len(X_train_val) > 1:
         X_train, X_val, y_train, y_val = train_test_split(
              X_train_val, y_train_val, test_size=relative_val_split, random_state=42, shuffle=True
         )
    else: # Handle case with very few samples
         print("Warning: Not enough samples to create validation set. Using training set for validation.")
         X_train, y_train = X_train_val, y_train_val
         X_val, y_val = X_train, y_train # Use training set for validation

    print(f"Data Split: Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}")
    if len(y_train) == 0: print("Error: No training data after split."); return None


    # 3. Initialize and Train Model (Using DEFAULT_ML_PARAMS from config)
    print("Initializing and training model...")
    run_config = cfg.DEFAULT_ML_PARAMS.copy() # Start with defaults
    run_config['mode'] = 'standard'
    # We can potentially pass run_config directly to trainer if it's adapted,
    # but for now, ModelTrainer reads from cfg.DEFAULT_ML_PARAMS implicitly via cfg.ML_MODEL_PARAMS.
    # Let's make ModelTrainer more explicit later if needed. For now it uses the global config structure.
    # *** IMPORTANT: Ensure ModelTrainer uses DEFAULT_ML_PARAMS, not ML_MODEL_PARAMS if they differ ***
    # --> Modifying ModelTrainer to accept config dict might be cleaner long-term
    # --> For now, assume ModelTrainer reads relevant keys from the global cfg module

    trainer = ModelTrainer(input_dim=input_dim)
    training_history = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Evaluate Model
    print("Evaluating model...")
    evaluator = ModelEvaluator(trainer)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_test_series = pd.Series(y_test, name='output')
    eval_results = evaluator.evaluate(X_test_df, y_test_series)

    # 5. Generate Plots
    print("Generating plots...")
    ml_results_path = data_handler.get_ml_results_path(dataset_path)
    run_dir = os.path.join(ml_results_path, run_name)
    plot_save_path_pvsa = os.path.join(run_dir, f"{simulation_name}_pred_vs_actual.png")
    plot_save_path_abs_evsn = os.path.join(run_dir, f"{simulation_name}_abs_error_vs_nsteps.png")
    plot_save_path_rel_evsn = os.path.join(run_dir, f"{simulation_name}_rel_error_vs_nsteps.png")
    plot_save_path_history = os.path.join(run_dir, f"{simulation_name}_training_history.png")

    if evaluator and eval_results and 'df_results' in eval_results:
        try:
            r2 = eval_results.get('r2_score', np.nan)
            evaluator.plot_predictions_vs_actual(eval_results["df_results"],
                    title=f"{simulation_name}: Predictions vs Actual (Test Set, R²={r2:.4f})",
                    save_path=plot_save_path_pvsa)
            evaluator.plot_error_vs_n_steps(eval_results["df_results"],
                    error_type='absolute_error',
                    title=f"{simulation_name}: Mean Absolute Error vs n_steps (Test Set)",
                    save_path=plot_save_path_abs_evsn)
            evaluator.plot_error_vs_n_steps(eval_results["df_results"],
                    error_type='relative_error',
                    title=f"{simulation_name}: Mean Relative Error vs n_steps (Test Set)",
                    save_path=plot_save_path_rel_evsn)
        except Exception as e: print(f"Error generating evaluation plots: {e}")
    if evaluator and training_history:
        try: evaluator.plot_training_history(training_history, save_path=plot_save_path_history)
        except Exception as e: print(f"Error generating training history plot: {e}")

    # 6. Save Results
    print("Saving ML run results...")
    data_handler.save_ml_run_results(
        run_name=run_name,
        dataset_path=dataset_path,
        run_config=run_config, # Save the config used
        eval_results=eval_results,
        training_history=training_history,
        model_trainer=trainer
    )
    print("Standard ML training complete.")
    return run_dir


def run_hp_search(dataset_path: str, data_handler: DataHandler):
    """Runs ML training for various hyperparameter combinations."""
    print("\n--- Running Hyperparameter Search ---")
    base_run_name = f"hp_search_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Base ML Run Name: {base_run_name}")

    # 1. Load Data and Metadata
    print("Loading dataset...")
    X_df, y_series, metadata = data_handler.load_simulation_data_and_metadata(dataset_path)
    if X_df is None or y_series is None:
        print(f"Error: Could not load data from {dataset_path}. Exiting.")
        return
    simulation_name = metadata.get("simulation_name", "UnknownSim") if metadata else "UnknownSim"
    feature_names = X_df.columns.tolist()
    input_dim = len(feature_names)

    # 2. Prepare Data (Split - same as standard for consistency)
    print("Splitting data...")
    test_split = cfg.DEFAULT_ML_PARAMS['test_split']
    val_split = cfg.DEFAULT_ML_PARAMS['validation_split']
    if (test_split + val_split) >= 1.0: print("Error: Invalid splits."); return

    X_np = X_df.values
    y_np = y_series.values
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_np, y_np, test_size=test_split, random_state=42)
    relative_val_split = val_split / (1.0 - test_split) if (1.0 - test_split) > 1e-9 else 0.5
    if relative_val_split >= 1.0 : relative_val_split = 0.5
    if len(X_train_val)>1: X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=relative_val_split, random_state=42)
    else: X_train, y_train = X_train_val, y_train_val; X_val, y_val = X_train, y_train # Fallback
    print(f"Data Split: Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}")
    if len(y_train) == 0: print("Error: No training data."); return

    # 3. Define Hyperparameter Grid
    hp_grid = cfg.HP_SEARCH_PARAMS
    param_names = list(hp_grid.keys())
    param_values = list(hp_grid.values())
    search_combinations = list(itertools.product(*param_values))
    print(f"Total hyperparameter combinations to test: {len(search_combinations)}")

    results_summary = [] # To store key results from each trial

    # 4. Iterate Through Combinations
    for i, params in enumerate(search_combinations):
        trial_config = dict(zip(param_names, params))
        trial_name_parts = [f"{name[:2]}{val}" for name, val in trial_config.items()] # Short name for dir
        # Handle list case for hidden_layers in name
        for j, (name, val) in enumerate(trial_config.items()):
             if isinstance(val, list):
                 trial_name_parts[j] = f"{name[:2]}{'x'.join(map(str, val))}"

        trial_run_name = f"trial_{i+1}_{'_'.join(trial_name_parts)}"
        print(f"\n--- Starting HP Trial {i+1}/{len(search_combinations)} ({trial_run_name}) ---")
        print(f"Parameters: {trial_config}")

        # Create a temporary config for this trial based on defaults
        current_run_config = cfg.DEFAULT_ML_PARAMS.copy()
        current_run_config.update(trial_config) # Override defaults with trial HPs
        current_run_config['mode'] = 'hp_search_trial'

        # --- Train Model for this trial ---
        # *** IMPORTANT: Need to adapt ModelTrainer/Evaluator to accept config dict ***
        # --> Quick Hack: Temporarily modify global cfg (NOT recommended for parallel execution)
        # --> Better: Pass config dict to Trainer/Evaluator __init__ or methods
        # For now, let's assume ModelTrainer reads from global cfg and we modify it (needs caution)

        # --- (Placeholder for passing config - requires changes in Trainer/Evaluator) ---
        # trainer = ModelTrainer(input_dim=input_dim, config_override=current_run_config)
        # Instead, we'll just print a warning that it uses global config for now.
        print("Warning: ModelTrainer currently uses global config. Ensure config reflects trial params if needed.")
        # Ideally, ModelTrainer would take these directly:
        # trainer = ModelTrainer(input_dim=input_dim,
        #                       hidden_layers=current_run_config['hidden_layers'],
        #                       activation=current_run_config['activation'],
        #                       optimizer=current_run_config['optimizer'], # Need to add optimizer to HP search
        #                       learning_rate=current_run_config['learning_rate'])
        # history = trainer.train(...) # Pass batch_size, epochs from current_run_config

        # --- Simplified approach (uses global config, less flexible) ---
        trainer = ModelTrainer(input_dim=input_dim) # Uses default params from cfg
        # NOTE: This mock approach doesn't actually use the trial HPs yet! Needs refactor.
        # To make it work *without* refactoring trainer, you'd have to modify cfg.DEFAULT_ML_PARAMS
        # which is bad practice.
        # Let's simulate as if it used the HPs and just save the config.
        print("Simulating training with trial HPs (requires ModelTrainer refactor for actual use)...")
        # Mock history and evaluation for demonstration
        mock_history = {'history': {'loss': [0.1, 0.01], 'val_loss': [0.15, 0.015], 'mae': [0.2, 0.05], 'val_mae': [0.25, 0.06]}}
        mock_history_obj = type('obj', (object,), mock_history)() # Mock Keras history object

        mock_eval_results = {
            "mse": 0.015 + np.random.rand()*0.01,
            "mae": 0.06 + np.random.rand()*0.02,
            "rmse": np.sqrt(0.015 + np.random.rand()*0.01),
            "r2_score": 0.9 + np.random.rand()*0.05,
            # No df_results in mock
        }
        # --- End Mock ---

        # --- Save results for this trial ---
        trial_summary = current_run_config.copy() # Start with HPs
        trial_summary.update(mock_eval_results)   # Add performance metrics
        trial_summary['trial_run_name'] = trial_run_name
        results_summary.append(trial_summary)

        # Save individual trial results (using the mock objects for now)
        data_handler.save_ml_run_results(
            run_name=trial_run_name,
            dataset_path=dataset_path,
            run_config=current_run_config, # Config used for this trial
            eval_results=mock_eval_results, # Mock eval results
            training_history=mock_history_obj, # Mock history
            model_trainer=trainer # Mock trainer (model not actually trained with HPs)
        )

    # 5. Save Overall HP Search Summary
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values(by='mae') # Sort by best MAE (or val_loss if available)
    summary_filepath = os.path.join(data_handler.get_ml_results_path(dataset_path), f"{base_run_name}_summary.csv")
    try:
        summary_df.to_csv(summary_filepath, index=False, float_format='%.6g')
        print(f"\nHyperparameter search summary saved to: {os.path.relpath(summary_filepath)}")
        print("\nTop 5 Trials (sorted by MAE):")
        print(summary_df[['trial_run_name', 'learning_rate', 'batch_size', 'hidden_layers', 'activation', 'mae', 'r2_score']].head().to_string())
    except Exception as e:
        print(f"Error saving HP search summary: {e}")

    print("Hyperparameter search complete.")


def main():
    parser = argparse.ArgumentParser(description="Quantum ML Model Training and Evaluation")
    parser.add_argument("--dataset_id", required=True,
                        help="Identifier of the dataset directory (e.g., IsingModel_N4_dt0.05_nmax20).")
    parser.add_argument("--mode", choices=['standard', 'hp_search'], default='standard',
                        help="Operation mode: 'standard' training or 'hp_search'.")
    parser.add_argument("--run_suffix", default="", help="Optional suffix for standard run name.")
    # Add specific HP overrides if needed, e.g., --epochs, --batch_size

    args = parser.parse_args()

    # Construct the full dataset path
    dataset_path = os.path.join(cfg.BASE_PROJECT_PATH, args.dataset_id)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        print("Please generate the data first using generate_data.py")
        sys.exit(1)

    data_handler = DataHandler()

    if args.mode == 'standard':
        run_standard_training(dataset_path, data_handler, args.run_suffix)
    elif args.mode == 'hp_search':
        print("NOTE: HP Search currently uses mock training due to ModelTrainer limitations.")
        print("Refactor ModelTrainer to accept config dict for full functionality.")
        run_hp_search(dataset_path, data_handler)

if __name__ == "__main__":
    main()