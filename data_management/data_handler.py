# quantum_ml_simulation/data_management/data_handler.py
# Handles saving/loading of datasets with metadata and structured ML results
# Updated for Time-Series Vector Output (Wave 2)

import pandas as pd
import os
import json
import pickle
import numpy as np
import warnings
import time
from typing import Any, Dict, List, Tuple, Optional

# Use relative import
from ..config import simulation_params as cfg

class DataHandler:
    """Handles saving/loading of datasets with metadata and structured ML results."""

    def __init__(self):
        """Initializes the DataHandler."""
        self.base_project_path = cfg.BASE_PROJECT_PATH
        os.makedirs(self.base_project_path, exist_ok=True)
        print(f"DataHandler initialized. Base project path: {self.base_project_path}")

    # --- Simulation Data Handling ---

    def get_sim_data_path(self, dataset_path: str) -> str:
        """Gets the path to the simulation_data subdirectory within a dataset path."""
        return os.path.join(dataset_path, cfg.SIM_DATA_SUBDIR)

    # --- MODIFIED: Accepts output_observable_names ---
    def save_simulation_data_with_metadata(self,
                                           dataset: List[Dict],
                                           metadata: Dict,
                                           dataset_path: str,
                                           feature_names: List[str],
                                           output_observable_names: List[str]): # New argument
        """Saves simulation data (CSV, JSON) and its metadata (JSON)."""
        sim_data_dir = self.get_sim_data_path(dataset_path)
        os.makedirs(sim_data_dir, exist_ok=True)
        simulation_name = metadata.get("simulation_name", "UnknownSim")

        # 1. Save Metadata
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        try:
            metadata['generation_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata['expected_ml_features'] = feature_names # Store expected features
            metadata['output_observable_names'] = output_observable_names # Store output names
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"  - Metadata saved to: {os.path.relpath(metadata_filepath)}")
        except Exception as e:
            print(f"Error saving metadata to {metadata_filepath}: {e}")

        # 2. Save Dataset (CSV and JSON backup) - Pass output names to helper
        self._save_simulation_dataset_files(
            dataset,
            simulation_name,
            sim_data_dir,
            feature_names,
            output_observable_names # Pass to helper
        )

    # --- MODIFIED: Accepts and uses output_observable_names ---
    def _save_simulation_dataset_files(self, dataset, simulation_name, sim_data_dir,
                                       feature_names, output_observable_names):
        """Internal helper to save CSV and JSON dataset files, unpacking the output vector."""
        if not dataset:
            print("Warning: No simulation data provided to save.")
            return

        df_data = []
        # Determine all expected column names upfront
        expected_output_len = len(output_observable_names)
        # Identify other potential columns (like entanglement, validation_diff)
        other_keys = set()
        if dataset:
             other_keys = set(dataset[0].keys()) - {'input', 'output_vector'}

        for item in dataset:
            row = {}
            # 1. Add Input Features
            if 'input' in item and isinstance(item['input'], (list, np.ndarray)) and len(item['input']) == len(feature_names):
                row.update({name: val for name, val in zip(feature_names, item['input'])})
            else:
                print(f"Warning: Item skipped - mismatch in 'input' features/names: {item.get('input', 'Missing')}")
                continue

            # 2. Unpack Output Vector into named columns
            if 'output_vector' in item and isinstance(item['output_vector'], (list, np.ndarray)) and len(item['output_vector']) == expected_output_len:
                 row.update({name: val for name, val in zip(output_observable_names, item['output_vector'])})
            else:
                 print(f"Warning: Item skipped - mismatch or missing 'output_vector'. Expected len {expected_output_len}, got {item.get('output_vector', 'Missing')}")
                 # Fill with NaNs if vector is missing/wrong size? Or skip row? Let's fill.
                 row.update({name: np.nan for name in output_observable_names})
                 # Continue processing other keys even if output is bad for this row

            # 3. Add Other Keys (entanglement, validation_diff, etc.)
            for key in other_keys:
                row[key] = item.get(key, np.nan)

            df_data.append(row)

        if not df_data: print("Error: No valid data rows prepared."); return

        df = pd.DataFrame(df_data)

        # Define desired column order
        output_cols = output_observable_names
        other_cols_present = sorted([key for key in other_keys if key in df.columns])
        # Ensure feature_names, output_cols, and other_cols_present only contain columns actually in df
        final_feature_cols = [col for col in feature_names if col in df.columns]
        final_output_cols = [col for col in output_cols if col in df.columns]
        final_other_cols = [col for col in other_cols_present if col in df.columns]

        column_order = final_feature_cols + final_output_cols + final_other_cols
        # Make sure the order contains all columns, prevent loss of columns if one list was wrong
        present_cols = set(df.columns)
        final_column_order = [col for col in column_order if col in present_cols]
        missing_from_order = list(present_cols - set(final_column_order))
        if missing_from_order:
             print(f"Warning: Columns {missing_from_order} were present but not included in explicit order. Appending.")
             final_column_order.extend(sorted(missing_from_order))

        try:
            df = df[final_column_order]
        except KeyError as e:
             print(f"Error reordering columns: {e}. Saving with default order.")
             # Continue without reordering if error

        # --- Save CSV ---
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        try:
            df.to_csv(csv_filepath, index=False, float_format='%.8g') # Use sufficient precision
            print(f"  - Simulation data (CSV) saved to: {os.path.relpath(csv_filepath)}")
            print(f"    -> Columns: {df.columns.tolist()}") # Log columns saved
        except Exception as e: print(f"Error saving CSV: {e}")

        # --- Save JSON (still saves original structure with output_vector) ---
        json_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset_json.json") # Keep distinct name
        try:
            # Use the original dataset list to preserve output_vector structure
            serializable_dataset = [self._serialize_item(item) for item in dataset]
            with open(json_filepath, 'w') as f: json.dump(serializable_dataset, f, indent=2)
            print(f"  - Simulation data (JSON backup) saved to: {os.path.relpath(json_filepath)}")
        except Exception as e: print(f"Error saving JSON: {e}")


    # --- MODIFIED: Loads X_df and y_df (vector) based on metadata ---
    def load_simulation_data_and_metadata(self, dataset_path: str, load_target_vector: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame | pd.Series], Optional[Dict]]:
        """
        Loads simulation data CSV and metadata JSON.

        Args:
            dataset_path: Path to the specific dataset run folder.
            load_target_vector: If True, load the output observable columns into y_df.
                               If False, y_df will be None (useful for just analyzing inputs).

        Returns:
            Tuple: (X_df, y_df, metadata)
                   X_df: DataFrame containing input features.
                   y_df: DataFrame containing the target observable vector columns (if load_target_vector=True).
                   metadata: Dictionary containing the loaded metadata.
                   Returns (None, None, metadata) or (None, None, None) on failure.
        """
        sim_data_dir = self.get_sim_data_path(dataset_path)
        metadata = None
        X_df = None
        y_df = None # Now potentially a DataFrame

        # 1. Load Metadata (Crucial for identifying columns)
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        if os.path.exists(metadata_filepath):
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                print(f"  - Metadata loaded from: {os.path.relpath(metadata_filepath)}")
            except Exception as e:
                print(f"Error loading metadata from {metadata_filepath}: {e}")
                # Proceeding without metadata is risky for identifying columns
                print("Cannot reliably identify feature/output columns without metadata. Aborting load.")
                return None, None, None
        else:
            print(f"Error: Metadata file not found at {metadata_filepath}. Cannot load data.")
            return None, None, None

        # Check for essential keys in metadata
        feature_names = metadata.get('expected_ml_features')
        output_observable_names = metadata.get('output_observable_names')
        if not feature_names:
            print("Error: 'expected_ml_features' not found in metadata. Cannot load features.")
            return None, None, metadata
        if load_target_vector and not output_observable_names:
             print("Error: 'output_observable_names' not found in metadata. Cannot load target vector.")
             return None, None, metadata

        # 2. Load Data CSV
        simulation_name = metadata.get("simulation_name", "UnknownSim")
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        if os.path.exists(csv_filepath):
            print(f"Loading simulation data from: {os.path.relpath(csv_filepath)}")
            try:
                df = pd.read_csv(csv_filepath)

                # Verify all expected feature columns exist
                missing_features = [f for f in feature_names if f not in df.columns]
                if missing_features:
                    print(f"Error: Expected ML feature columns {missing_features} not found in CSV.")
                    return None, None, metadata

                # Verify all expected output columns exist (if loading target)
                missing_outputs = []
                if load_target_vector:
                     missing_outputs = [f for f in output_observable_names if f not in df.columns]
                     if missing_outputs:
                          print(f"Error: Expected output observable columns {missing_outputs} not found in CSV.")
                          return None, None, metadata

                # Select Feature DataFrame
                X_df = df[feature_names].copy()

                # Select Output DataFrame (if requested)
                if load_target_vector:
                    y_df = df[output_observable_names].copy()
                    # Drop rows where ANY output value is NaN
                    initial_rows = len(y_df)
                    rows_before_drop = len(X_df) # Should be same as y_df initially
                    valid_rows_mask = y_df.notna().all(axis=1)
                    y_df = y_df[valid_rows_mask]
                    X_df = X_df[valid_rows_mask] # Keep X and y aligned
                    dropped_count = initial_rows - len(y_df)
                    if dropped_count > 0:
                         print(f"Warning: Dropped {dropped_count} rows with NaN in one or more output columns.")
                    if y_df.empty:
                         print(f"Error: No valid data remaining after dropping NaNs in output columns.")
                         return None, None, metadata


                # Check for NaNs/Infs in features (optional, handled by some ML libs)
                if X_df.isnull().values.any() or np.isinf(X_df.values).any():
                    print("Warning: NaN or Inf values detected in feature columns (X). Consider imputation or review generation.")

                print(f"  - Loaded {len(X_df)} valid data points.")
                print(f"    -> Features: {feature_names}")
                if load_target_vector:
                     print(f"    -> Target Vector Columns: {output_observable_names}")

                return X_df, y_df, metadata

            except Exception as e:
                print(f"Error loading or processing data from {csv_filepath}: {e}")
                return None, None, metadata
        else:
            print(f"Data file not found: {csv_filepath}")
            return None, None, metadata


    def check_data_exists(self, dataset_path: str) -> bool:
        """Checks if the primary data file (CSV) exists for a dataset."""
        sim_data_dir = self.get_sim_data_path(dataset_path)
        metadata = None
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        if os.path.exists(metadata_filepath):
             try:
                 with open(metadata_filepath, 'r') as f: metadata = json.load(f)
             except: pass
        simulation_name = metadata.get("simulation_name", "UnknownSim") if metadata else "UnknownSim"
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        return os.path.exists(csv_filepath)

    # --- ML Results Handling ---
    # (Keep save_ml_run_results, load_ml_model_from_run, get_ml_results_path as is for now)
    # They operate on separate ML run folders and don't depend on the simulation data format directly.
    # We will adapt how we CALL save_ml_run_results in Wave 3.
    def get_ml_results_path(self, dataset_path: str) -> str:
         """Gets the path to the ml_results subdirectory within a dataset path."""
         return os.path.join(dataset_path, cfg.ML_RESULTS_SUBDIR)

    def save_ml_run_results(self, run_name: str, dataset_path: str, run_config: Dict,
                            eval_results: Dict, training_history, model_trainer):
         # ... (implementation remains the same as before) ...
         ml_results_dir = self.get_ml_results_path(dataset_path)
         run_dir = os.path.join(ml_results_dir, run_name)
         os.makedirs(run_dir, exist_ok=True)
         simulation_name = os.path.basename(dataset_path).split('_')[0] # Infer from path is safer here

         print(f"Saving ML run '{run_name}' results to: {os.path.relpath(run_dir)}")

         # 1. Save Run Config
         run_config_filepath = os.path.join(run_dir, "run_config.json")
         try:
             run_config['run_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
             with open(run_config_filepath, 'w') as f: json.dump(run_config, f, indent=4)
             print(f"  - Run config saved.")
         except Exception as e: print(f"Error saving run config: {e}")

         # 2. Save Model
         model_path = os.path.join(run_dir, f"{simulation_name}_model.keras")
         try: model_trainer.save_model(model_path)
         except Exception as e: print(f"Error saving Keras model: {e}")

         # 3. Save Eval Metrics
         metrics_filepath = os.path.join(run_dir, f"{simulation_name}_eval_metrics.json")
         # Use .item() for numpy types if needed
         serializable_metrics = {k: v.item() if isinstance(v, np.generic) else v
                                 for k, v in eval_results.items()
                                 if k not in ['y_pred', 'y_test', 'df_results']}
         try:
             with open(metrics_filepath, 'w') as f: json.dump(serializable_metrics, f, indent=4)
             print(f"  - Evaluation metrics saved.")
         except Exception as e: print(f"Error saving metrics: {e}")

         # 4. Save Predictions DataFrame
         if 'df_results' in eval_results and isinstance(eval_results['df_results'], pd.DataFrame):
             predictions_filepath = os.path.join(run_dir, f"{simulation_name}_predictions.csv")
             try:
                 eval_results['df_results'].to_csv(predictions_filepath, index=False, float_format='%.8g')
                 print(f"  - Predictions/Actuals DataFrame saved.")
             except Exception as e: print(f"Error saving predictions CSV: {e}")

         # 5. Save Training History
         if training_history and hasattr(training_history, 'history'):
             history_filepath = os.path.join(run_dir, f"{simulation_name}_train_history.pkl")
             try:
                 serializable_history = {k: [float(val) for val in v] for k, v in training_history.history.items()}
                 with open(history_filepath, 'wb') as f: pickle.dump(serializable_history, f)
                 print(f"  - Training history saved.")
             except Exception as e: print(f"Error saving training history: {e}")

    def load_ml_model_from_run(self, dataset_path: str, run_name: str):
         # ... (implementation remains the same as before) ...
         ml_results_dir = self.get_ml_results_path(dataset_path)
         run_dir = os.path.join(ml_results_dir, run_name)
         simulation_name = os.path.basename(dataset_path).split('_')[0]
         model_path = os.path.join(run_dir, f"{simulation_name}_model.keras")
         try:
             from tensorflow.keras.models import load_model # type: ignore
             if os.path.exists(model_path):
                 model = load_model(model_path)
                 print(f"ML model loaded successfully from run '{run_name}'")
                 return model
             else: print(f"ML model file not found for run '{run_name}'"); return None
         except Exception as e: print(f"Error loading ML model from run '{run_name}': {e}"); return None

    # --- Helper for JSON Serialization ---
    def _serialize_item(self, item):
        # ... (implementation remains the same as before) ...
        serializable_item = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                try: serializable_item[k] = v.tolist()
                except TypeError: serializable_item[k] = [self._serialize_item(elem) if isinstance(elem, dict) else elem for elem in v]
            elif isinstance(v, np.integer): serializable_item[k] = int(v)
            elif isinstance(v, np.floating): serializable_item[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, np.complexfloating): serializable_item[k] = {'real': float(v.real), 'imag': float(v.imag)}
            elif isinstance(v, np.bool_): serializable_item[k] = bool(v)
            elif isinstance(v, (list, tuple)): serializable_item[k] = [self._serialize_item(elem) if isinstance(elem, dict) else elem for elem in v]
            elif isinstance(v, float) and np.isnan(v): serializable_item[k] = None
            else: serializable_item[k] = v
        return serializable_item