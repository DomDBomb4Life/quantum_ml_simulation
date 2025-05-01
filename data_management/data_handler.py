# quantum_ml_simulation/data_management/data_handler.py
import pandas as pd
import os
import json
import pickle
import numpy as np
import warnings
import time
from typing import Any, Dict, List, Tuple, Optional

# Use relative import
from ..config import simulation_params as cfg # Renamed import

class DataHandler:
    """Handles saving/loading of datasets with metadata and structured ML results."""

    def __init__(self):
        """Initializes the DataHandler."""
        # Base project path is now central
        self.base_project_path = cfg.BASE_PROJECT_PATH
        os.makedirs(self.base_project_path, exist_ok=True)
        print(f"DataHandler initialized. Base project path: {self.base_project_path}")

    # --- Simulation Data Handling ---

    def get_sim_data_path(self, dataset_path: str) -> str:
        """Gets the path to the simulation_data subdirectory within a dataset path."""
        return os.path.join(dataset_path, cfg.SIM_DATA_SUBDIR)

    def save_simulation_data_with_metadata(self,
                                           dataset: List[Dict],
                                           metadata: Dict,
                                           dataset_path: str,
                                           feature_names: List[str]):
        """Saves simulation data (CSV, JSON) and its metadata (JSON)."""
        sim_data_dir = self.get_sim_data_path(dataset_path)
        os.makedirs(sim_data_dir, exist_ok=True)
        simulation_name = metadata.get("simulation_name", "UnknownSim")

        # 1. Save Metadata
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        try:
            # Add generation timestamp to metadata
            metadata['generation_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata['expected_ml_features'] = feature_names # Store expected features
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"  - Metadata saved to: {os.path.relpath(metadata_filepath)}")
        except Exception as e:
            print(f"Error saving metadata to {metadata_filepath}: {e}")
            # Continue saving data even if metadata fails, but warn user

        # 2. Save Dataset (CSV and JSON backup) - Reuse existing logic if available
        self._save_simulation_dataset_files(dataset, simulation_name, sim_data_dir, feature_names)


    def _save_simulation_dataset_files(self, dataset, simulation_name, sim_data_dir, feature_names):
        """Internal helper to save CSV and JSON dataset files."""
        if not dataset:
            print("Warning: No simulation data provided to save.")
            return

        # --- Prepare DataFrame ---
        # (Copy relevant logic from previous save_simulation_data method)
        df_data = []
        all_keys = set().union(*(d.keys() for d in dataset))
        for item in dataset:
             row = {}
             if 'input' in item and isinstance(item['input'], (list, np.ndarray)) and len(item['input']) == len(feature_names):
                 row.update({name: val for name, val in zip(feature_names, item['input'])})
             else:
                 print(f"Warning: Item skipped - mismatch in 'input' features/names: {item.get('input', 'Missing')}")
                 continue
             for key in all_keys:
                 if key != 'input':
                     row[key] = item.get(key, np.nan)
             df_data.append(row)

        if not df_data: print("Error: No valid data rows prepared."); return
        df = pd.DataFrame(df_data)
        output_col = 'output'
        other_cols = sorted([col for col in df.columns if col not in feature_names and col != output_col])
        column_order = feature_names + ([output_col] if output_col in df.columns else []) + other_cols
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]

        # --- Save CSV ---
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        try:
            df.to_csv(csv_filepath, index=False, float_format='%.8g')
            print(f"  - Simulation data (CSV) saved to: {os.path.relpath(csv_filepath)}")
        except Exception as e: print(f"Error saving CSV: {e}")

        # --- Save JSON ---
        json_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.json")
        try:
            serializable_dataset = [self._serialize_item(item) for item in dataset]
            with open(json_filepath, 'w') as f: json.dump(serializable_dataset, f, indent=2)
            print(f"  - Simulation data (JSON backup) saved to: {os.path.relpath(json_filepath)}")
        except Exception as e: print(f"Error saving JSON: {e}")


    def load_simulation_data_and_metadata(self, dataset_path: str, target_column: str = 'output') -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[Dict]]:
        """Loads simulation data CSV and metadata JSON."""
        sim_data_dir = self.get_sim_data_path(dataset_path)
        metadata = None
        X_df = None
        y_series = None

        # 1. Load Metadata
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        if os.path.exists(metadata_filepath):
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                print(f"  - Metadata loaded from: {os.path.relpath(metadata_filepath)}")
            except Exception as e:
                print(f"Error loading metadata from {metadata_filepath}: {e}")
                # Continue without metadata if loading failed
        else:
            print(f"Warning: Metadata file not found at {metadata_filepath}")

        # 2. Load Data CSV
        simulation_name = metadata.get("simulation_name", "UnknownSim") if metadata else "UnknownSim"
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        if os.path.exists(csv_filepath):
            print(f"Loading simulation data from: {os.path.relpath(csv_filepath)}")
            try:
                df = pd.read_csv(csv_filepath)
                if target_column not in df.columns:
                    print(f"Error: Target column '{target_column}' not found in {csv_filepath}.")
                    return None, None, metadata # Return metadata even if data load fails
                
                # Drop NaNs in target
                initial_rows = len(df)
                df.dropna(subset=[target_column], inplace=True)
                if len(df) < initial_rows: print(f"Warning: Dropped {initial_rows - len(df)} rows with NaN in target '{target_column}'.")
                
                if df.empty:
                     print(f"Error: No valid data after dropping NaNs in '{target_column}'.")
                     return None, None, metadata

                y_series = df[target_column]
                # Select only the features specified in metadata, if available
                if metadata and 'expected_ml_features' in metadata:
                    expected_features = metadata['expected_ml_features']
                    missing = [f for f in expected_features if f not in df.columns]
                    if missing:
                         print(f"Error: Expected features {missing} from metadata not found in CSV columns {df.columns.tolist()}.")
                         return None, None, metadata
                    X_df = df[expected_features].copy()
                    print(f"  - Loaded {len(X_df)} data points. Using features from metadata: {expected_features}")
                else:
                    # Fallback: Use all columns except target if metadata is missing feature info
                    X_df = df.drop(columns=[target_column])
                    print(f"Warning: Using all columns except target as features (metadata incomplete). Features: {X_df.columns.tolist()}")

                # Check for NaNs/Infs in features
                if X_df.isnull().values.any() or np.isinf(X_df.values).any():
                    print("Warning: NaN or Inf values detected in feature columns (X). Consider imputation.")

                return X_df, y_series, metadata

            except Exception as e:
                print(f"Error loading data from {csv_filepath}: {e}")
                return None, None, metadata # Return metadata even if data load fails
        else:
            print(f"Data file not found: {csv_filepath}")
            return None, None, metadata


    def check_data_exists(self, dataset_path: str) -> bool:
        """Checks if the primary data file (CSV) exists for a dataset."""
        sim_data_dir = self.get_sim_data_path(dataset_path)
        # We need simulation_name to check the CSV, try loading metadata first
        metadata = None
        metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
        if os.path.exists(metadata_filepath):
             try:
                 with open(metadata_filepath, 'r') as f: metadata = json.load(f)
             except: pass # Ignore errors here, just trying to get name
        simulation_name = metadata.get("simulation_name", "UnknownSim") if metadata else "UnknownSim"
        csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")
        return os.path.exists(csv_filepath)

    # --- ML Results Handling ---

    def get_ml_results_path(self, dataset_path: str) -> str:
         """Gets the path to the ml_results subdirectory within a dataset path."""
         return os.path.join(dataset_path, cfg.ML_RESULTS_SUBDIR)

    def save_ml_run_results(self,
                            run_name: str,
                            dataset_path: str,
                            run_config: Dict, # ML hyperparameters used for this run
                            eval_results: Dict, # Results from evaluator
                            training_history, # Keras history
                            model_trainer): # Trainer instance with model
        """Saves all artifacts for a specific ML run in a dedicated subfolder."""
        ml_results_dir = self.get_ml_results_path(dataset_path)
        run_dir = os.path.join(ml_results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        simulation_name = os.path.basename(dataset_path).split('_')[0] # Infer from dataset path

        print(f"Saving ML run '{run_name}' results to: {os.path.relpath(run_dir)}")

        # 1. Save Run Configuration (ML parameters)
        run_config_filepath = os.path.join(run_dir, "run_config.json")
        try:
            run_config['run_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(run_config_filepath, 'w') as f:
                json.dump(run_config, f, indent=4)
            print(f"  - Run config saved to: {os.path.basename(run_config_filepath)}")
        except Exception as e: print(f"Error saving run config: {e}")

        # 2. Save Model
        model_path = os.path.join(run_dir, f"{simulation_name}_model.keras")
        try: model_trainer.save_model(model_path)
        except Exception as e: print(f"Error saving Keras model: {e}")

        # 3. Save Evaluation Metrics
        metrics_filepath = os.path.join(run_dir, f"{simulation_name}_eval_metrics.json")
        serializable_metrics = {k: float(v) if isinstance(v, (np.number, float, int)) else v
                                for k, v in eval_results.items()
                                if k not in ['y_pred', 'y_test', 'df_results']} # Exclude large arrays/dfs
        try:
            with open(metrics_filepath, 'w') as f: json.dump(serializable_metrics, f, indent=4)
            print(f"  - Evaluation metrics saved to: {os.path.basename(metrics_filepath)}")
        except Exception as e: print(f"Error saving metrics: {e}")

        # 4. Save Predictions DataFrame (if exists)
        if 'df_results' in eval_results and isinstance(eval_results['df_results'], pd.DataFrame):
            predictions_filepath = os.path.join(run_dir, f"{simulation_name}_predictions.csv")
            try:
                eval_results['df_results'].to_csv(predictions_filepath, index=False, float_format='%.8g')
                print(f"  - Predictions/Actuals saved to: {os.path.basename(predictions_filepath)}")
            except Exception as e: print(f"Error saving predictions CSV: {e}")

        # 5. Save Training History
        if training_history and hasattr(training_history, 'history'):
            history_filepath = os.path.join(run_dir, f"{simulation_name}_train_history.pkl")
            try:
                serializable_history = {k: [float(val) for val in v] for k, v in training_history.history.items()}
                with open(history_filepath, 'wb') as f: pickle.dump(serializable_history, f)
                print(f"  - Training history saved to: {os.path.basename(history_filepath)}")
            except Exception as e: print(f"Error saving training history: {e}")


    def load_ml_model_from_run(self, dataset_path: str, run_name: str):
         """Loads a Keras model from a specific ML run."""
         ml_results_dir = self.get_ml_results_path(dataset_path)
         run_dir = os.path.join(ml_results_dir, run_name)
         simulation_name = os.path.basename(dataset_path).split('_')[0]
         model_path = os.path.join(run_dir, f"{simulation_name}_model.keras")

         try:
             from tensorflow.keras.models import load_model # type: ignore
             if os.path.exists(model_path):
                 model = load_model(model_path)
                 print(f"ML model loaded successfully from run '{run_name}': {os.path.relpath(model_path)}")
                 return model
             else:
                 print(f"ML model file not found for run '{run_name}': {model_path}")
                 return None
         except Exception as e:
             print(f"Error loading ML model from run '{run_name}': {e}")
             return None

    # --- Helper for JSON Serialization ---
    # (Use the _serialize_item method from previous version)
    def _serialize_item(self, item):
        """Helper to make dictionary items JSON serializable (NumPy 2.0 compatible)."""
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