# quantum_ml_simulation/data_management/data_handler.py
# Handles saving and loading of simulation data and ML results

import pandas as pd
import os
import json
import pickle
import numpy as np
import warnings

# Use relative import
from ..config import simulation_params as cfg

class DataHandler:
    """Handles saving and loading of simulation datasets and ML model artifacts."""

    def __init__(self, base_data_path: str = cfg.DATA_PATH, base_results_path: str = cfg.ML_RESULTS_PATH):
        """
        Initializes the DataHandler.

        Args:
            base_data_path: The root directory for saving simulation data.
            base_results_path: The root directory for saving ML results.
        """
        self.data_path = base_data_path
        self.results_path = base_results_path
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        print(f"DataHandler initialized.")
        print(f"  Simulation Data Path: {self.data_path}")
        print(f"  ML Results Path:      {self.results_path}")

    def _serialize_item(self, item):
        """Helper to make dictionary items JSON serializable (NumPy 2.0 compatible)."""
        serializable_item = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                # Convert array elements recursively if they are complex/objects
                try:
                    serializable_item[k] = v.tolist()
                except TypeError: # Handle potential issues with complex objects in arrays
                    serializable_item[k] = [self._serialize_item(elem) if isinstance(elem, dict) else elem for elem in v]

            # Use isinstance checks with base types (np.integer, np.floating)
            elif isinstance(v, np.integer): # Catches all NumPy integer types
                serializable_item[k] = int(v)
            elif isinstance(v, np.floating): # Catches all NumPy float types
                if np.isnan(v):
                    serializable_item[k] = None # Represent NaN as null in JSON
                else:
                    serializable_item[k] = float(v)
            elif isinstance(v, np.complexfloating): # Catches complex types
                 serializable_item[k] = {'real': float(v.real), 'imag': float(v.imag)}
            elif isinstance(v, np.bool_):
                 serializable_item[k] = bool(v)
            elif isinstance(v, (list, tuple)): # Handle nested structures
                 serializable_item[k] = [self._serialize_item(elem) if isinstance(elem, dict) else elem for elem in v]
            # Handle standard Python types that might be NaN (though less common)
            elif isinstance(v, float) and np.isnan(v):
                 serializable_item[k] = None
            else:
                serializable_item[k] = v # Assume other types are directly serializable
        return serializable_item


    def save_simulation_data(self, dataset: list[dict], simulation_name: str, feature_names: list[str]):
        """
        Saves the generated quantum simulation data as both CSV and JSON.

        Args:
            dataset: List of dictionaries [{'input': [...], 'output': ..., ...}, ...].
                     Each dictionary represents a simulation run.
            simulation_name: Name of the simulation (e.g., "IsingModel").
            feature_names: List of strings for the ML input feature columns.
        """
        if not dataset:
            print("Warning: No simulation data provided to save.")
            return

        # --- Prepare DataFrame (includes all relevant columns) ---
        df_data = []
        # Determine all possible keys from the dataset (handles optional keys)
        all_keys = set()
        for item in dataset:
            all_keys.update(item.keys())

        for item in dataset:
            row = {}
            # Add input features with their names
            if 'input' in item and isinstance(item['input'], (list, np.ndarray)) and len(item['input']) == len(feature_names):
                row.update({name: val for name, val in zip(feature_names, item['input'])})
            else:
                 print(f"Warning: Skipping item due to mismatch in 'input' features/names: {item.get('input', 'Missing')}")
                 continue # Skip row if input is bad

            # Add other keys found in the dataset (output, validation_diff, etc.)
            for key in all_keys:
                 if key != 'input': # Already handled
                     row[key] = item.get(key, np.nan) # Use NaN for missing optional keys

            df_data.append(row)

        if not df_data:
             print("Error: No valid data rows could be prepared for saving.")
             return

        df = pd.DataFrame(df_data)
        # Reorder columns: features first, then output, then others alphabetically
        output_col = 'output'
        other_cols = sorted([col for col in df.columns if col not in feature_names and col != output_col])
        column_order = feature_names + ([output_col] if output_col in df.columns else []) + other_cols
        # Ensure all columns exist before reordering
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]


        # --- Save as CSV ---
        csv_filepath = os.path.join(self.data_path, f"{simulation_name}_dataset.csv")
        try:
            df.to_csv(csv_filepath, index=False, float_format='%.8g') # More precision
            print(f"Simulation data (CSV) saved to: {csv_filepath}")
        except Exception as e:
            print(f"Error saving simulation data as CSV to {csv_filepath}: {e}")

        # --- Save as JSON (more structured, includes raw params if present) ---
        json_filepath = os.path.join(self.data_path, f"{simulation_name}_dataset.json")
        try:
            # Use the original dataset list for JSON to preserve potential raw structure
            serializable_dataset = [self._serialize_item(item) for item in dataset]
            with open(json_filepath, 'w') as f:
                json.dump(serializable_dataset, f, indent=2) # Use indent=2 for readability
            print(f"Simulation data (JSON) saved to: {json_filepath}")
        except Exception as e:
            print(f"Error saving simulation data as JSON to {json_filepath}: {e}")


    def load_simulation_data(self, simulation_name: str, target_column: str = 'output') -> tuple[pd.DataFrame | None, pd.Series | None, list | None]:
        """
        Loads simulation data from the CSV file.

        Args:
            simulation_name: Name of the simulation.
            target_column: The name of the column to be used as the ML target (y).

        Returns:
            Tuple: (X_df, y_series, feature_names)
                   X_df: DataFrame containing input features.
                   y_series: Series containing the target variable.
                   feature_names: List of input feature names.
                   Returns (None, None, None) if loading fails or data is invalid.
        """
        csv_filepath = os.path.join(self.data_path, f"{simulation_name}_dataset.csv")
        if not os.path.exists(csv_filepath):
            print(f"Data file not found: {csv_filepath}")
            return None, None, None

        print(f"Loading simulation data from: {csv_filepath}")
        try:
            df = pd.read_csv(csv_filepath)

            # --- Data Validation ---
            if target_column not in df.columns:
                print(f"Error: Target column '{target_column}' not found in {csv_filepath}. Available columns: {df.columns.tolist()}")
                return None, None, None

            # Check for NaN values in target
            initial_rows = len(df)
            df.dropna(subset=[target_column], inplace=True)
            if len(df) < initial_rows:
                 print(f"Warning: Dropped {initial_rows - len(df)} rows with NaN in target column '{target_column}'.")

            if df.empty:
                 print(f"Error: No valid data remaining after dropping NaNs in '{target_column}'.")
                 return None, None, None

            y_series = df[target_column]
            X_df = df.drop(columns=[target_column])

            # Identify feature columns (assuming they are all columns except the target for now)
            # A more robust approach might store feature names separately during saving.
            # For now, we infer from the columns remaining after dropping the target.
            # Let's refine this: infer based on simulation's known features if possible
            # This requires passing the simulation object or names, simpler to use remaining cols
            feature_names = X_df.columns.tolist()

            # Check for NaN/inf in feature columns
            with warnings.catch_warnings(): # Supress warnings during check
                warnings.simplefilter("ignore")
                if X_df.isnull().values.any() or np.isinf(X_df.values).any():
                    print("Warning: NaN or Inf values detected in feature columns (X). Consider imputation or review data generation.")
                    # Depending on strategy, could drop rows or impute here. For now, just warn.
                    # Example: X_df.fillna(X_df.mean(), inplace=True) # Impute with mean

            print(f"Successfully loaded {len(df)} data points. Features: {feature_names}")
            return X_df, y_series, feature_names

        except Exception as e:
            print(f"Error loading data from {csv_filepath}: {e}")
            return None, None, None


    def save_ml_results(self, results: dict, training_history, simulation_name: str, model_trainer):
        """
        Saves ML model, training history, evaluation metrics, and predictions.

        Args:
            results: Dictionary from ModelEvaluator.evaluate() containing metrics, y_pred, y_test.
            training_history: Keras History object from model.fit().
            simulation_name: Name of the simulation.
            model_trainer: The ModelTrainer instance containing the trained model.
        """
        sim_results_path = os.path.join(self.results_path, simulation_name)
        os.makedirs(sim_results_path, exist_ok=True)
        print(f"Saving ML results for {simulation_name} to: {sim_results_path}")

        # 1. Save Keras Model
        model_path = os.path.join(sim_results_path, f"{simulation_name}_model.keras")
        try:
            model_trainer.save_model(model_path) # Use trainer's method
        except Exception as e:
             print(f"Error saving Keras model: {e}")

        # 2. Save Evaluation Metrics (e.g., MSE, MAE)
        metrics_filepath = os.path.join(sim_results_path, f"{simulation_name}_eval_metrics.json")
        # Ensure metrics are JSON serializable (e.g., numpy floats to python floats)
        eval_metrics = {k: float(v) for k, v in results.items() if k not in ['y_pred', 'y_test', 'df_results']}
        try:
            with open(metrics_filepath, 'w') as f:
                json.dump(eval_metrics, f, indent=4)
            print(f"  - ML evaluation metrics saved to: {os.path.basename(metrics_filepath)}")
        except Exception as e:
            print(f"Error saving metrics JSON: {e}")

        # 3. Save Predictions vs Actual (including features if available)
        predictions_filepath = os.path.join(sim_results_path, f"{simulation_name}_predictions.csv")
        try:
            if 'df_results' in results and isinstance(results['df_results'], pd.DataFrame):
                 pred_df = results['df_results'] # Use df from evaluator if present
            elif 'y_test' in results and 'y_pred' in results:
                 pred_df = pd.DataFrame({'y_actual': results['y_test'].flatten(), 'y_predicted': results['y_pred'].flatten()})
            else:
                 print("Warning: Cannot save predictions CSV, missing 'y_test'/'y_pred' or 'df_results'.")
                 pred_df = None

            if pred_df is not None:
                 pred_df.to_csv(predictions_filepath, index=False, float_format='%.8g')
                 print(f"  - ML predictions/actuals saved to: {os.path.basename(predictions_filepath)}")
        except Exception as e:
            print(f"Error saving predictions CSV: {e}")

        # 4. Save Training History (Loss curves, etc.)
        history_filepath = os.path.join(sim_results_path, f"{simulation_name}_train_history.pkl")
        try:
            # Ensure history object is not None and has history attribute
            if training_history and hasattr(training_history, 'history'):
                # Make history JSON serializable before pickling (optional, but safer)
                serializable_history = {k: [float(val) for val in v] for k, v in training_history.history.items()}
                with open(history_filepath, 'wb') as f:
                    pickle.dump(serializable_history, f)
                print(f"  - Training history saved to: {os.path.basename(history_filepath)}")
            else:
                print("Warning: Training history object is invalid or missing. Skipping save.")
        except Exception as e:
            print(f"Error saving training history: {e}")

    def load_ml_model(self, simulation_name: str):
         """Loads a previously saved Keras model."""
         sim_results_path = os.path.join(self.results_path, simulation_name)
         model_path = os.path.join(sim_results_path, f"{simulation_name}_model.keras")
         # Requires a ModelTrainer instance to load into, or return the loaded model
         # For simplicity, let's assume we instantiate a new trainer and load into it
         # This needs tensorflow to be imported where called.
         try:
              from tensorflow.keras.models import load_model # type: ignore # Local import
              if os.path.exists(model_path):
                   model = load_model(model_path)
                   print(f"ML model loaded successfully from: {model_path}")
                   return model
              else:
                   print(f"ML model file not found: {model_path}")
                   return None
         except Exception as e:
              print(f"Error loading ML model from {model_path}: {e}")
              return None