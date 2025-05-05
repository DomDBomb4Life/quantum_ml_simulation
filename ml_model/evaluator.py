# quantum_ml_simulation/ml_model/evaluator.py
# Evaluates the ML model (Vector Output) and generates key analysis plots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress
import os
import time
import warnings
from typing import Optional, Dict, List

# Use relative import
from ..config import simulation_params as cfg
# Assuming ModelTrainer class definition exists
# from .trainer import ModelTrainer # Not strictly needed here, just need model

class ModelEvaluator:
    """Evaluates the ML model (vector output) and creates visualizations."""

    def __init__(self, model_trainer):
        """Initializes the ModelEvaluator."""
        if model_trainer is None or model_trainer.model is None:
            raise ValueError("ModelEvaluator requires a valid ModelTrainer instance.")
        self.model = model_trainer.model
        # Store output dim for convenience (optional)
        self.output_dim = model_trainer.output_dim

    # --- MODIFIED: Handles 2D y_test, calculates L2 error ---
    def evaluate(self, X_test_df: pd.DataFrame, y_test_np: np.ndarray) -> Dict:
        """
        Evaluates the model on the test set for vector outputs.

        Args:
            X_test_df: DataFrame of test features (n_samples, n_features).
            y_test_np: NumPy array of actual target vectors (n_samples, n_outputs).

        Returns:
            A dictionary containing:
            - 'mse': Mean Squared Error (averaged over outputs).
            - 'mae': Mean Absolute Error (averaged over outputs).
            - 'rmse': Root Mean Squared Error (averaged).
            - 'r2_score': R-squared score (variance weighted or uniform average).
            - 'y_pred_np': NumPy array of predicted vectors (n_samples, n_outputs).
            - 'y_test_np': NumPy array of actual vectors (passed in).
            - 'df_results': DataFrame with X_test features and 'l2_norm_error'.
        """
        if self.model is None: raise RuntimeError("Model not available.")
        if X_test_df is None or y_test_np is None: raise ValueError("Test data cannot be None.")
        if X_test_df.empty or len(y_test_np) == 0: raise ValueError("Test data cannot be empty.")
        if len(X_test_df) != len(y_test_np): raise ValueError("X_test and y_test must have the same number of samples.")
        if y_test_np.shape[1] != self.output_dim: raise ValueError(f"y_test has wrong output dimension ({y_test_np.shape[1]} != {self.output_dim})")

        print("\n--- Evaluating Model on Test Set (Vector Output) ---")
        start_eval_time = time.time()

        X_test_np_features = X_test_df.values
        # Predict vector outputs
        y_pred_np = self.model.predict(X_test_np_features) # Shape (n_samples, n_outputs)

        # Calculate standard multi-output metrics (averaged by default)
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_np, y_pred_np) # Default is 'variance_weighted' averaging

        # --- Calculate L2 Norm Error for each sample ---
        error_vectors = y_pred_np - y_test_np # Shape (n_samples, n_outputs)
        l2_norm_error = np.linalg.norm(error_vectors, axis=1) # Norm along output dim -> Shape (n_samples,)

        print(f"Evaluation complete in {time.time() - start_eval_time:.2f}s")
        print(f"Test Set Performance (Averaged Metrics):")
        print(f"  Mean Squared Error (MSE):       {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"  Mean Absolute Error (MAE):      {mae:.6f}")
        print(f"  R-squared (R2 Score):           {r2:.6f}")
        # Report statistics on the L2 norm error
        print(f"L2 Norm Error Statistics:")
        print(f"  Mean:   {np.mean(l2_norm_error):.6f}")
        print(f"  Std Dev:{np.std(l2_norm_error):.6f}")
        print(f"  Min:    {np.min(l2_norm_error):.6f}")
        print(f"  Max:    {np.max(l2_norm_error):.6f}")


        # Create results DataFrame with features and the L2 error
        df_results = X_test_df.copy()
        df_results['l2_norm_error'] = l2_norm_error

        return {
            "mse": mse, "mae": mae, "rmse": rmse, "r2_score": r2,
            "y_pred_np": y_pred_np, # Return the full prediction array
            "y_test_np": y_test_np, # Return the full test array
            "df_results": df_results # DF contains features and L2 error per sample
        }

    # --- MODIFIED: Plots only one specified observable index ---
    def plot_predictions_vs_actual(self,
                                   y_test_np: np.ndarray,
                                   y_pred_np: np.ndarray,
                                   observable_index: int = 0,
                                   observable_name: str = "Observable 0",
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None):
        """
        Generates a scatter plot of predicted vs actual values for a *single*
        specified observable (output dimension).
        """
        print(f"Generating plot: Predictions vs Actual for {observable_name}")
        if y_test_np is None or y_pred_np is None:
             print("Error: Missing y_test or y_pred data.")
             return
        if y_test_np.ndim != 2 or y_pred_np.ndim != 2 or y_test_np.shape != y_pred_np.shape:
            print(f"Error: y_test and y_pred must be 2D arrays with the same shape.")
            print(f"Shapes: y_test={y_test_np.shape}, y_pred={y_pred_np.shape}")
            return
        if not (0 <= observable_index < y_test_np.shape[1]):
            print(f"Error: observable_index {observable_index} out of bounds for output dim {y_test_np.shape[1]}.")
            return

        y_test_single = y_test_np[:, observable_index]
        y_pred_single = y_pred_np[:, observable_index]

        # Calculate R2 score for this specific observable
        r2_single = r2_score(y_test_single, y_pred_single)
        plot_label = f'ML Predictions (R2={r2_single:.4f})'
        default_title = f"Predictions vs Actual: {observable_name} (Test Set)"

        plt.figure(figsize=(8, 8))
        min_val = min(y_test_single.min(), y_pred_single.min())
        max_val = max(y_test_single.max(), y_pred_single.max())
        padding = 0.05 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
        plot_min = min_val - padding
        plot_max = max_val + padding

        plt.scatter(y_test_single, y_pred_single, alpha=0.5, label=plot_label, s=20)
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Ideal (y=x)')
        plt.xlabel(f"Actual Output ({observable_name})", fontsize=12)
        plt.ylabel(f"Predicted Output ({observable_name})", fontsize=12)
        plt.title(title if title else default_title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  - Plot saved to: {os.path.basename(save_path)}")
                plt.close()
            except Exception as e: print(f"Error saving plot: {e}"); plt.show()
        else: plt.show()

    # --- MODIFIED: Plots error based on a column in df_results (e.g., 'l2_norm_error') ---
    def plot_error_vs_n_steps(self, df_results: pd.DataFrame, error_column: str = 'l2_norm_error', title: Optional[str] = None, save_path: Optional[str] = None):
        """
        Plots the specified error metric (mean +/- std dev) against 'n_steps'.
        Assumes df_results contains 'n_steps' and the 'error_column'.
        """
        feature_to_plot = 'n_steps' # X-axis
        default_title = f"Mean {error_column.replace('_',' ').title()} vs {feature_to_plot}"
        print(f"Generating plot: {title if title else default_title} (using {error_column})")


        if df_results is None or feature_to_plot not in df_results.columns or error_column not in df_results.columns:
            print(f"Error: Cannot plot. Missing columns '{feature_to_plot}' or '{error_column}'.")
            print(f"Available columns: {df_results.columns.tolist() if df_results is not None else 'None'}")
            return

        # Group by 'n_steps' and aggregate the specified error column
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            try:
                # Ensure error column is numeric before aggregation
                if not pd.api.types.is_numeric_dtype(df_results[error_column]):
                     print(f"Error: Error column '{error_column}' is not numeric.")
                     return
                # Drop NaNs in the specific error column before grouping
                plot_data = df_results[[feature_to_plot, error_column]].dropna()
                if plot_data.empty:
                     print("Warning: No valid data points after dropping NaNs for error plot.")
                     return
                error_stats = plot_data.groupby(feature_to_plot)[error_column].agg(['mean', 'std']).reset_index()
                error_stats = error_stats.assign(std=error_stats['std'].fillna(0))
            except Exception as e: print(f"Error calculating error statistics: {e}"); return

        if error_stats.empty: print(f"Warning: No error stats calculated for '{feature_to_plot}'."); return

        # --- Quantify Trend ---
        slope, intercept, r_value, p_value, std_err = linregress(error_stats[feature_to_plot], error_stats['mean'])
        trend_label = f'Trend (Slope={slope:.2e}, R2={r_value**2:.3f})' # Use R2 for label
        print(f"  - Linear Trend ({error_column} vs {feature_to_plot}): Slope={slope:.4e}, R^2={r_value**2:.4f}, p-value={p_value:.3f}")

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        n_values = error_stats[feature_to_plot]
        mean_error = error_stats['mean']
        std_error = error_stats['std']

        label_mean = f'Mean {error_column.replace("_"," ").title()}'
        plt.plot(n_values, mean_error, marker='o', linestyle='-', color='b', label=label_mean)
        plt.fill_between(n_values, mean_error - std_error, mean_error + std_error, color='b', alpha=0.2, label='Std Dev')
        plt.plot(n_values, intercept + slope * n_values, 'r--', linewidth=1.5, label=trend_label)

        y_label = f'Mean {error_column.replace("_"," ").title()}'
        plt.xlabel(f"Number of Time Steps ({feature_to_plot})", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title if title else default_title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)

        unique_n_steps = sorted(plot_data[feature_to_plot].unique()) # Use unique steps from actual data plotted
        if len(unique_n_steps) <= 20: plt.xticks(unique_n_steps)
        else: plt.xticks(np.linspace(min(unique_n_steps), max(unique_n_steps), num=10, dtype=int))

        plt.xlim(min(n_values)-0.5, max(n_values)+0.5)
        plt.ylim(bottom=0)
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  - Plot saved to: {os.path.basename(save_path)}")
                plt.close()
            except Exception as e: print(f"Error saving plot: {e}"); plt.show()
        else: plt.show()

    # plot_training_history method remains the same
    def plot_training_history(self, history, save_path: str | None = None):
        # ... (implementation is unchanged) ...
        if history is None or not hasattr(history, 'history') or not history.history: print("Warning: Invalid history object."); return
        history_dict = history.history; epochs = range(1, len(history_dict.get('loss', [])) + 1)
        metrics_to_plot = {}
        if 'loss' in history_dict and 'val_loss' in history_dict: metrics_to_plot['Loss'] = ('loss', 'val_loss')
        if 'mae' in history_dict and 'val_mae' in history_dict: metrics_to_plot['Mean Absolute Error (MAE)'] = ('mae', 'val_mae')
        if not metrics_to_plot: print("Warning: No suitable metrics found in history."); return
        num_plots = len(metrics_to_plot); plt.figure(figsize=(12, 5 * num_plots)); plot_index = 1
        for plot_title, (train_metric, val_metric) in metrics_to_plot.items():
            plt.subplot(num_plots, 1, plot_index); plt.plot(epochs, history_dict[train_metric], 'bo-', label=f'Training {plot_title}'); plt.plot(epochs, history_dict[val_metric], 'rs-', label=f'Validation {plot_title}'); plt.title(f'Training and Validation {plot_title}', fontsize=12, fontweight='bold'); plt.xlabel('Epochs', fontsize=10); plt.ylabel(plot_title, fontsize=10); plt.legend(fontsize=9); plt.grid(True, linestyle='--', alpha=0.6); plot_index += 1
        plt.tight_layout()
        if save_path:
             try: os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"  - Training history plot saved to: {os.path.basename(save_path)}"); plt.close()
             except Exception as e: print(f"Error saving training history plot: {e}"); plt.show()
        else: plt.show()

    # calculate_distance method is less relevant now, as vector error is primary
    # but can be kept to calculate average distance metric if desired
    def calculate_distance(self, y_test, y_pred, metric='mse') -> float:
        # ... (implementation is unchanged, calculates averaged metric) ...
        if metric.lower() == 'mse': distance = mean_squared_error(y_test, y_pred)
        elif metric.lower() == 'mae': distance = mean_absolute_error(y_test, y_pred)
        elif metric.lower() == 'rmse': distance = np.sqrt(mean_squared_error(y_test, y_pred))
        else: print(f"Warning: Unknown distance metric '{metric}'. Defaulting to MSE."); distance = mean_squared_error(y_test, y_pred)
        print(f"Averaged Distance metric ({metric.upper()}): {distance:.6f}")
        return distance