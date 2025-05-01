# quantum_ml_simulation/ml_model/evaluator.py
# Evaluates the trained ML model and generates key analysis plots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Added r2_score
from scipy.stats import linregress # For trend analysis
import os
import time
import warnings

# Use relative import
from ..config import simulation_params as cfg

class ModelEvaluator:
    """Evaluates the ML model and creates visualizations."""

    def __init__(self, model_trainer):
        """
        Initializes the ModelEvaluator.

        Args:
            model_trainer: An instance of ModelTrainer containing the trained model.
        """
        if model_trainer is None or model_trainer.model is None:
            raise ValueError("ModelEvaluator requires a valid ModelTrainer instance with a trained model.")
        self.model = model_trainer.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model on the test set and calculates metrics.

        Args:
            X_test: DataFrame of test features.
            y_test: Series of actual target values for the test set.

        Returns:
            A dictionary containing evaluation results and metrics.
        """
        if self.model is None:
            raise RuntimeError("Model is not available for evaluation.")
        if X_test is None or y_test is None:
             raise ValueError("Test data (X_test, y_test) cannot be None.")
        if X_test.empty or y_test.empty:
             raise ValueError("Test data cannot be empty.")


        print("\n--- Evaluating Model on Test Set ---")
        start_eval_time = time.time()

        X_test_np = X_test.values
        y_test_np = y_test.values

        y_pred_np = self.model.predict(X_test_np).flatten()

        # Calculate standard metrics
        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_np, y_pred_np) # Calculate R-squared

        print(f"Evaluation complete in {time.time() - start_eval_time:.2f}s")
        print(f"Test Set Performance:")
        print(f"  Mean Squared Error (MSE):       {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"  Mean Absolute Error (MAE):      {mae:.6f}")
        print(f"  R-squared (R2 Score):           {r2:.6f}") # Display R-squared


        # Create a detailed DataFrame for analysis
        df_results = X_test.copy()
        df_results['y_actual'] = y_test_np
        df_results['y_predicted'] = y_pred_np
        df_results['absolute_error'] = np.abs(df_results['y_predicted'] - df_results['y_actual'])
        df_results['squared_error'] = (df_results['y_predicted'] - df_results['y_actual'])**2

        # Calculate relative error, handle division by zero
        # Use a small epsilon to avoid division by zero and large errors for small y_actual
        epsilon = 1e-8
        df_results['relative_error'] = df_results['absolute_error'] / (np.abs(df_results['y_actual']) + epsilon)


        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2, # Add R2 score
            "y_pred": y_pred_np,
            "y_test": y_test_np,
            "df_results": df_results # Include the detailed results DataFrame with relative error
        }

    def plot_predictions_vs_actual(self, df_results: pd.DataFrame, title: str = "Predictions vs Actual", save_path: str | None = None):
        """Generates a scatter plot of predicted vs actual values."""
        print(f"Generating plot: {title}")
        if df_results is None or 'y_actual' not in df_results or 'y_predicted' not in df_results:
             print("Error: Cannot plot predictions vs actual. Missing required columns.")
             return

        y_test = df_results['y_actual']
        y_pred = df_results['y_predicted']

        # Add R2 score to the plot
        r2 = r2_score(y_test, y_pred)
        plot_label = f'ML Predictions (R² = {r2:.4f})'

        plt.figure(figsize=(8, 8))
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        # Add padding
        padding = 0.05 * (max_val - min_val)
        plot_min = min_val - padding
        plot_max = max_val + padding

        plt.scatter(y_test, y_pred, alpha=0.5, label=plot_label, s=20)
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Ideal (y=x)')
        plt.xlabel("Actual Quantum Simulation Output (<O>)", fontsize=12)
        plt.ylabel("ML Model Predicted Output (<O>)", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
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
            except Exception as e:
                print(f"Error saving plot to {save_path}: {e}")
                plt.show()
        else:
            plt.show()


    def plot_error_vs_n_steps(self, df_results: pd.DataFrame, error_type: str = 'absolute_error', title: str = "Error vs Time Steps (n)", save_path: str | None = None):
        """
        Plots the specified error type (mean +/- std dev) against 'n_steps'.

        Args:
            df_results: DataFrame containing features and error columns.
            error_type: Column name of the error to plot ('absolute_error', 'relative_error').
            title: Plot title.
            save_path: Optional path to save the plot.
        """
        print(f"Generating plot: {title} (using {error_type})")
        feature_to_plot = 'n_steps'

        if df_results is None or feature_to_plot not in df_results.columns or error_type not in df_results.columns:
            print(f"Error: Cannot plot error vs {feature_to_plot}. Missing columns '{feature_to_plot}' or '{error_type}'.")
            print(f"Available columns: {df_results.columns.tolist() if df_results is not None else 'None'}")
            return

        # Calculate error statistics, suppressing potential FutureWarnings from pandas
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            try:
                error_stats = df_results.groupby(feature_to_plot)[error_type].agg(['mean', 'std']).reset_index()
                # Use assign to avoid SettingWithCopyWarning if fillna creates a copy
                error_stats = error_stats.assign(std=error_stats['std'].fillna(0))
            except Exception as e:
                 print(f"Error calculating error statistics for {error_type}: {e}")
                 return

        if error_stats.empty:
             print(f"Warning: No error statistics could be calculated for '{feature_to_plot}'.")
             return

        # --- Quantify Trend ---
        slope, intercept, r_value, p_value, std_err = linregress(error_stats[feature_to_plot], error_stats['mean'])
        trend_label = f'Trend (Slope={slope:.2e}, R²={r_value**2:.3f})'
        print(f"  - Linear Trend ({error_type} vs {feature_to_plot}): Slope={slope:.4e}, Intercept={intercept:.4e}, R^2={r_value**2:.4f}")

        plt.figure(figsize=(10, 6))
        n_values = error_stats[feature_to_plot]
        mean_error = error_stats['mean']
        std_error = error_stats['std']

        # Plot mean error line
        label_mean = f'Mean {error_type.replace("_", " ").title()}'
        plt.plot(n_values, mean_error, marker='o', linestyle='-', color='b', label=label_mean)

        # Add shaded region for standard deviation
        plt.fill_between(n_values, mean_error - std_error, mean_error + std_error,
                         color='b', alpha=0.2, label='Std Dev')

        # Plot trend line
        plt.plot(n_values, intercept + slope * n_values, 'r--', linewidth=1.5, label=trend_label)


        y_label = f'Mean {error_type.replace("_", " ").title()}'
        if error_type == 'absolute_error':
            y_label += ' |Predicted - Actual|'

        plt.xlabel(f"Number of Time Steps ({feature_to_plot})", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)

        unique_n_steps = sorted(df_results[feature_to_plot].unique())
        if len(unique_n_steps) <= 20: # Show all ticks if 20 or less
             plt.xticks(unique_n_steps)
        else: # Otherwise, use ~10 ticks
             plt.xticks(np.linspace(min(unique_n_steps), max(unique_n_steps), num=10, dtype=int))

        plt.xlim(min(n_values)-0.5, max(n_values)+0.5)
        plt.ylim(bottom=0) # Error generally non-negative
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  - Plot saved to: {os.path.basename(save_path)}")
                plt.close()
            except Exception as e:
                 print(f"Error saving plot to {save_path}: {e}")
                 plt.show()
        else:
            plt.show()


    def plot_training_history(self, history, save_path: str | None = None):
        """Plots the training and validation loss and MAE from Keras history."""
        if history is None or not hasattr(history, 'history') or not history.history:
             print("Warning: Invalid or empty Keras history object provided. Cannot plot training curves.")
             return

        history_dict = history.history
        epochs = range(1, len(history_dict.get('loss', [])) + 1)

        # Determine available metrics
        metrics_to_plot = {}
        if 'loss' in history_dict and 'val_loss' in history_dict:
            metrics_to_plot['Loss'] = ('loss', 'val_loss')
        if 'mae' in history_dict and 'val_mae' in history_dict:
             metrics_to_plot['Mean Absolute Error (MAE)'] = ('mae', 'val_mae')
        # Add other metrics like 'mse' if needed

        if not metrics_to_plot:
             print("Warning: No suitable metrics (loss/val_loss, mae/val_mae) found in history dict.")
             return

        num_plots = len(metrics_to_plot)
        plt.figure(figsize=(12, 5 * num_plots))
        plot_index = 1

        for plot_title, (train_metric, val_metric) in metrics_to_plot.items():
            plt.subplot(num_plots, 1, plot_index)
            plt.plot(epochs, history_dict[train_metric], 'bo-', label=f'Training {plot_title}')
            plt.plot(epochs, history_dict[val_metric], 'rs-', label=f'Validation {plot_title}')
            plt.title(f'Training and Validation {plot_title}', fontsize=12, fontweight='bold')
            plt.xlabel('Epochs', fontsize=10)
            plt.ylabel(plot_title, fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_index += 1

        plt.tight_layout()

        if save_path:
             try:
                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
                 print(f"  - Training history plot saved to: {os.path.basename(save_path)}")
                 plt.close()
             except Exception as e:
                  print(f"Error saving training history plot to {save_path}: {e}")
                  plt.show()
        else:
             plt.show()

    # calculate_distance method remains the same
    def calculate_distance(self, y_test, y_pred, metric='mse') -> float:
        """Calculates a distance metric between actual and predicted values."""
        # ... (implementation is unchanged) ...
        if metric.lower() == 'mse':
            distance = mean_squared_error(y_test, y_pred)
            metric_name = "Mean Squared Error (MSE)"
        elif metric.lower() == 'mae':
            distance = mean_absolute_error(y_test, y_pred)
            metric_name = "Mean Absolute Error (MAE)"
        elif metric.lower() == 'rmse':
             distance = np.sqrt(mean_squared_error(y_test, y_pred))
             metric_name = "Root Mean Squared Error (RMSE)"
        else:
            print(f"Warning: Unknown distance metric '{metric}'. Defaulting to MSE.")
            distance = mean_squared_error(y_test, y_pred)
            metric_name = "Mean Squared Error (MSE)"

        print(f"Distance metric ({metric_name}): {distance:.6f}")
        return distance