# quantum_ml_simulation/ml_model/evaluator.py
# Evaluates the trained ML model and generates key analysis plots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import time

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
            A dictionary containing:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'y_pred': NumPy array of predicted values
            - 'y_test': NumPy array of actual test values
            - 'df_results': DataFrame combining features, actuals, predictions, and error
        """
        if self.model is None:
            raise RuntimeError("Model is not available for evaluation.")
        if X_test is None or y_test is None:
             raise ValueError("Test data (X_test, y_test) cannot be None.")

        print("\n--- Evaluating Model on Test Set ---")
        start_eval_time = time.time()

        # Ensure input is numpy array for prediction if needed by model
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

        y_pred_np = self.model.predict(X_test_np).flatten() # Flatten for metrics

        mse = mean_squared_error(y_test_np, y_pred_np)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        rmse = np.sqrt(mse)

        print(f"Evaluation complete in {time.time() - start_eval_time:.2f}s")
        print(f"Test Set Performance:")
        print(f"  Mean Squared Error (MSE):      {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"  Mean Absolute Error (MAE):     {mae:.6f}")

        # Create a DataFrame for easier analysis and plotting
        df_results = X_test.copy()
        df_results['y_actual'] = y_test_np
        df_results['y_predicted'] = y_pred_np
        df_results['absolute_error'] = np.abs(df_results['y_predicted'] - df_results['y_actual'])
        df_results['squared_error'] = (df_results['y_predicted'] - df_results['y_actual'])**2

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred_np,
            "y_test": y_test_np,
            "df_results": df_results # Include the detailed results DataFrame
        }

    def plot_predictions_vs_actual(self, df_results: pd.DataFrame, title: str = "Predictions vs Actual", save_path: str | None = None):
        """Generates a scatter plot of predicted vs actual values."""
        print(f"Generating plot: {title}")
        if df_results is None or 'y_actual' not in df_results or 'y_predicted' not in df_results:
             print("Error: Cannot plot predictions vs actual. Missing required columns in DataFrame.")
             return

        y_test = df_results['y_actual']
        y_pred = df_results['y_predicted']

        plt.figure(figsize=(8, 8))
        min_val = min(y_test.min(), y_pred.min()) - 0.1 * abs(min(y_test.min(), y_pred.min()))
        max_val = max(y_test.max(), y_pred.max()) + 0.1 * abs(max(y_test.max(), y_pred.max()))
        plt.scatter(y_test, y_pred, alpha=0.5, label='ML Predictions', s=20) # Smaller points
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
        plt.xlabel("Actual Quantum Simulation Output (<O>)", fontsize=12)
        plt.ylabel("ML Model Predicted Output (<O>)", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # Ensure aspect ratio is equal
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  - Plot saved to: {os.path.basename(save_path)}")
                plt.close() # Close the plot after saving
            except Exception as e:
                print(f"Error saving plot to {save_path}: {e}")
                plt.show() # Show if saving failed
        else:
            plt.show()

    def plot_error_vs_n_steps(self, df_results: pd.DataFrame, title: str = "Error vs Time Steps (n)", save_path: str | None = None):
        """
        Plots the Mean Absolute Error (MAE) with std dev against the number of time steps 'n'.
        This is crucial for evaluating the hypothesis.
        """
        print(f"Generating plot: {title}")
        feature_to_plot = 'n_steps' # The feature representing time steps

        if df_results is None or feature_to_plot not in df_results.columns or 'absolute_error' not in df_results.columns:
            print(f"Error: Cannot plot error vs {feature_to_plot}. Missing required columns in DataFrame.")
            print(f"Available columns: {df_results.columns.tolist() if df_results is not None else 'None'}")
            return

        # Use pandas for grouping and averaging
        try:
            # Group by 'n_steps' and calculate mean and std dev of the absolute error
            error_stats = df_results.groupby(feature_to_plot)['absolute_error'].agg(['mean', 'std']).reset_index()
            # Fill NaN std dev with 0 (happens if only one data point for an n_step)
            error_stats['std'].fillna(0, inplace=True)

        except Exception as e:
             print(f"Error calculating error statistics: {e}")
             return

        if error_stats.empty:
             print(f"Warning: No error statistics could be calculated for '{feature_to_plot}'.")
             return


        plt.figure(figsize=(10, 6))
        n_values = error_stats[feature_to_plot]
        mean_error = error_stats['mean']
        std_error = error_stats['std']

        # Plot mean error line
        plt.plot(n_values, mean_error, marker='o', linestyle='-', color='b', label='Mean Absolute Error (MAE)')

        # Add shaded region for standard deviation
        plt.fill_between(n_values, mean_error - std_error, mean_error + std_error,
                         color='b', alpha=0.2, label='Std Dev')

        # Alternative: Error bars (can look cluttered with many points)
        # plt.errorbar(n_values, mean_error, yerr=std_error,
        #              fmt='-o', color='b', ecolor='lightblue', elinewidth=1.5, capsize=5,
        #              label='Mean Absolute Error (with std dev)')

        plt.xlabel("Number of Time Steps (n)", fontsize=12)
        plt.ylabel("Mean Absolute Error |Predicted - Actual|", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        # Ensure ticks cover the range of n_steps
        unique_n_steps = sorted(df_results[feature_to_plot].unique())
        if len(unique_n_steps) < 15: # Show all ticks if not too many
             plt.xticks(unique_n_steps)
        else: # Otherwise, let matplotlib decide or set specific ticks
             plt.xticks(np.linspace(min(unique_n_steps), max(unique_n_steps), num=10, dtype=int))
        plt.xlim(min(n_values)-0.5, max(n_values)+0.5) # Add slight padding
        plt.ylim(bottom=0) # Error cannot be negative
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  - Plot saved to: {os.path.basename(save_path)}")
                plt.close() # Close the plot after saving
            except Exception as e:
                 print(f"Error saving plot to {save_path}: {e}")
                 plt.show() # Show if saving failed
        else:
            plt.show()

    def calculate_distance(self, y_test, y_pred, metric='mse') -> float:
        """
        Calculates a distance metric between actual and predicted values.

        Args:
            y_test: Array of actual values.
            y_pred: Array of predicted values.
            metric: The distance metric to use ('mse', 'mae', 'rmse').

        Returns:
            The calculated distance.
        """
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