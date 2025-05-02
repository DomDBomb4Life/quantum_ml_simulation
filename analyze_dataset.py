# quantum_ml_simulation/analyze_dataset.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For pairplots and heatmaps
from typing import List, Optional

# Use relative imports
from .config import simulation_params as cfg
from .data_management.data_handler import DataHandler

def plot_histograms(df: pd.DataFrame, columns: List[str], save_dir: str):
    """Plots histograms for specified columns."""
    print(f"  Generating histograms for: {columns}")
    num_cols = len(columns)
    num_rows = (num_cols + 2) // 3 # Aim for 3 plots per row
    plt.figure(figsize=(15, 4 * num_rows))
    for i, col in enumerate(columns):
        plt.subplot(num_rows, 3, i + 1)
        try:
            # Drop NaNs for histogram plotting only for the current column
            data_to_plot = df[col].dropna()
            if data_to_plot.empty:
                 print(f"    - Skipping histogram for {col} (all NaN or empty)")
                 plt.title(f"{col}\n(No Valid Data)", fontsize=10)
                 plt.xlabel("Value")
                 plt.ylabel("Frequency")
            else:
                 plt.hist(data_to_plot, bins=30, edgecolor='k', alpha=0.7)
                 plt.title(f"Distribution of {col}", fontsize=10)
                 plt.xlabel("Value", fontsize=9)
                 plt.ylabel("Frequency", fontsize=9)
                 plt.xticks(fontsize=8)
                 plt.yticks(fontsize=8)
                 # Add mean/std annotation
                 mean_val = data_to_plot.mean()
                 std_val = data_to_plot.std()
                 plt.text(0.6, 0.9, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
                          ha='center', va='center', transform=plt.gca().transAxes, fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        except KeyError:
            print(f"    - Column '{col}' not found for histogram.")
        except Exception as e:
             print(f"    - Error plotting histogram for {col}: {e}")
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, "histograms.png")
    try:
        plt.savefig(save_path, dpi=150)
        print(f"  - Histograms saved to: {os.path.basename(save_path)}")
        plt.close()
    except Exception as e:
        print(f"    - Error saving histograms: {e}")
        plt.show()

def plot_pairplot(df: pd.DataFrame, columns: List[str], save_dir: str, sample_frac: float = 0.2):
    """Plots a pairplot (scatter matrix) for specified columns."""
    if not columns or df[columns].isnull().all().all():
         print("  Skipping pairplot (no valid columns or all data is NaN).")
         return
    # Reduce the number of columns for pairplot if too many, to avoid crashes/unreadable plots
    max_pairplot_cols = 8 # Keep this manageable
    if len(columns) > max_pairplot_cols:
        print(f"  Warning: Pairplot requested for {len(columns)} columns. Limiting to first {max_pairplot_cols}.")
        columns = columns[:max_pairplot_cols]
        # Ensure all columns actually exist
        columns = [c for c in columns if c in df.columns]
        if not columns or df[columns].isnull().all().all():
             print("  Skipping pairplot (no valid columns after filtering).")
             return


    print(f"  Generating pairplot for: {columns} (using {sample_frac*100:.0f}% sample)")

    # Sample data if large to speed up plotting
    plot_df_subset = df[columns] # Select columns first
    if len(plot_df_subset) > 5000:
        plot_df = plot_df_subset.dropna().sample(frac=sample_frac, random_state=42)
        if plot_df.empty:
             print("  Skipping pairplot (no valid data after sampling/dropping NaNs).")
             return
    else:
         plot_df = plot_df_subset.dropna()
         if plot_df.empty:
              print("  Skipping pairplot (no valid data after dropping NaNs).")
              return

    try:
        # Use seaborn for pairplot
        g = sns.pairplot(plot_df, diag_kind='hist', plot_kws={'alpha':0.4, 's':10}) # Use smaller points
        g.fig.suptitle("Pairwise Relationships and Distributions", y=1.02, fontsize=14)
        save_path = os.path.join(save_dir, "pairplot.png")
        plt.savefig(save_path, dpi=150)
        print(f"  - Pairplot saved to: {os.path.basename(save_path)}")
        plt.close()
    except Exception as e:
        print(f"    - Error generating/saving pairplot: {e}")


def plot_feature_vs_target(df: pd.DataFrame, features: List[str], target: str, save_dir: str):
    """Plots scatter plots of each feature vs the target variable."""
    print(f"  Generating feature vs target plots (Target: {target})")
    if target not in df.columns:
        print(f"    - Target column '{target}' not found. Skipping plots.")
        return

    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        print(f"    - No valid feature columns found for plotting against '{target}'. Skipping.")
        return

    num_cols = len(valid_features)
    num_rows = (num_cols + 2) // 3 # Aim for 3 plots per row
    plt.figure(figsize=(15, 4 * num_rows))

    for i, feat in enumerate(valid_features):
        plt.subplot(num_rows, 3, i + 1)
        try:
            # Drop NaNs only for the pair being plotted
            plot_data = df[[feat, target]].dropna()
            if plot_data.empty:
                print(f"    - Skipping {feat} vs {target} (no valid data points).")
                plt.title(f"{feat} vs {target}\n(No Valid Data)", fontsize=10)
            else:
                # Use a smaller sample if too many points
                if len(plot_data) > 10000:
                    plot_data = plot_data.sample(10000, random_state=42)
                    # print(f"      - Sampled {len(plot_data)} points for {feat} vs {target} plot.")

                plt.scatter(plot_data[feat], plot_data[target], alpha=0.3, s=10)
                plt.title(f"{target} vs {feat}", fontsize=10)
                plt.xlabel(feat, fontsize=9)
                plt.ylabel(target, fontsize=9)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.grid(True, linestyle='--', alpha=0.5)
        except Exception as e:
            print(f"    - Error plotting {feat} vs {target}: {e}")

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, f"features_vs_{target}.png")
    try:
        plt.savefig(save_path, dpi=150)
        print(f"  - Feature vs Target plots saved to: {os.path.basename(save_path)}")
        plt.close()
    except Exception as e:
        print(f"    - Error saving feature vs target plots: {e}")
        plt.show()

def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, fixed_params: dict, save_dir: str):
    """Plots a heatmap of z_col vs x_col and y_col, for fixed other parameters."""
    print(f"  Generating heatmap: {z_col} vs ({x_col}, {y_col}) for fixed {fixed_params}")
    if not all(c in df.columns for c in [x_col, y_col, z_col]):
        print(f"    - Skipping heatmap. Missing one of columns: {x_col}, {y_col}, {z_col}")
        return

    # Filter data based on fixed parameters
    query_parts = []
    for col, val in fixed_params.items():
        if col not in df.columns:
            print(f"    - Warning: Fixed parameter column '{col}' not found in data. Cannot filter.")
            continue
        # Use tolerance for floating point comparisons
        if isinstance(val, (float, np.floating)): # Use numpy type check too
             tolerance = abs(val) * 1e-6 + 1e-9 # Relative and absolute tolerance
             query_parts.append(f"`{col}` >= {val - tolerance} and `{col}` <= {val + tolerance}")
        elif isinstance(val, (int, np.integer)): # Exact match for integers
             query_parts.append(f"`{col}` == {int(val)}") # Ensure it's a Python int for query string
        else: # Treat other types (like strings if any) as exact match
             query_parts.append(f"`{col}` == '{val}'") # Add quotes for strings


    query_string = " and ".join(query_parts)
    try:
         if query_string:
              filtered_df = df.query(query_string)[[x_col, y_col, z_col]].dropna()
         else: # No fixed parameters specified or found
              filtered_df = df[[x_col, y_col, z_col]].dropna()

         if filtered_df.empty:
              print(f"    - Skipping heatmap. No data points match fixed parameters: {fixed_params}")
              return
         # Ensure x_col and y_col are treated as categorical/discrete for pivot
         # If they are continuous, this heatmap might not be suitable.
         # Check if there are too many unique values for a meaningful heatmap
         if filtered_df[x_col].nunique() > 50 or filtered_df[y_col].nunique() > 50:
              print(f"    - Skipping heatmap. Too many unique values in {x_col} or {y_col} for a grid ({filtered_df[x_col].nunique()}, {filtered_df[y_col].nunique()}).")
              return


         if filtered_df[[x_col, y_col]].duplicated().any():
              # If duplicate (x,y) pairs exist, take the mean of z
              # print(f"    - Warning: Duplicate ({x_col}, {y_col}) pairs found. Averaging '{z_col}'.")
              # Use ordered categories for consistent pivot table index/columns
              x_dtype = pd.CategoricalDtype(categories=sorted(filtered_df[x_col].unique()), ordered=True)
              y_dtype = pd.CategoricalDtype(categories=sorted(filtered_df[y_col].unique()), ordered=True)
              filtered_df[x_col] = filtered_df[x_col].astype(x_dtype)
              filtered_df[y_col] = filtered_df[y_col].astype(y_dtype)

              pivot_data = filtered_df.groupby([y_col, x_col])[z_col].mean().reset_index()
              pivot_table = pivot_data.pivot(index=y_col, columns=x_col, values=z_col)
         else:
              # Use ordered categories even for non-duplicates
              x_dtype = pd.CategoricalDtype(categories=sorted(filtered_df[x_col].unique()), ordered=True)
              y_dtype = pd.CategoricalDtype(categories=sorted(filtered_df[y_col].unique()), ordered=True)
              filtered_df[x_col] = filtered_df[x_col].astype(x_dtype)
              filtered_df[y_col] = filtered_df[y_col].astype(y_dtype)
              pivot_table = filtered_df.pivot(index=y_col, columns=x_col, values=z_col)


    except Exception as e:
         print(f"    - Error filtering or pivoting data for heatmap: {e}")
         # import traceback; traceback.print_exc() # Debugging
         return

    if pivot_table.empty:
        print(f"    - Skipping heatmap. Pivot table is empty after processing.")
        return

    plt.figure(figsize=(10, 8))
    # *** REMOVED cbar_label ***
    sns.heatmap(pivot_table, annot=False, fmt=".2f", cmap="viridis") # Fixed line

    fixed_params_str = ', '.join([f'{k}={v}' for k,v in fixed_params.items()])
    plt.title(f"{z_col} vs ({x_col}, {y_col})\n(Fixed: {fixed_params_str})", fontsize=12)
    plt.xlabel(x_col, fontsize=10)
    plt.ylabel(y_col, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout(pad=1.5)

    filename = f"heatmap_{z_col}_vs_{x_col}_{y_col}"
    for k,v in fixed_params.items():
        # Format fixed params nicely for filename
        if isinstance(v, (float, np.floating)):
             filename += f"_{k}{v:.3f}".replace('.', 'p') # Use 'p' for decimal point
        elif isinstance(v, (int, np.integer)):
             filename += f"_{k}{int(v)}"
        else:
             filename += f"_{k}{v}".replace(' ', '_') # Replace spaces if any

    filename+=".png"
    save_path = os.path.join(save_dir, filename)

    try:
        plt.savefig(save_path, dpi=150)
        print(f"  - Heatmap saved to: {os.path.basename(save_path)}")
        plt.close()
    except Exception as e:
        print(f"    - Error saving heatmap: {e}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze generated quantum simulation datasets.")
    parser.add_argument("--dataset_id", required=True,
                        help="Identifier of the dataset directory (e.g., IsingModel_N3_dt0.100_nmax10).")

    args = parser.parse_args()

    # Construct the full dataset path
    dataset_path = os.path.join(cfg.BASE_PROJECT_PATH, args.dataset_id)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"--- Analyzing Dataset: {args.dataset_id} ---")
    data_handler = DataHandler()

    # Create analysis output directory
    analysis_dir = os.path.join(dataset_path, "analysis_plots")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Saving analysis plots to: {os.path.relpath(analysis_dir)}")

    # 1. Load Data and Metadata
    print("\n[1] Loading Data...")
    # Load the full CSV here to get all generated columns (output, entanglement, validation_diff)
    try:
         sim_data_dir = data_handler.get_sim_data_path(dataset_path)
         # Try to get simulation name from metadata first
         metadata = None
         metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
         if os.path.exists(metadata_filepath):
              try:
                  with open(metadata_filepath, 'r') as f: metadata = json.load(f)
              except: pass # Ignore errors

         simulation_name = metadata.get("simulation_name", args.dataset_id.split('_')[0]) if metadata else args.dataset_id.split('_')[0]
         csv_filepath = os.path.join(sim_data_dir, f"{simulation_name}_dataset.csv")

         if os.path.exists(csv_filepath):
              full_df = pd.read_csv(csv_filepath)
              print(f"  Loaded full data CSV: {os.path.relpath(csv_filepath)}")
         else:
              print(f"Error: Full data CSV not found at {os.path.relpath(csv_filepath)}. Cannot perform analysis.")
              sys.exit(1)
    except Exception as e:
         print(f"Error loading full data CSV: {e}")
         sys.exit(1)

    # Get metadata again (or ensure it's loaded correctly above) for feature names
    if metadata is None:
         metadata_filepath = os.path.join(sim_data_dir, "metadata.json")
         if os.path.exists(metadata_filepath):
              try:
                  with open(metadata_filepath, 'r') as f: metadata = json.load(f)
                  print(f"  Loaded metadata: {os.path.relpath(metadata_filepath)}")
              except Exception as e:
                   print(f"  Warning: Error loading metadata during analysis: {e}. Proceeding without metadata details.")
                   metadata = {}
         else:
              print("  Warning: Metadata file not found during analysis. Proceeding without metadata details.")
              metadata = {}


    # --- Identify Columns ---
    # Get expected ML features from metadata or infer from CSV columns (excluding known metrics)
    known_metrics = ['output', 'entanglement', 'validation_diff']
    if metadata and 'expected_ml_features' in metadata:
         feature_cols = metadata['expected_ml_features']
         # Verify they exist in the DataFrame
         missing_features = [f for f in feature_cols if f not in full_df.columns]
         if missing_features:
              print(f"  Warning: Expected ML features {missing_features} not found in DataFrame. Inferring features.")
              feature_cols = [col for col in full_df.columns if col not in known_metrics]
         else:
              # Ensure order matches metadata
              full_df = full_df[feature_cols + [col for col in full_df.columns if col not in feature_cols]].copy() # Reorder
    else:
         # Infer features: all columns except known metrics
         feature_cols = [col for col in full_df.columns if col not in known_metrics]
         print(f"  Warning: Metadata missing 'expected_ml_features'. Inferring features: {feature_cols}")


    target_col = 'output'
    extra_cols = [col for col in known_metrics if col != target_col and col in full_df.columns]
    all_numeric_cols = [col for col in full_df.columns if pd.api.types.is_numeric_dtype(full_df[col])] # Use all numeric for histograms


    # --- Basic Data Cleaning (Optional, for plotting robustness) ---
    # Drop rows where the target is NaN (should already be handled by data_handler load, but double-check)
    if target_col in full_df.columns:
         initial_rows = len(full_df)
         full_df.dropna(subset=[target_col], inplace=True)
         if len(full_df) < initial_rows:
              print(f"  Dropped {initial_rows - len(full_df)} rows with NaN target values before analysis.")

    if full_df.empty:
         print("Error: DataFrame is empty after cleaning. Cannot perform analysis.")
         sys.exit(1)


    print(f"  Identified Feature Columns: {feature_cols}")
    print(f"  Identified Target Column: '{target_col}'")
    print(f"  Identified Extra Columns: {extra_cols}")


    # 2. Print Basic Statistics
    print("\n[2] Basic Statistics...")
    try:
         # Describe only numeric columns
         desc_df = full_df[all_numeric_cols].describe()
         # Add number of unique values for non-numeric columns if any? Not needed for this project
         print(desc_df.to_string())

         print("\n[Missing Values Check]")
         print(full_df.isnull().sum().to_string())

    except Exception as e:
         print(f"  Error generating descriptive statistics: {e}")

    # 3. Generate Visualizations
    print("\n[3] Generating Visualizations...")

    # Histograms of all numeric columns
    plot_histograms(full_df, all_numeric_cols, analysis_dir)

    # Pairplot of features + target + entanglement (if available) - select columns carefully
    cols_for_pairplot = feature_cols.copy()
    if target_col in full_df.columns: cols_for_pairplot.append(target_col)
    if 'entanglement' in extra_cols: cols_for_pairplot.append('entanglement')
    # Remove 'validation_diff' from pairplot as it's usually very small
    cols_for_pairplot = [c for c in cols_for_pairplot if c in full_df.columns] # Final check
    if len(cols_for_pairplot) > 1:
        plot_pairplot(full_df, cols_for_pairplot, analysis_dir)
    else:
        print("  Skipping pairplot (not enough columns).")


    # Scatter plots of features vs. target
    if target_col in full_df.columns:
        plot_feature_vs_target(full_df, feature_cols, target_col, analysis_dir)
    else:
        print(f"  Skipping Feature vs Target plots (target '{target_col}' not found).")


    # Scatter plots of features vs. entanglement (CRUCIAL)
    if 'entanglement' in extra_cols:
        plot_feature_vs_target(full_df, feature_cols, 'entanglement', analysis_dir)
    else:
         print("  Skipping Feature vs Entanglement plots (column not found).")


    # Scatter plots of features vs. validation_diff (Trotter error)
    if 'validation_diff' in extra_cols:
        plot_feature_vs_target(full_df, feature_cols, 'validation_diff', analysis_dir)
    else:
         print("  Skipping Feature vs Validation Diff plots (column not found).")


    # Heatmaps (Example: output vs two main parameters at fixed n_steps)
    # Requires identifying the main varying parameters for the simulation
    sim_name = metadata.get("simulation_name", args.dataset_id.split('_')[0]) if metadata else args.dataset_id.split('_')[0]
    heatmap_param_map = {
        "IsingModel": ("J", "B"),
        "SpinChainPotential": ("n_steps", "k"), # k varies, n_steps varies
        "DimerizedHeisenberg": ("J1", "J2"), # J1 varies, J2 varies
    }
    # For SpinChainPotential & DimerizedHeisenberg, let's plot Output and Entanglement
    # as a function of J and B (Ising), or k and n_steps (Potential), or J1 and J2 (Dimerized)
    # Need to identify which columns are the varying parameters besides n_steps
    param_vars = [p for p in feature_cols if p != 'n_steps'] # Identify non-n_step features


    if sim_name in heatmap_param_map and len(param_vars) >= 1: # Need at least one other parameter
        # For now, let's plot output/entanglement vs n_steps and the *first* other param
        # More sophisticated logic could plot vs pairs of params (J,B), (k,n_steps), (J1,J2)
        # Let's aim for plotting output/entanglement vs (primary_param, n_steps)
        # Example: Ising (J, n_steps), Potential (k, n_steps), Dimerized (J1, n_steps)
        # This means we need to fix the *other* parameter(s)

        if sim_name == "IsingModel" and 'J' in feature_cols and 'B' in feature_cols and 'n_steps' in feature_cols:
             # Plot Output/Entanglement vs J and B, for a fixed n_steps
             n_steps_vals = sorted(full_df['n_steps'].unique())
             if len(n_steps_vals) > 1:
                  fixed_n = n_steps_vals[len(n_steps_vals) // 2] # Middle n
                  # Need to fix one of J or B to make a 2D heatmap (J vs B)
                  # Let's skip the J vs B heatmap for simplicity right now unless requested.
                  # The 'features_vs_target' plots already cover J vs output and B vs output (at all n)

             # Alternative heatmap idea: Output/Entanglement vs n_steps and one parameter (averaged over others)
             # This needs aggregation first, more complex.

             # Let's stick to the original heatmap idea: Output/Entanglement vs two variables, fixing others.
             # For Ising: Fix n_steps, plot output vs J and B.
             if 'J' in feature_cols and 'B' in feature_cols and 'n_steps' in feature_cols:
                 n_steps_vals = sorted(full_df['n_steps'].unique())
                 if len(n_steps_vals) > 1:
                      fixed_n_steps_value = n_steps_vals[len(n_steps_vals) // 2]
                      print(f"  Attempting heatmaps vs J and B at fixed n_steps={fixed_n_steps_value}")
                      fixed_params = {'n_steps': fixed_n_steps_value}
                      if target_col in full_df.columns:
                           plot_heatmap(full_df, 'J', 'B', target_col, fixed_params=fixed_params, save_dir=analysis_dir)
                      if 'entanglement' in extra_cols:
                           plot_heatmap(full_df, 'J', 'B', 'entanglement', fixed_params=fixed_params, save_dir=analysis_dir)

        elif sim_name == "SpinChainPotential" and 'n_steps' in feature_cols and 'k' in feature_cols:
             # Plot Output/Entanglement vs n_steps and k (no other params to fix)
             print(f"  Attempting heatmaps vs n_steps and k (no other params to fix)")
             fixed_params = {} # No other params to fix
             if target_col in full_df.columns:
                  plot_heatmap(full_df, 'n_steps', 'k', target_col, fixed_params=fixed_params, save_dir=analysis_dir)
             if 'entanglement' in extra_cols:
                  plot_heatmap(full_df, 'n_steps', 'k', 'entanglement', fixed_params=fixed_params, save_dir=analysis_dir)

        elif sim_name == "DimerizedHeisenberg" and 'J1' in feature_cols and 'J2' in feature_cols and 'n_steps' in feature_cols:
             # Plot Output/Entanglement vs J1 and J2, fixing n_steps
             n_steps_vals = sorted(full_df['n_steps'].unique())
             if len(n_steps_vals) > 1:
                  fixed_n_steps_value = n_steps_vals[len(n_steps_vals) // 2]
                  print(f"  Attempting heatmaps vs J1 and J2 at fixed n_steps={fixed_n_steps_value}")
                  fixed_params = {'n_steps': fixed_n_steps_value}
                  if target_col in full_df.columns:
                       plot_heatmap(full_df, 'J1', 'J2', target_col, fixed_params=fixed_params, save_dir=analysis_dir)
                  if 'entanglement' in extra_cols:
                       plot_heatmap(full_df, 'J1', 'J2', 'entanglement', fixed_params=fixed_params, save_dir=analysis_dir)
        else:
             print(f"  Skipping heatmaps for {sim_name} - not configured or missing necessary columns.")


    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    # Add a command line argument for setting the initial state type for analysis context?
    # Or just rely on the metadata.
    main()