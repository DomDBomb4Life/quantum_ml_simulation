# quantum_ml_simulation/config/simulation_params.py
import numpy as np
import os

# --- Core Simulation Setup ---
# Keep these as defaults, can be overridden by generate_data.py args
N_QUBITS = 4 # Default N
DELTA_T = 0.05 # Default dt
N_STEPS_RANGE = list(range(1, 21)) # Default n steps (Tmax = 1.0)
J_RANGE = np.linspace(0.1, 2.0, 11).round(3).tolist() # Reduced density for faster testing
B_RANGE = np.linspace(0.0, 1.0, 11).round(3).tolist() # Reduced density for faster testing
INITIAL_STATE_TYPE = "superposition"
SIMULATION_MODE = "statevector"
N_SHOTS = 2048
VALIDATE_TROTTER = True
VALIDATION_MAX_QUBITS = 4
VALIDATION_TOLERANCE = 0.005
COMPUTE_ENTANGLEMENT = True
ENTANGLEMENT_PARTITION = N_QUBITS // 2

# --- Simulation-Specific Parameter Dictionaries ---
# These define the *structure* and default ranges if not overridden
# Measurement operator is key here
SIMULATION_CONFIGS = {
    "IsingModel": {
        "class_path": "quantum_ml_simulation.simulations.ising_model.IsingModelSimulation",
        "params": ["n_steps", "J", "B"], # Order matters for param tuples
        "default_ranges": { # Used if not overridden by CLI args
            "n_qubits": N_QUBITS,
            "n_steps": N_STEPS_RANGE,
            "J": J_RANGE,
            "B": B_RANGE
        },
        "measurement_operator": "Z1Z2",
        "extra_args": {} # Any other args needed by __init__ besides n_qubits, op
    },
    # Add entries for SpinChainPotential, DimerizedHeisenberg when implemented
    # "SpinChainPotential": { ... },
    # "DimerizedHeisenberg": { ... },
}

# --- Default ML Parameters ---
# Used by train_evaluate_ml.py in 'standard' mode
DEFAULT_ML_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mean_squared_error",
    "epochs": 100, # Reduced default epochs for faster runs
    "batch_size": 64, # Slightly larger batch size
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    # Data splitting ratios used in train_evaluate_ml.py
    "validation_split": 0.2,
    "test_split": 0.2
}

# --- ML Hyperparameter Search Space ---
# Used by train_evaluate_ml.py in 'hp_search' mode
HP_SEARCH_PARAMS = {
    'learning_rate': [0.01, 0.001, 0.0005],
    'batch_size': [32, 64, 128],
    'hidden_layers': [
        [64, 32],
        [128, 64],
        [128, 64, 32],
        # [256, 128, 64] # Can add more complex structures
    ],
    'activation': ['relu', 'tanh']
    # 'optimizer': ['adam', 'sgd'] # Can add optimizer search too
}

# --- Data Storage Configuration ---
# Base path for all project results
BASE_PROJECT_PATH = "./project_runs"
# Subdirectory for generated simulation data (will contain metadata)
SIM_DATA_SUBDIR = "simulation_data"
# Subdirectory for ML results (will contain run-specific folders)
ML_RESULTS_SUBDIR = "ml_results"

# --- Utility Function for Path Generation ---
def get_dataset_path(simulation_name: str, n_qubits: int, dt: float, n_steps_max: int) -> str:
    """Generates a unique directory path for a specific dataset configuration."""
    dataset_id = f"{simulation_name}_N{n_qubits}_dt{dt:.3f}_nmax{n_steps_max}"
    return os.path.join(BASE_PROJECT_PATH, dataset_id)

print("Configuration loaded.")
# Avoid printing all ranges here, becomes too verbose
print(f"  Default N_Qubits: {N_QUBITS}, Default Delta_T: {DELTA_T}")
print(f"  Base Project Path: {BASE_PROJECT_PATH}")
print("-" * 30)