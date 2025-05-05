# quantum_ml_simulation/config/simulation_params.py
# Configuration for simulations, ML, sampling, and initial states.

import numpy as np
import os

# --- Core Simulation Setup ---
N_QUBITS = 3 # Default N
DELTA_T = 0.1 # Default dt
N_STEPS_RECORD_POINTS = list(range(1, 11)) # Default list of steps to record

# --- Data Generation Strategy ---
SAMPLING_METHOD = "random" # 'grid' or 'random'
DEFAULT_NUM_RANDOM_PARAMETER_SETS = 1000 # Used if SAMPLING_METHOD is 'random'
# Note: If 'grid', number of sets is determined by product of range lengths below.

# --- Initial State Configuration ---
INITIAL_STATE_CONFIG = {
    "type": "random_rotations", # 'zero', 'superposition', 'random_rotations'
    # Parameters for 'random_rotations': Angles sampled uniformly
    "rotation_angle_range": (0, 2 * np.pi), # Tuple (min, max) for theta, phi, lambda
}

# --- Simulation Backend ---
SIMULATION_MODE = "statevector" # 'statevector' or 'shots'
N_SHOTS = 1000

# --- Advanced Feature Flags ---
VALIDATE_TROTTER = True
VALIDATION_MAX_QUBITS = 4
VALIDATION_TOLERANCE = 0.01
COMPUTE_ENTANGLEMENT = True
ENTANGLEMENT_PARTITION = N_QUBITS // 2 if N_QUBITS > 1 else 0

# --- Simulation-Specific Parameter Dictionaries ---
# 'params': System parameters that vary (ML features excluding n_steps & initial state).
# 'default_sampling_ranges': Defines (min, max) for random sampling OR list for grid sampling.
# 'extra_args': Fixed parameters for the simulation class __init__.
SIMULATION_CONFIGS = {
    "IsingModel": {
        "class_path": "quantum_ml_simulation.simulations.ising_model.IsingModelSimulation",
        "params": ["J", "B"], # System parameters that vary
        "default_sampling_ranges": {
            "n_qubits": 3,
            "n_steps": N_STEPS_RECORD_POINTS, # Use global default list of steps
            # Define ranges for sampling or grid points
            "J": (0.1, 1.0), # For random sampling (min, max)
             # "J": np.linspace(0.1, 1.0, 10).round(2).tolist(), # For grid sampling
            "B": (0.0, 0.5), # For random sampling (min, max)
             # "B": np.linspace(0.0, 0.5, 11).round(2).tolist(), # For grid sampling
        },
        "extra_args": {}
    },
    "SpinChainPotential": {
        "class_path": "quantum_ml_simulation.simulations.spin_chain_potential.SpinChainPotentialSimulation",
        "params": ["k"], # System parameter that varies
        "default_sampling_ranges": {
            "n_qubits": 3,
            "n_steps": N_STEPS_RECORD_POINTS,
            "k": (0.0, 0.5), # For random sampling
             # "k": np.linspace(0.0, 0.5, 11).round(2).tolist(), # For grid sampling
        },
        "extra_args": {"fixed_J": 0.5} # Fixed J
    },
    "DimerizedHeisenberg": {
        "class_path": "quantum_ml_simulation.simulations.dimerized_heisenberg.DimerizedHeisenbergSimulation",
        "params": ["J1", "J2"], # System parameters that vary
        "default_sampling_ranges": {
            "n_qubits": 3,
            "n_steps": N_STEPS_RECORD_POINTS,
            "J1": (0.1, 1.0), # For random sampling
             # "J1": np.linspace(0.1, 1.0, 10).round(2).tolist(), # For grid sampling
            "J2": (0.1, 1.0), # For random sampling
             # "J2": np.linspace(0.1, 1.0, 10).round(2).tolist(), # For grid sampling
        },
        "extra_args": {}
    },
}

# --- Default ML Parameters ---
# (Unchanged - Keep as is)
DEFAULT_ML_PARAMS = {
    "hidden_layers": [128, 64, 32], "activation": "relu", "optimizer": "adam",
    "learning_rate": 0.001, "loss": "mean_squared_error", "epochs": 100,
    "batch_size": 64, "early_stopping_patience": 15, "reduce_lr_patience": 7,
    "validation_split": 0.2, "test_split": 0.2
}

# --- ML Hyperparameter Search Space ---
# (Unchanged - Keep as is)
HP_SEARCH_PARAMS = {
    'learning_rate': [0.01, 0.001, 0.0005], 'batch_size': [32, 64, 128],
    'hidden_layers': [[64, 32], [128, 64], [128, 64, 32]],
    'activation': ['relu', 'tanh', "leaky_relu"]
}

# --- Data Storage Configuration ---
# (Unchanged - Keep as is)
BASE_PROJECT_PATH = "./project_runs"
SIM_DATA_SUBDIR = "simulation_data"
ML_RESULTS_SUBDIR = "ml_results"
def get_dataset_path(simulation_name: str, n_qubits: int, dt: float, n_steps_max: int) -> str:
    dataset_id = f"{simulation_name}_N{n_qubits}_dt{dt:.3f}_nmax{n_steps_max}"
    # Optional: Add sampling method/size to path?
    # dataset_id += f"_{SAMPLING_METHOD}"
    return os.path.join(BASE_PROJECT_PATH, dataset_id)

print("Configuration loaded (Hybrid Sampling & Initial State Ready).")
print(f"  Default Sampling Method: {SAMPLING_METHOD}")
print(f"  Default Initial State: {INITIAL_STATE_CONFIG['type']}")
print(f"  Default N_Qubits: {N_QUBITS}, Default Delta_T: {DELTA_T}")
print(f"  Base Project Path: {BASE_PROJECT_PATH}")
print("-" * 30)