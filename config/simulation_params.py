# quantum_ml_simulation/config/simulation_params.py
import numpy as np
import os

# --- Core Simulation Setup ---
# Keep these as defaults, can be overridden by generate_data.py args
N_QUBITS = 3 # Default N (Sim Cards 4 & 5 used N=3)
DELTA_T = 0.1 # Default dt (as per Sim Cards)
N_STEPS_RANGE = list(range(1, 40)) # Default n steps (1 to 10, as per Sim Cards)

# Default Ranges (can be overridden)
# Reduced density for faster default runs, but can be increased via CLI
J_RANGE = np.linspace(0.1, 1.0, 10).round(3).tolist() # Example J range
B_RANGE = np.linspace(0.0, 0.5, 11).round(3).tolist() # Example B range
K_RANGE = np.linspace(0.0, 0.5, 11).round(3).tolist() # Example k range (for Potential)
J1_RANGE = np.linspace(0.1, 1.0, 10).round(3).tolist() # Example J1 range (for Dimerized)
J2_RANGE = np.linspace(0.1, 1.0, 10).round(3).tolist() # Example J2 range (for Dimerized)

INITIAL_STATE_TYPE = "superposition" # |000> default as per Sim Cards
SIMULATION_MODE = "statevector" # Use exact statevector for now
N_SHOTS = 1000 # As per Sim Cards (relevant if mode='shots')

# --- Advanced Feature Flags ---
VALIDATE_TROTTER = True
VALIDATION_MAX_QUBITS = 4 # Increase slightly if needed
VALIDATION_TOLERANCE = 0.01 # Relax tolerance slightly for complex Hamiltonians
COMPUTE_ENTANGLEMENT = True
ENTANGLEMENT_PARTITION = N_QUBITS // 2 if N_QUBITS > 1 else 0

# --- Simulation-Specific Parameter Dictionaries ---
SIMULATION_CONFIGS = {
    "IsingModel": {
        "class_path": "quantum_ml_simulation.simulations.ising_model.IsingModelSimulation",
        "params": ["n_steps", "J", "B"], # Order matters
        "default_ranges": {
            "n_qubits": 3, # Sim Card 1 uses N=3
            "n_steps": list(range(1, 11)),
            "J": np.linspace(0.1, 1.0, 10).round(2).tolist(), # Card 1 J range
            "B": np.linspace(0.0, 0.5, 11).round(2).tolist() # Card 1 B range
        },
        "measurement_operator": "Z1Z2", # Card 1 specified <Z0 Z1> (using 0-based index)
        "extra_args": {}
    },
    "SpinChainPotential": {
        "class_path": "quantum_ml_simulation.simulations.spin_chain_potential.SpinChainPotentialSimulation",
        "params": ["n_steps", "k"], # J is fixed
        "default_ranges": {
            "n_qubits": 3, # Card 4 uses N=3
            "n_steps": list(range(1, 11)),
            "k": np.linspace(0.0, 0.5, 11).round(2).tolist(), # Card 4 k range
            # J is fixed, passed via extra_args
        },
        "measurement_operator": "Z1Z2", # Card 4 specified <Z0 Z1>
        "extra_args": {
            "fixed_J": 0.5 # Fixed J value as per Card 4
        }
    },
    "DimerizedHeisenberg": {
        "class_path": "quantum_ml_simulation.simulations.dimerized_heisenberg.DimerizedHeisenbergSimulation",
        "params": ["n_steps", "J1", "J2"], # Order matters
        "default_ranges": {
            "n_qubits": 3, # Card 5 uses N=3
            "n_steps": list(range(1, 11)),
            "J1": np.linspace(0.1, 1.0, 10).round(2).tolist(), # Card 5 J1 range
            "J2": np.linspace(0.1, 1.0, 10).round(2).tolist() # Card 5 J2 range
        },
        "measurement_operator": "Z1Z2", # Card 5 specified <Z0 Z1>
        "extra_args": {}
    },
}

# --- Default ML Parameters ---
# (Keep as is from previous version)
DEFAULT_ML_PARAMS = {
    "hidden_layers": [128, 64, 32], "activation": "relu", "optimizer": "adam",
    "learning_rate": 0.001, "loss": "mean_squared_error", "epochs": 100,
    "batch_size": 64, "early_stopping_patience": 15, "reduce_lr_patience": 7,
    "validation_split": 0.2, "test_split": 0.2
}

# --- ML Hyperparameter Search Space ---
# (Keep as is from previous version)
HP_SEARCH_PARAMS = {
    'learning_rate': [0.01, 0.001, 0.0005], 'batch_size': [32, 64, 128],
    'hidden_layers': [[64, 32], [128, 64], [128, 64, 32]],
    'activation': ['relu', 'tanh']
}

# --- Data Storage Configuration ---
# (Keep as is from previous version)
BASE_PROJECT_PATH = "./project_runs"
SIM_DATA_SUBDIR = "simulation_data"
ML_RESULTS_SUBDIR = "ml_results"
def get_dataset_path(simulation_name: str, n_qubits: int, dt: float, n_steps_max: int) -> str:
    dataset_id = f"{simulation_name}_N{n_qubits}_dt{dt:.3f}_nmax{n_steps_max}"
    return os.path.join(BASE_PROJECT_PATH, dataset_id)

print("Configuration loaded with IsingModel, SpinChainPotential, DimerizedHeisenberg.")
print(f"  Default N_Qubits: {N_QUBITS}, Default Delta_T: {DELTA_T}")
print(f"  Base Project Path: {BASE_PROJECT_PATH}")
print("-" * 30)