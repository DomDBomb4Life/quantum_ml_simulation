# quantum_ml_simulation/config/simulation_params.py
# Contains configuration parameters for the simulations
import numpy as np
import os

# --- Common Simulation Parameters ---
N_QUBITS_DEFAULT = 3 # Default for most simulations
INITIAL_STATE = "000" # Represents |000> for 3 qubits
N_SHOTS = 1000       # Default shots (used if switching to sampling backend)
DELTA_T = 0.1        # Time step duration
N_STEPS_RANGE = list(range(1, 11)) # n = 1 to 10 (number of Trotter steps)

# --- Simulation 1: Ising Model ---
ISING_PARAMS = {
    "n_qubits": N_QUBITS_DEFAULT,
    "J_range": [round(0.1 * i, 2) for i in range(1, 11)], # 0.1 to 1.0
    "B_range": [round(0.05 * i, 2) for i in range(0, 11)], # 0 to 0.5
    "measurement_operator": "Z1Z2" # String identifier for <Z_0 Z_1> (on qubits 0 and 1)
}

# --- Simulation 4: Spin Chain Potential ---
SPIN_CHAIN_POTENTIAL_PARAMS = {
    "n_qubits": N_QUBITS_DEFAULT,
    "J_fixed": 0.5,
    "k_range": [round(0.05 * i, 2) for i in range(0, 11)], # 0 to 0.5
    "measurement_operator": "Z1Z2" # <Z_0 Z_1>
}

# --- Simulation 5: Dimerized Heisenberg ---
DIMERIZED_HEISENBERG_PARAMS = {
    "n_qubits": N_QUBITS_DEFAULT,
    "J1_range": [round(0.1 * i, 2) for i in range(1, 11)],
    "J2_range": [round(0.1 * i, 2) for i in range(1, 11)],
    "measurement_operator": "Z1Z2" # <Z_0 Z_1>
}

# --- Test Simulation: Single Qubit Rotation ---
TEST_SIM_PARAMS = {
    "n_qubits": 1,
    "B_range": [round(0.1 * np.pi * i, 3) for i in range(11)], # 0 to pi
    "measurement_operator": "Z0" # <Z> on qubit 0
}


# --- ML Parameters ---
ML_MODEL_PARAMS = {
    "input_dim": None, # To be set dynamically based on simulation feature count
    "output_dim": 1, # Predicting a single expectation value
    "hidden_layers": [64, 32], # Example architecture
    "activation": "relu",
    "optimizer": "adam", # Default optimizer
    "learning_rate": 0.001,
    "loss": "mean_squared_error",
    "epochs": 150, # Increased epochs slightly for potential overfitting goal
    "batch_size": 32,
    "validation_split": 0.2, # Fraction of *training* data used for validation during training
    "test_split": 0.2 # Fraction of *total* data held out for final testing
}

# --- Data Storage Paths ---
BASE_SAVE_PATH = "./project_results_mvp" # Changed for MVP stage
DATA_PATH = os.path.join(BASE_SAVE_PATH, "simulation_data")
ML_RESULTS_PATH = os.path.join(BASE_SAVE_PATH, "ml_results")

# Ensure base directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(ML_RESULTS_PATH, exist_ok=True)

print(f"Configuration loaded. Data will be saved in: {BASE_SAVE_PATH}")