# quantum_ml_simulation/config/simulation_params.py
# Contains configuration parameters for the simulations
import numpy as np
import os

# --- Core Simulation Setup ---
# Objective 4: Scale System Size
N_QUBITS = 4 # Increased system size (3 is too small for complexity)

# Objective 5: Extend Evolution Time and Parameter Ranges
DELTA_T = 0.05 # Smaller time step for better accuracy
N_STEPS_RANGE = list(range(1, 21)) # n = 1 to 20 (total time T up to 1.0)

# Parameter Ranges (using linspace for density)
# J: Interaction strength
J_RANGE = np.linspace(0.1, 2.0, 21).round(3).tolist() # 0.1 to 2.0 (21 points)
# B: Transverse field strength
B_RANGE = np.linspace(0.0, 1.0, 21).round(3).tolist() # 0.0 to 1.0 (21 points)

# Objective 5: Use Superposition Initial State (handled in simulation class)
INITIAL_STATE_TYPE = "superposition" # 'zero' or 'superposition'

# Objective 6: Incorporate Shot-Based Measurements
SIMULATION_MODE = "statevector" # 'statevector' or 'shots'
N_SHOTS = 2048 # Number of shots for shot-based simulation

# --- Controls for Advanced Features ---
# Objective 2: Add Validation Against Exact Diagonalization
VALIDATE_TROTTER = True # Enable validation for small systems?
VALIDATION_MAX_QUBITS = 4 # Only validate up to this many qubits (ED is expensive)
VALIDATION_TOLERANCE = 0.005 # Tolerance for validation difference

# Objective 7: Quantify Entanglement
COMPUTE_ENTANGLEMENT = True # Compute bipartite entanglement entropy?
ENTANGLEMENT_PARTITION = N_QUBITS // 2 # Partition size for entanglement (e.g., first half)


# --- Simulation-Specific Parameters (Example: Ising) ---
ISING_PARAMS = {
    "n_qubits": N_QUBITS, # Use global N_QUBITS
    "J_range": J_RANGE,   # Use global J_RANGE
    "B_range": B_RANGE,   # Use global B_RANGE
    "measurement_operator": "Z1Z2" # Measure correlation on qubits 0, 1
    # Add other simulation types (Heisenberg, etc.) here if needed
}

# --- ML Parameters (Keep defaults for now, adjust after data generation) ---
ML_MODEL_PARAMS = {
    "input_dim": None,
    "output_dim": 1, # Still predicting one primary value (<Z1Z2>)
    "hidden_layers": [128, 64, 32], # Slightly deeper for potentially more complex data
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mean_squared_error",
    "epochs": 200,
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.2
}

# --- Data Storage Paths ---
BASE_SAVE_PATH = f"./project_results_nq{N_QUBITS}_t{DELTA_T:.2f}" # More descriptive path
DATA_PATH = os.path.join(BASE_SAVE_PATH, "simulation_data")
ML_RESULTS_PATH = os.path.join(BASE_SAVE_PATH, "ml_results")

# Ensure base directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(ML_RESULTS_PATH, exist_ok=True)

print(f"Configuration loaded:")
print(f"  N_Qubits: {N_QUBITS}, Delta_T: {DELTA_T}, N_Steps: {N_STEPS_RANGE[0]}-{N_STEPS_RANGE[-1]}")
print(f"  J Range: {J_RANGE[0]} to {J_RANGE[-1]} ({len(J_RANGE)} points)")
print(f"  B Range: {B_RANGE[0]} to {B_RANGE[-1]} ({len(B_RANGE)} points)")
print(f"  Initial State: {INITIAL_STATE_TYPE}")
print(f"  Simulation Mode: {SIMULATION_MODE}" + (f", Shots: {N_SHOTS}" if SIMULATION_MODE == 'shots' else ""))
print(f"  Validate Trotter (<= {VALIDATION_MAX_QUBITS} qubits): {VALIDATE_TROTTER}")
print(f"  Compute Entanglement (Partition: {ENTANGLEMENT_PARTITION}): {COMPUTE_ENTANGLEMENT}")
print(f"  Data will be saved in: {BASE_SAVE_PATH}")
print("-" * 30)