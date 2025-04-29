# # quantum_ml_simulation/simulations/ising_model.py
# Implementation for the Transverse Field Ising Model (Enhanced)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp

# Use relative imports
from .base_simulation import BaseSimulation
from ..config import simulation_params as cfg

class IsingModelSimulation(BaseSimulation):
    """
    Implements the N-qubit Transverse Field Ising Model simulation.
    Hamiltonian: H = J * Sum_{i=0}^{N-2} Z_i Z_{i+1} + B * Sum_{i=0}^{N-1} X_i
    Uses Second-Order Symmetric Trotterization: exp(-iHt) ~ [exp(-i H_X dt/2) exp(-i H_ZZ dt) exp(-i H_X dt/2)]^n_steps
    Initial state can be |0...0> or |+...+\>.
    Measures specified operator (e.g., <Z0Z1>).
    """
    def __init__(self):
        """Initializes the Ising Model simulation using parameters from config."""
        super().__init__(
            simulation_name="IsingModel",
            n_qubits=cfg.ISING_PARAMS["n_qubits"],
            measurement_operator=cfg.ISING_PARAMS["measurement_operator"]
        )
        self.j_range = cfg.ISING_PARAMS["J_range"]
        self.b_range = cfg.ISING_PARAMS["B_range"]
        # Note: n_steps_range and delta_t are inherited from BaseSimulation/config

    def get_parameter_space(self) -> list[tuple]:
        """Generates unique combinations of (n_steps, J, B)."""
        if not self.n_steps_range or not self.j_range or self.b_range is None:
             print("Warning: One or more parameter ranges for Ising Model are empty or None.")
             return []
        return list(itertools.product(self.n_steps_range, self.j_range, self.b_range))

    def get_hamiltonian_operator(self, params: tuple) -> SparsePauliOp:
        """Constructs the Ising Hamiltonian SparsePauliOp for given parameters."""
        _, J, B = params # n_steps is not part of the Hamiltonian itself
        num_qubits = self.n_qubits
        pauli_list_zz = []
        pauli_list_x = []

        # ZZ terms: J * Sum Z_i Z_{i+1}
        if not np.isclose(J, 0.0):
            for i in range(num_qubits - 1):
                op_str = ['I'] * num_qubits
                op_str[i] = 'Z'
                op_str[i+1] = 'Z'
                # Convert to Qiskit order (qN-1...q0) and add J coefficient
                pauli_list_zz.append(("".join(op_str[::-1]), J))

        # X terms: B * Sum X_i
        if not np.isclose(B, 0.0):
            for i in range(num_qubits):
                op_str = ['I'] * num_qubits
                op_str[i] = 'X'
                 # Convert to Qiskit order (qN-1...q0) and add B coefficient
                pauli_list_x.append(("".join(op_str[::-1]), B))

        # Combine terms
        hamiltonian = SparsePauliOp.from_list(pauli_list_zz + pauli_list_x)
        return hamiltonian

    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """Builds the 2nd-Order Trotterized circuit for the Ising model."""
        n_steps, J, B = params
        dt = self.delta_t
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"Ising_n{n_steps}_J{J}_B{B}")

        # --- Initial State ---
        if self.initial_state_type == 'superposition':
            circuit.h(range(self.n_qubits))
            circuit.barrier() # Separate initial state prep visually
        # Else: default is |0...0>

        # --- Second-Order Trotter Steps ---
        # exp(-iHt) ~ [exp(-i H_X dt/2) exp(-i H_ZZ dt) exp(-i H_X dt/2)]^n_steps
        # Note on angles: Qiskit gates exp(-i*angle/2*Pauli)
        # For exp(-i * Coeff * Pauli * time), angle = 2 * Coeff * time

        angle_rx_half = 2.0 * B * (dt / 2.0) # Angle for X terms for dt/2
        angle_zz_full = 2.0 * J * dt        # Angle for ZZ terms for dt

        for step in range(n_steps):
            # --- Step 1: exp(-i H_X dt/2) ---
            if not np.isclose(B, 0.0):
                for i in range(self.n_qubits):
                    circuit.rx(angle_rx_half, i)

            # --- Step 2: exp(-i H_ZZ dt) ---
            if not np.isclose(J, 0.0):
                 for i in range(self.n_qubits - 1):
                    circuit.rzz(angle_zz_full, i, i + 1) # Apply to adjacent qubits

            # --- Step 3: exp(-i H_X dt/2) ---
            if not np.isclose(B, 0.0):
                for i in range(self.n_qubits):
                    circuit.rx(angle_rx_half, i)

            # Optional barrier between full Trotter steps
            if n_steps > 1 and step < n_steps - 1:
                 circuit.barrier(label=f"Trotter Step {step+1}")


        # --- Add Measurement for Shots Mode ---
        # If running in shots mode, measurements are needed.
        # Measure all qubits to corresponding classical bits.
        # We add this here, assuming the simulator checks for it.
        if self.q_simulator.simulation_mode == "shots":
            circuit.measure_all(inplace=True)

        return circuit

    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """Formats (n_steps, J, B) into a list [n_steps, J, B]."""
        n_steps, J, B = params
        # Add n_qubits as a feature? Or assume fixed per dataset?
        # For now, keep it simple.
        return [float(n_steps), float(J), float(B)]

    def get_ml_input_feature_names(self) -> list[str]:
        """Returns the feature names corresponding to _format_params_for_ml."""
        # If n_qubits added above, add 'n_qubits' here
        return ['n_steps', 'J', 'B']