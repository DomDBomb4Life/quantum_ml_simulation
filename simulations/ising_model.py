# quantum_ml_simulation/simulations/ising_model.py
# Implementation for the Transverse Field Ising Model (Refactored Init)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional # For type hinting

# Use relative imports
from .base_simulation import BaseSimulation
# No direct cfg import needed here for simulation parameters anymore

class IsingModelSimulation(BaseSimulation):
    """
    Implements the N-qubit Transverse Field Ising Model simulation.
    Hamiltonian: H = J * Sum_{i=0}^{N-2} Z_i Z_{i+1} + B * Sum_{i=0}^{N-1} X_i
    Uses Second-Order Symmetric Trotterization.
    Parameter ranges (J, B) are expected in the `param_ranges` dict passed to __init__.
    """
    # --- MODIFIED __init__ ---
    def __init__(self,
                 n_qubits: int,
                 measurement_operator: str,
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Dict[str, List]): # Expect 'J' and 'B' keys here
        """
        Initializes the Ising Model simulation.

        Args:
            (Inherited from BaseSimulation): n_qubits, measurement_operator, delta_t,
                                            n_steps_range, initial_state_type, simulation_mode
            param_ranges: Dictionary containing lists for 'J' and 'B' ranges.
        """
        super().__init__(
            simulation_name="IsingModel", # Set fixed name for this class
            n_qubits=n_qubits,
            measurement_operator=measurement_operator,
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_type=initial_state_type,
            simulation_mode=simulation_mode,
            param_ranges=param_ranges # Pass the ranges dict to base class
        )
        # Ranges are now stored in self.param_ranges['J'] and self.param_ranges['B']
        # Verify required ranges are present
        if 'J' not in self.param_ranges or 'B' not in self.param_ranges:
             raise ValueError("IsingModelSimulation requires 'J' and 'B' in param_ranges.")
        print(f"  Ising J Range: {len(self.param_ranges['J'])} points from {self.param_ranges['J'][0]} to {self.param_ranges['J'][-1]}")
        print(f"  Ising B Range: {len(self.param_ranges['B'])} points from {self.param_ranges['B'][0]} to {self.param_ranges['B'][-1]}")

    def get_parameter_space(self) -> list[tuple]:
        """Generates unique combinations of (n_steps, J, B) from instance attributes."""
        # Use ranges stored in self.param_ranges and self.n_steps_range
        j_range = self.param_ranges.get('J', [])
        b_range = self.param_ranges.get('B', [])
        if not self.n_steps_range or not j_range or not b_range:
             print("Warning: n_steps, J, or B range is empty. Cannot generate parameter space.")
             return []
        # The order MUST match get_ml_input_feature_names and _format_params_for_ml
        return list(itertools.product(self.n_steps_range, j_range, b_range))

    # --- get_hamiltonian_operator method remains the same ---
    # It correctly uses self.n_qubits and extracts J, B from the params tuple
    def get_hamiltonian_operator(self, params: tuple) -> SparsePauliOp:
        """Constructs the Ising Hamiltonian SparsePauliOp for given parameters."""
        # params tuple is (n_steps, J, B) based on get_parameter_space order
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
                pauli_list_zz.append(("".join(op_str[::-1]), J))

        # X terms: B * Sum X_i
        if not np.isclose(B, 0.0):
            for i in range(num_qubits):
                op_str = ['I'] * num_qubits
                op_str[i] = 'X'
                pauli_list_x.append(("".join(op_str[::-1]), B))

        hamiltonian = SparsePauliOp.from_list(pauli_list_zz + pauli_list_x)
        return hamiltonian

    # --- build_circuit_for_params method remains the same ---
    # It correctly uses self.n_qubits, self.delta_t, self.initial_state_type, self.simulation_mode
    # and extracts J, B from the params tuple
    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """Builds the 2nd-Order Trotterized circuit for the Ising model."""
        n_steps, J, B = params
        dt = self.delta_t
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"Ising_n{n_steps}_J{J:.3f}_B{B:.3f}")

        if self.initial_state_type == 'superposition':
            circuit.h(range(self.n_qubits))
            circuit.barrier()

        angle_rx_half = 2.0 * B * (dt / 2.0)
        angle_zz_full = 2.0 * J * dt

        for step in range(n_steps):
            if not np.isclose(B, 0.0):
                for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)
            if not np.isclose(J, 0.0):
                 for i in range(self.n_qubits - 1): circuit.rzz(angle_zz_full, i, i + 1)
            if not np.isclose(B, 0.0):
                for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)
            # if n_steps > 1 and step < n_steps - 1: circuit.barrier() # Optional: reduce clutter

        if self.simulation_mode == "shots":
            circuit.measure_all(inplace=True)

        return circuit

    # --- _format_params_for_ml method remains the same ---
    # It defines the structure of the ML input vector based on the params tuple order
    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """Formats (n_steps, J, B) into a list [n_steps, J, B]."""
        n_steps, J, B = params
        return [float(n_steps), float(J), float(B)]

    # --- get_ml_input_feature_names method remains the same ---
    # It defines the names corresponding to the _format_params_for_ml output order
    def get_ml_input_feature_names(self) -> list[str]:
        """Returns the feature names: ['n_steps', 'J', 'B']."""
        return ['n_steps', 'J', 'B']