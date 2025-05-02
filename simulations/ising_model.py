# quantum_ml_simulation/simulations/ising_model.py
# Implementation for the Transverse Field Ising Model (Time-Series Mode)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple

# Use relative imports
from .base_simulation import BaseSimulation

class IsingModelSimulation(BaseSimulation):
    """
    Implements the N-qubit Transverse Field Ising Model simulation for time-series data.
    Hamiltonian: H = J * Sum Z_i Z_{i+1} + B * Sum X_i
    Uses Second-Order Symmetric Trotterization.
    """
    def __init__(self,
                 n_qubits: int,
                 # measurement_operator: str, # Removed - output defined by observables
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Dict[str, List]): # Expect 'J' and 'B' keys here
        """Initializes the Ising Model simulation."""
        super().__init__(
            simulation_name="IsingModel",
            n_qubits=n_qubits,
            # measurement_operator="N/A", # Not used directly for main output
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_type=initial_state_type,
            simulation_mode=simulation_mode,
            param_ranges=param_ranges
        )
        if 'J' not in self.param_ranges or 'B' not in self.param_ranges:
             raise ValueError("IsingModelSimulation requires 'J' and 'B' in param_ranges.")
        # Print statements for ranges are now in BaseSimulation __init__ if needed


    # --- Implement Abstract Methods ---

    def get_parameter_space_without_n(self) -> Iterator[tuple]:
        """Generates unique combinations of (J, B) from instance attributes."""
        j_range = self.param_ranges.get('J', [])
        b_range = self.param_ranges.get('B', [])
        if not j_range or not b_range:
             print("Warning: J or B range is empty.")
             return iter(()) # Return empty iterator
        # Order: J, B (consistent with how params_without_n is used later)
        yield from itertools.product(j_range, b_range)

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, params_without_n: tuple):
        """Appends one symmetric Trotter step for exp(-i H delta_t)."""
        # params_without_n = (J, B)
        J, B = params_without_n
        dt = self.delta_t

        # Angles: exp(-i*angle/2*Pauli) -> angle = 2 * Coeff * time
        angle_rx_half = 2.0 * B * (dt / 2.0)
        angle_zz_full = 2.0 * J * dt

        # Symmetric Trotter: exp(-i H_X dt/2) exp(-i H_ZZ dt) exp(-i H_X dt/2)
        if not np.isclose(B, 0.0):
            for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)
        if not np.isclose(J, 0.0):
            for i in range(self.n_qubits - 1): circuit.rzz(angle_zz_full, i, i + 1)
        if not np.isclose(B, 0.0):
            for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)

    def _format_params_for_ml_input(self, n_step: int, params_without_n: tuple) -> List[float]:
        """Formats input as [n_step, J, B]."""
        J, B = params_without_n
        return [float(n_step), float(J), float(B)]

    def get_ml_input_feature_names(self) -> List[str]:
        """Returns the input feature names: ['n_steps', 'J', 'B']."""
        return ['n_steps', 'J', 'B']

    def get_observables_to_track(self) -> List[str]:
        """Returns observables for the ML output vector."""
        # Example: Single Z, nearest-neighbor ZZ, single X
        observables = []
        for i in range(self.n_qubits):
            observables.append(f"Z{i}")
        for i in range(self.n_qubits - 1):
            observables.append(f"Z{i}Z{i+1}")
        for i in range(self.n_qubits):
            observables.append(f"X{i}")
        return observables

    def get_hamiltonian_operator(self, params_without_n: tuple) -> SparsePauliOp:
        """Constructs the Ising Hamiltonian (n_steps is irrelevant here)."""
        J, B = params_without_n
        # (Logic is identical to the previous version, just using self.n_qubits)
        num_qubits = self.n_qubits
        pauli_list_zz = []
        pauli_list_x = []
        if not np.isclose(J, 0.0):
            for i in range(num_qubits - 1):
                op_str = ['I'] * num_qubits; op_str[i] = 'Z'; op_str[i+1] = 'Z'
                pauli_list_zz.append(("".join(op_str[::-1]), J))
        if not np.isclose(B, 0.0):
            for i in range(num_qubits):
                op_str = ['I'] * num_qubits; op_str[i] = 'X'
                pauli_list_x.append(("".join(op_str[::-1]), B))
        if not pauli_list_zz and not pauli_list_x:
             return SparsePauliOp('I'*num_qubits, coeffs=[0.0]) # Zero Hamiltonian
        return SparsePauliOp.from_list(pauli_list_zz + pauli_list_x)

    # build_circuit_for_params is no longer needed as _add_trotter_step handles the core logic