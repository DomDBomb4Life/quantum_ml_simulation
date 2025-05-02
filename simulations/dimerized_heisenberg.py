# quantum_ml_simulation/simulations/dimerized_heisenberg.py
# Dimerized Heisenberg Chain (Time-Series Mode)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple

# Use relative imports
from .base_simulation import BaseSimulation

class DimerizedHeisenbergSimulation(BaseSimulation):
    """
    Implements the N-qubit Dimerized Heisenberg Chain (Time-Series Mode).
    Hamiltonian: H = J1 Sum_even[Heis] + J2 Sum_odd[Heis]
    Uses Second-Order Symmetric Trotterization.
    """
    def __init__(self,
                 n_qubits: int,
                 # measurement_operator: str, # Removed
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Dict[str, List]): # Expect 'J1' and 'J2'
        """Initializes the Dimerized Heisenberg simulation."""
        super().__init__(
            simulation_name="DimerizedHeisenberg",
            n_qubits=n_qubits,
            # measurement_operator="N/A",
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_type=initial_state_type,
            simulation_mode=simulation_mode,
            param_ranges=param_ranges
        )
        if 'J1' not in self.param_ranges or 'J2' not in self.param_ranges:
            raise ValueError("DimerizedHeisenbergSimulation requires 'J1' and 'J2' in param_ranges.")


    # --- Implement Abstract Methods ---

    def get_parameter_space_without_n(self) -> Iterator[tuple]:
        """Generates unique combinations of (J1, J2)."""
        j1_range = self.param_ranges.get('J1', [])
        j2_range = self.param_ranges.get('J2', [])
        if not j1_range or not j2_range:
            print("Warning: J1 or J2 range is empty.")
            return iter(())
        # Order: J1, J2
        yield from itertools.product(j1_range, j2_range)

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, params_without_n: tuple):
        """Appends one symmetric Trotter step."""
        # params_without_n = (J1, J2)
        J1, J2 = params_without_n
        dt = self.delta_t

        # Angles
        angle_j1_half = 2.0 * J1 * (dt / 2.0)
        angle_j2_full = 2.0 * J2 * dt

        # Symmetric Trotter: exp(-i H_J1 dt/2) exp(-i H_J2 dt) exp(-i H_J1 dt/2)
        # Apply H_J1 terms (even pairs: 0, 2, ...)
        if not np.isclose(J1, 0.0):
            for i in range(0, self.n_qubits - 1, 2):
                circuit.rxx(angle_j1_half, i, i + 1)
                circuit.ryy(angle_j1_half, i, i + 1)
                circuit.rzz(angle_j1_half, i, i + 1)
        # Apply H_J2 terms (odd pairs: 1, 3, ...)
        if not np.isclose(J2, 0.0):
            for i in range(1, self.n_qubits - 1, 2):
                circuit.rxx(angle_j2_full, i, i + 1)
                circuit.ryy(angle_j2_full, i, i + 1)
                circuit.rzz(angle_j2_full, i, i + 1)
        # Apply H_J1 terms again
        if not np.isclose(J1, 0.0):
            for i in range(0, self.n_qubits - 1, 2):
                circuit.rxx(angle_j1_half, i, i + 1)
                circuit.ryy(angle_j1_half, i, i + 1)
                circuit.rzz(angle_j1_half, i, i + 1)

    def _format_params_for_ml_input(self, n_step: int, params_without_n: tuple) -> List[float]:
        """Formats input as [n_step, J1, J2]."""
        J1, J2 = params_without_n
        return [float(n_step), float(J1), float(J2)]

    def get_ml_input_feature_names(self) -> List[str]:
        """Returns the input feature names: ['n_steps', 'J1', 'J2']."""
        return ['n_steps', 'J1', 'J2']

    def get_observables_to_track(self) -> List[str]:
        """Returns observables for the ML output vector."""
        # Similar to potential: Single Z, nearest-neighbor ZZ, XX, YY
        observables = []
        for i in range(self.n_qubits):
            observables.append(f"Z{i}")
        for i in range(self.n_qubits - 1):
            observables.append(f"Z{i}Z{i+1}")
            observables.append(f"X{i}X{i+1}")
            observables.append(f"Y{i}Y{i+1}")
        return observables

    def get_hamiltonian_operator(self, params_without_n: tuple) -> SparsePauliOp:
        """Constructs the Hamiltonian (n_steps is irrelevant here)."""
        J1, J2 = params_without_n
        # (Logic is identical to previous version)
        num_qubits = self.n_qubits
        ham_list = []
        for i in range(num_qubits - 1):
            J_current = J1 if i % 2 == 0 else J2
            if not np.isclose(J_current, 0.0):
                op_xx = ['I']*num_qubits; op_xx[i]='X'; op_xx[i+1]='X'; ham_list.append(("".join(op_xx[::-1]), J_current))
                op_yy = ['I']*num_qubits; op_yy[i]='Y'; op_yy[i+1]='Y'; ham_list.append(("".join(op_yy[::-1]), J_current))
                op_zz = ['I']*num_qubits; op_zz[i]='Z'; op_zz[i+1]='Z'; ham_list.append(("".join(op_zz[::-1]), J_current))
        if not ham_list: return SparsePauliOp('I'*num_qubits, coeffs=[0.0])
        return SparsePauliOp.from_list(ham_list)