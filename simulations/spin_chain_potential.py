# quantum_ml_simulation/simulations/spin_chain_potential.py
# Heisenberg Spin Chain with Potential (Time-Series Mode)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple

# Use relative imports
from .base_simulation import BaseSimulation

class SpinChainPotentialSimulation(BaseSimulation):
    """
    Implements the N-qubit Heisenberg Chain with Potential (Time-Series Mode).
    Hamiltonian: H = J Sum[H_Heis] + Sum[V_i Z_i]
    Uses Second-Order Symmetric Trotterization.
    """
    def __init__(self,
                 n_qubits: int,
                 # measurement_operator: str, # Removed
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Dict[str, List], # Expects 'k'
                 fixed_J: float):
        """Initializes the Spin Chain with Potential simulation."""
        super().__init__(
            simulation_name="SpinChainPotential",
            n_qubits=n_qubits,
            # measurement_operator="N/A",
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_type=initial_state_type,
            simulation_mode=simulation_mode,
            param_ranges=param_ranges
        )
        self.fixed_J = fixed_J
        if 'k' not in self.param_ranges:
            raise ValueError("SpinChainPotentialSimulation requires 'k' in param_ranges.")
        if self.n_qubits != 3:
             print(f"Warning: Potential V_i = [k, 2k, k] is specific to N=3. Using N={self.n_qubits}.")
        print(f"  SpinChainPotential Fixed J: {self.fixed_J}")


    def get_potential_vector(self, k: float) -> List[float]:
         """Returns the potential V_i for each qubit based on k."""
         # Keep the N=3 specific potential for now
         if self.n_qubits == 3:
             return [k, 2*k, k]
         else:
             print(f"Warning: Using default potential V_i = k for N={self.n_qubits}.")
             return [k] * self.n_qubits

    # --- Implement Abstract Methods ---

    def get_parameter_space_without_n(self) -> Iterator[tuple]:
        """Generates unique combinations of (k,) parameters."""
        k_range = self.param_ranges.get('k', [])
        if not k_range: print("Warning: k range is empty."); return iter(())
        # Yield tuples (even single-element ones)
        yield from ((k,) for k in k_range) # Note the comma to make it a tuple

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, params_without_n: tuple):
        """Appends one symmetric Trotter step."""
        # params_without_n = (k,)
        k, = params_without_n # Unpack the single element tuple
        J = self.fixed_J
        dt = self.delta_t

        # Angles
        angle_heis_half = 2.0 * J * (dt / 2.0)
        potential_values = self.get_potential_vector(k)
        angles_potential_full = [2.0 * V_i * dt for V_i in potential_values]

        # Symmetric Trotter: exp(-i H_int dt/2) exp(-i H_pot dt) exp(-i H_int dt/2)
        if not np.isclose(J, 0.0):
            for i in range(self.n_qubits - 1):
                circuit.rxx(angle_heis_half, i, i + 1)
                circuit.ryy(angle_heis_half, i, i + 1)
                circuit.rzz(angle_heis_half, i, i + 1)
        for i in range(self.n_qubits):
            if not np.isclose(angles_potential_full[i], 0.0):
                circuit.rz(angles_potential_full[i], i)
        if not np.isclose(J, 0.0):
            for i in range(self.n_qubits - 1):
                circuit.rxx(angle_heis_half, i, i + 1)
                circuit.ryy(angle_heis_half, i, i + 1)
                circuit.rzz(angle_heis_half, i, i + 1)

    def _format_params_for_ml_input(self, n_step: int, params_without_n: tuple) -> List[float]:
        """Formats input as [n_step, k]."""
        k, = params_without_n
        return [float(n_step), float(k)]

    def get_ml_input_feature_names(self) -> List[str]:
        """Returns the input feature names: ['n_steps', 'k']."""
        return ['n_steps', 'k']

    def get_observables_to_track(self) -> List[str]:
        """Returns observables for the ML output vector."""
        # Example: Single Z, nearest-neighbor ZZ, maybe XX/YY correlations
        observables = []
        for i in range(self.n_qubits):
            observables.append(f"Z{i}")
        for i in range(self.n_qubits - 1):
            observables.append(f"Z{i}Z{i+1}")
            observables.append(f"X{i}X{i+1}") # Add XX and YY?
            observables.append(f"Y{i}Y{i+1}")
        return observables

    def get_hamiltonian_operator(self, params_without_n: tuple) -> SparsePauliOp:
        """Constructs the Hamiltonian (n_steps is irrelevant here)."""
        k, = params_without_n
        J = self.fixed_J
        # (Logic is identical to the previous version)
        num_qubits = self.n_qubits
        ham_list = []
        if not np.isclose(J, 0.0):
            for i in range(num_qubits - 1):
                op_xx = ['I']*num_qubits; op_xx[i]='X'; op_xx[i+1]='X'; ham_list.append(("".join(op_xx[::-1]), J))
                op_yy = ['I']*num_qubits; op_yy[i]='Y'; op_yy[i+1]='Y'; ham_list.append(("".join(op_yy[::-1]), J))
                op_zz = ['I']*num_qubits; op_zz[i]='Z'; op_zz[i+1]='Z'; ham_list.append(("".join(op_zz[::-1]), J))
        potential_values = self.get_potential_vector(k)
        for i in range(num_qubits):
             V_i = potential_values[i]
             if not np.isclose(V_i, 0.0):
                 op_z = ['I']*num_qubits; op_z[i]='Z'; ham_list.append(("".join(op_z[::-1]), V_i))
        if not ham_list: return SparsePauliOp('I'*num_qubits, coeffs=[0.0])
        return SparsePauliOp.from_list(ham_list)