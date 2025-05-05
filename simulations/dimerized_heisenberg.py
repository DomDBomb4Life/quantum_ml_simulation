# quantum_ml_simulation/simulations/dimerized_heisenberg.py
# Dimerized Heisenberg Chain (Hybrid Sampling & Initial State Variation)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple, Union

# Use relative imports
from .base_simulation import BaseSimulation

class DimerizedHeisenbergSimulation(BaseSimulation):
    """
    Implements the N-qubit Dimerized Heisenberg Chain (Time-Series Mode).
    Supports grid/random sampling for J1, J2 and varying initial states.
    """
    def __init__(self,
                 n_qubits: int,
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_config: Dict,
                 simulation_mode: str,
                 sampling_method: str,
                 num_parameter_sets: Optional[int],
                 parameter_config: Dict, # Expects 'J1', 'J2' sampling info
                 **extra_args):
        """Initializes the Dimerized Heisenberg simulation."""
        super().__init__(
            simulation_name="DimerizedHeisenberg",
            n_qubits=n_qubits,
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_config=initial_state_config,
            simulation_mode=simulation_mode,
            sampling_method=sampling_method,
            num_parameter_sets=num_parameter_sets,
            parameter_config=parameter_config,
            **extra_args
        )
        if 'J1' not in self.parameter_config or 'J2' not in self.parameter_config:
            raise ValueError("DimerizedHeisenbergSimulation requires 'J1' and 'J2' keys in parameter_config.")

    # --- Implement Abstract/Overridden Methods ---

    def get_system_parameter_names(self) -> List[str]:
        """Returns the varying system parameter names: ['J1', 'J2']."""
        return ["J1", "J2"]

    def get_param_sampling_config(self) -> Dict[str, Union[Tuple[float, float], List]]:
        """Returns the sampling config for J1 and J2."""
        return {
            "J1": self.parameter_config.get("J1"),
            "J2": self.parameter_config.get("J2")
        }

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, system_params: tuple):
        """Appends one symmetric Trotter step."""
        J1, J2 = system_params
        dt = self.delta_t
        angle_j1_half = 2.0 * J1 * (dt / 2.0)
        angle_j2_full = 2.0 * J2 * dt

        # Symmetric Trotter: exp(-i H_J1 dt/2) exp(-i H_J2 dt) exp(-i H_J1 dt/2)
        if not np.isclose(J1, 0.0):
            for i in range(0, self.n_qubits - 1, 2):
                circuit.rxx(angle_j1_half, i, i + 1); circuit.ryy(angle_j1_half, i, i + 1); circuit.rzz(angle_j1_half, i, i + 1)
        if not np.isclose(J2, 0.0):
            for i in range(1, self.n_qubits - 1, 2):
                circuit.rxx(angle_j2_full, i, i + 1); circuit.ryy(angle_j2_full, i, i + 1); circuit.rzz(angle_j2_full, i, i + 1)
        if not np.isclose(J1, 0.0):
            for i in range(0, self.n_qubits - 1, 2):
                circuit.rxx(angle_j1_half, i, i + 1); circuit.ryy(angle_j1_half, i, i + 1); circuit.rzz(angle_j1_half, i, i + 1)

    # get_ml_input_feature_names is handled by BaseSimulation
    # _format_params_for_ml_input is handled by BaseSimulation

    def get_observables_to_track(self) -> List[str]:
        """Returns observables for the ML output vector. Corrected format."""
        observables = []
        for i in range(self.n_qubits): observables.append(f"Z{i}")
        for i in range(self.n_qubits - 1):
            observables.append(f"ZZ{i}{i+1}")
            observables.append(f"XX{i}{i+1}")
            observables.append(f"YY{i}{i+1}")
        return observables

    def get_hamiltonian_operator(self, system_params: tuple) -> SparsePauliOp:
        """Constructs the Hamiltonian using system parameters J1, J2."""
        J1, J2 = system_params
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