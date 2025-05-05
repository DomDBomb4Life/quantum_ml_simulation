# quantum_ml_simulation/simulations/ising_model.py
# Ising Model (Hybrid Sampling & Initial State Variation)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple, Union

# Use relative imports
from .base_simulation import BaseSimulation

class IsingModelSimulation(BaseSimulation):
    """
    Implements the N-qubit Transverse Field Ising Model simulation.
    Supports time-series generation with grid/random sampling and varying initial states.
    """
    def __init__(self,
                 n_qubits: int,
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_config: Dict,
                 simulation_mode: str,
                 sampling_method: str,
                 num_parameter_sets: Optional[int],
                 parameter_config: Dict, # Contains sampling ranges/values for J, B
                 **extra_args): # Catch any fixed args (none expected here)
        """Initializes the Ising Model simulation."""
        super().__init__(
            simulation_name="IsingModel",
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
        # Verify required system parameters are in config
        if "J" not in self.parameter_config or "B" not in self.parameter_config:
             raise ValueError("IsingModelSimulation requires 'J' and 'B' keys in parameter_config.")


    # --- Implement Abstract/Overridden Methods ---

    def get_system_parameter_names(self) -> List[str]:
        """Returns the varying system parameter names: ['J', 'B']."""
        return ["J", "B"]

    def get_param_sampling_config(self) -> Dict[str, Union[Tuple[float, float], List]]:
        """Returns the sampling config for J and B."""
        # Extract J and B config from the stored self.parameter_config
        return {
            "J": self.parameter_config.get("J"),
            "B": self.parameter_config.get("B")
        }

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, system_params: tuple):
        """Appends one symmetric Trotter step for Ising Hamiltonian."""
        J, B = system_params # Unpack system parameters
        dt = self.delta_t
        angle_rx_half = 2.0 * B * (dt / 2.0)
        angle_zz_full = 2.0 * J * dt

        if not np.isclose(B, 0.0):
            for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)
        if not np.isclose(J, 0.0):
            for i in range(self.n_qubits - 1): circuit.rzz(angle_zz_full, i, i + 1)
        if not np.isclose(B, 0.0):
            for i in range(self.n_qubits): circuit.rx(angle_rx_half, i)

    # get_ml_input_feature_names is handled by BaseSimulation

    # _format_params_for_ml_input is handled by BaseSimulation

    def get_observables_to_track(self) -> List[str]:
        """Returns observables for the ML output vector. Corrected format."""
        observables = []
        # Single qubit Z and X
        for i in range(self.n_qubits):
            observables.append(f"Z{i}")
            observables.append(f"X{i}") # Add X due to transverse field
        # Nearest-neighbor ZZ
        for i in range(self.n_qubits - 1):
            observables.append(f"ZZ{i}{i+1}")
        return observables

    def get_hamiltonian_operator(self, system_params: tuple) -> SparsePauliOp:
        """Constructs the Ising Hamiltonian using system parameters."""
        J, B = system_params
        num_qubits = self.n_qubits
        pauli_list = []
        if not np.isclose(J, 0.0):
            for i in range(num_qubits - 1):
                op_str = ['I'] * num_qubits; op_str[i] = 'Z'; op_str[i+1] = 'Z'
                pauli_list.append(("".join(op_str[::-1]), J))
        if not np.isclose(B, 0.0):
            for i in range(num_qubits):
                op_str = ['I'] * num_qubits; op_str[i] = 'X'
                pauli_list.append(("".join(op_str[::-1]), B))
        if not pauli_list:
             return SparsePauliOp('I'*num_qubits, coeffs=[0.0])
        return SparsePauliOp.from_list(pauli_list)

    # get_parameter_space_without_n is removed (handled by BaseSimulation._generate_parameter_sets)