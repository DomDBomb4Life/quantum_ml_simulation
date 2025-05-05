# quantum_ml_simulation/simulations/spin_chain_potential.py
# Heisenberg Spin Chain with Potential (Hybrid Sampling & Initial State Variation)

import qiskit
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Dict, Optional, Iterator, Tuple, Union

# Use relative imports
from .base_simulation import BaseSimulation

class SpinChainPotentialSimulation(BaseSimulation):
    """
    Implements the N-qubit Heisenberg Chain with Potential (Time-Series Mode).
    Supports grid/random sampling for 'k' and varying initial states.
    J is fixed via extra_args.
    """
    def __init__(self,
                 n_qubits: int,
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_config: Dict,
                 simulation_mode: str,
                 sampling_method: str,
                 num_parameter_sets: Optional[int],
                 parameter_config: Dict, # Expects 'k' sampling info
                 fixed_J: float, # Passed via extra_args
                 **extra_args): # Catch others
        """Initializes the Spin Chain with Potential simulation."""
        super().__init__(
            simulation_name="SpinChainPotential",
            n_qubits=n_qubits,
            delta_t=delta_t,
            n_steps_range=n_steps_range,
            initial_state_config=initial_state_config,
            simulation_mode=simulation_mode,
            sampling_method=sampling_method,
            num_parameter_sets=num_parameter_sets,
            parameter_config=parameter_config,
            fixed_J=fixed_J, # Store fixed J
            **extra_args
        )
        self.fixed_J = fixed_J # Also store directly for easier access
        if 'k' not in self.parameter_config:
             raise ValueError("SpinChainPotential requires 'k' key in parameter_config.")
        if self.n_qubits != 3:
             print(f"Warning: Potential V_i = [k, 2k, k] specific to N=3. Using N={self.n_qubits}.")
        print(f"  SpinChainPotential Fixed J: {self.fixed_J}")

    def get_potential_vector(self, k: float) -> List[float]:
         """Returns the potential V_i for each qubit based on k."""
         if self.n_qubits == 3: return [k, 2*k, k]
         else: print(f"Warning: Using default potential V_i=k for N={self.n_qubits}."); return [k]*self.n_qubits

    # --- Implement Abstract/Overridden Methods ---

    def get_system_parameter_names(self) -> List[str]:
        """Returns the varying system parameter names: ['k']."""
        return ["k"]

    def get_param_sampling_config(self) -> Dict[str, Union[Tuple[float, float], List]]:
        """Returns the sampling config for k."""
        return {"k": self.parameter_config.get("k")}

    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, system_params: tuple):
        """Appends one symmetric Trotter step."""
        k, = system_params # Unpack system parameter 'k'
        J = self.fixed_J    # Use the fixed J value
        dt = self.delta_t
        angle_heis_half = 2.0 * J * (dt / 2.0)
        potential_values = self.get_potential_vector(k)
        angles_potential_full = [2.0 * V_i * dt for V_i in potential_values]

        # Symmetric Trotter step
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
        """Constructs the Hamiltonian using system parameter k and fixed J."""
        k, = system_params
        J = self.fixed_J
        num_qubits = self.n_qubits
        ham_list = []
        # Heisenberg term
        if not np.isclose(J, 0.0):
            for i in range(num_qubits - 1):
                op_xx = ['I']*num_qubits; op_xx[i]='X'; op_xx[i+1]='X'; ham_list.append(("".join(op_xx[::-1]), J))
                op_yy = ['I']*num_qubits; op_yy[i]='Y'; op_yy[i+1]='Y'; ham_list.append(("".join(op_yy[::-1]), J))
                op_zz = ['I']*num_qubits; op_zz[i]='Z'; op_zz[i+1]='Z'; ham_list.append(("".join(op_zz[::-1]), J))
        # Potential term
        potential_values = self.get_potential_vector(k)
        for i in range(num_qubits):
             V_i = potential_values[i]
             if not np.isclose(V_i, 0.0):
                 op_z = ['I']*num_qubits; op_z[i]='Z'; ham_list.append(("".join(op_z[::-1]), V_i))
        if not ham_list: return SparsePauliOp('I'*num_qubits, coeffs=[0.0])
        return SparsePauliOp.from_list(ham_list)