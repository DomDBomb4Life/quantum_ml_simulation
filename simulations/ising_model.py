# quantum_ml_simulation/simulations/ising_model.py
# Implementation for Simulation Card 1: Transverse Field Ising Model

import qiskit
import itertools
import numpy as np

# Use relative imports
from .base_simulation import BaseSimulation
from ..config import simulation_params as cfg
from ..quantum_runner import circuit_builder as cb

class IsingModelSimulation(BaseSimulation):
    """
    Implements the 3-qubit Transverse Field Ising Model simulation.
    Hamiltonian: H = J (Z0Z1 + Z1Z2) + B (X0 + X1 + X2)
    Uses Trotterization for time evolution exp(-iHt).
    Measures <Z0Z1>.
    """
    def __init__(self):
        """Initializes the Ising Model simulation using parameters from config."""
        super().__init__(
            simulation_name="IsingModel",
            n_qubits=cfg.ISING_PARAMS["n_qubits"],
            measurement_operator=cfg.ISING_PARAMS["measurement_operator"]
        )
        # Ensure n_qubits is consistent
        if self.n_qubits != 3:
             raise ValueError("IsingModelSimulation currently implemented for 3 qubits only.")

        self.j_range = cfg.ISING_PARAMS["J_range"]
        self.b_range = cfg.ISING_PARAMS["B_range"]
        print(f"  Parameter ranges: J={self.j_range}, B={self.b_range}, n={self.n_steps_range}")

    def get_parameter_space(self) -> list[tuple]:
        """Generates unique combinations of (n_steps, J, B)."""
        # Ensure ranges are valid
        if not self.n_steps_range or not self.j_range or self.b_range is None:
             print("Warning: One or more parameter ranges for Ising Model are empty or None.")
             return []
        return list(itertools.product(self.n_steps_range, self.j_range, self.b_range))

    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """Builds the Trotterized circuit for one time step of the Ising model."""
        n_steps, J, B = params
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"Ising_n{n_steps}_J{J}_B{B}")
        # Initial state |000> is the default state in Qiskit

        # --- Trotter Steps ---
        for step in range(n_steps):
            # Apply ZZ interactions (Part 1 of Trotter step: exp(-i H_ZZ * dt))
            # H_ZZ = J (Z0Z1 + Z1Z2)
            if not np.isclose(J, 0.0):
                # Angle consideration: If H = J*Op, evolution is exp(-i J*Op*dt).
                # If Rzz(theta) implements exp(-i * theta * ZZ), then theta = J * dt.
                # If Rzz(theta) implements exp(-i * theta/2 * ZZ), then theta = 2 * J * dt.
                # Assuming create_rzz_gate implements exp(-i * angle * ZZ) based on its Rz usage.
                angle_zz = J * self.delta_t

                # Apply Rzz to qubits (0, 1)
                rzz_gate_01 = cb.create_rzz_gate(angle_zz)
                circuit.append(rzz_gate_01, [0, 1])

                # Apply Rzz to qubits (1, 2)
                rzz_gate_12 = cb.create_rzz_gate(angle_zz)
                circuit.append(rzz_gate_12, [1, 2])

            # Apply Transverse Field (Part 2 of Trotter step: exp(-i H_X * dt))
            # H_X = B (X0 + X1 + X2)
            if not np.isclose(B, 0.0):
                # Angle consideration: If Rx(theta) implements exp(-i * theta/2 * X),
                # then for H=B*X, we need theta/2 = B*dt => theta = 2 * B * dt.
                # Qiskit's RXGate(angle) implements exp(-i * angle/2 * X).
                angle_rx = 2 * B * self.delta_t
                rx_gate = cb.create_rx_gate(angle_rx) # Use the function which returns RXGate instance

                for q_idx in range(self.n_qubits):
                   circuit.append(rx_gate, [q_idx]) # Append the RXGate object

            # Optional barrier for visualization per Trotter step
            if n_steps > 1 and step < n_steps - 1 : # Don't add after last step
                 circuit.barrier()

        return circuit

    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """Formats (n_steps, J, B) into a list [n_steps, J, B]."""
        n_steps, J, B = params
        return [float(n_steps), float(J), float(B)]

    def get_ml_input_feature_names(self) -> list[str]:
        """Returns the feature names corresponding to _format_params_for_ml."""
        return ['n_steps', 'J', 'B']