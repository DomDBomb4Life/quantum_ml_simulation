# quantum_ml_simulation/simulations/test_simulation.py
# Implementation for a minimal test simulation: Single Qubit Rotation

import qiskit
import itertools
import numpy as np

# Use relative imports
from .base_simulation import BaseSimulation
from ..config import simulation_params as cfg
from ..quantum_runner import circuit_builder as cb

class TestSimulation(BaseSimulation):
    """
    Implements a simple 1-qubit test simulation.
    Hamiltonian: H = B * X0
    Uses direct evolution (Rx gate) for time steps exp(-iHt).
    Measures <Z0>.
    """
    def __init__(self):
        """Initializes the Test simulation using parameters from config."""
        super().__init__(
            simulation_name="TestSim",
            n_qubits=cfg.TEST_SIM_PARAMS["n_qubits"],
            measurement_operator=cfg.TEST_SIM_PARAMS["measurement_operator"]
        )
        # Ensure n_qubits is consistent
        if self.n_qubits != 1:
             raise ValueError("TestSimulation is implemented for 1 qubit only.")

        self.b_range = cfg.TEST_SIM_PARAMS["B_range"]
        print(f"  Parameter ranges: B={self.b_range}, n={self.n_steps_range}")


    def get_parameter_space(self) -> list[tuple]:
        """Generates unique combinations of (n_steps, B)."""
        if not self.n_steps_range or self.b_range is None:
             print("Warning: Parameter range(s) for Test Simulation are empty or None.")
             return []
        return list(itertools.product(self.n_steps_range, self.b_range))

    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """Builds the circuit for the test simulation."""
        n_steps, B = params
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"Test_n{n_steps}_B{B:.2f}")
        # Initial state |0> is the default state in Qiskit

        # Total evolution time T = n_steps * delta_t
        # Total evolution U = exp(-i H T) = exp(-i * B*X0 * n_steps*delta_t)
        # We compare this to Rx(theta) = exp(-i * theta/2 * X0)
        # So, theta/2 = B * n_steps * delta_t
        # theta = 2 * B * n_steps * self.delta_t

        # Alternatively, apply step-by-step
        if not np.isclose(B, 0.0):
             # theta_step = 2 * B * self.delta_t
             # rx_gate = cb.create_rx_gate(theta_step)
             # for _ in range(n_steps):
             #      circuit.append(rx_gate, [0])

             # Simpler: Calculate total angle and apply once (since H is time-independent)
             total_angle_rx = 2 * B * n_steps * self.delta_t
             circuit.rx(total_angle_rx, 0) # Use circuit's rx method directly

        return circuit

    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """Formats (n_steps, B) into a list [n_steps, B]."""
        n_steps, B = params
        return [float(n_steps), float(B)]

    def get_ml_input_feature_names(self) -> list[str]:
        """Returns the feature names corresponding to _format_params_for_ml."""
        return ['n_steps', 'B']