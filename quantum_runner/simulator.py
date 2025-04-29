# quantum_ml_simulation/quantum_runner/simulator.py
# Runs quantum circuits and calculates results using Qiskit Aer

import qiskit
# Ensure qiskit-aer is installed: pip install qiskit-aer
try:
    from qiskit_aer import AerSimulator
except ImportError:
    print("ERROR: qiskit-aer not found. Please install it: pip install qiskit-aer")
    raise

from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
from ..config import simulation_params as cfg # Use relative import

class QuantumSimulator:
    """Handles the execution of quantum circuits and calculation of expectation values."""

    def __init__(self, backend_name: str = "aer_simulator_statevector", n_shots: int = cfg.N_SHOTS):
        """
        Initializes the simulator.

        Args:
            backend_name: Name of the Qiskit Aer backend (default: statevector simulator).
            n_shots: Number of shots (relevant for sampling simulators, stored for potential future use).
        """
        try:
            # Use statevector simulator for exact expectation values initially
            self.backend = AerSimulator(method='statevector')
        except qiskit.exceptions.QiskitError as e:
             print(f"Error initializing AerSimulator: {e}")
             print("Ensure qiskit-aer is installed and compatible.")
             raise
        self.n_shots = n_shots
        # print(f"QuantumSimulator initialized with backend: {self.backend.configuration().backend_name}")

    def run_circuit_and_get_statevector(self, circuit: qiskit.QuantumCircuit) -> Statevector:
        """
        Executes the given quantum circuit on the statevector simulator and returns the final statevector.

        Args:
            circuit: The Qiskit QuantumCircuit to execute.

        Returns:
            A Qiskit Statevector object representing the final state.
        """
        # Statevector simulator doesn't need explicit measurement, but saving state is crucial
        circuit_to_run = circuit.copy() # Work on a copy to avoid modifying original
        circuit_to_run.save_statevector()
        result = self.backend.run(circuit_to_run).result()
        statevector = result.get_statevector(circuit_to_run)
        return statevector

    def calculate_expectation_value(self, statevector: Statevector, observable_str: str) -> float:
        """
        Calculates the expectation value <psi|O|psi> for a given statevector and observable.

        Args:
            statevector: The final statevector |psi> of the system.
            observable_str: A string identifier for the observable O (e.g., "Z1Z2", "Z0").

        Returns:
            The calculated expectation value (a real number).

        Raises:
            ValueError: If the observable string is not supported or incompatible with the number of qubits.
        """
        num_qubits = statevector.num_qubits
        observable_op = self._get_qiskit_operator(observable_str, num_qubits)

        # Calculate expectation value using Qiskit's Statevector method
        # <psi|O|psi> = statevector.conjugate().dot(observable_op @ statevector)
        # Qiskit provides a convenience method:
        try:
            expected_value = statevector.expectation_value(observable_op).real
            # Ensure the imaginary part is negligible (should be for Hermitian observables)
            imag_part = statevector.expectation_value(observable_op).imag
            if not np.isclose(imag_part, 0.0):
                 print(f"Warning: Non-negligible imaginary part ({imag_part:.2e}) in expectation value for {observable_str}.")
            return expected_value
        except Exception as e:
            print(f"Error calculating expectation value for {observable_str} on {num_qubits} qubits: {e}")
            raise

    def _get_qiskit_operator(self, observable_str: str, num_qubits: int) -> SparsePauliOp:
        """
        Converts a string identifier into a Qiskit SparsePauliOp object.
        Qiskit's Pauli strings read right-to-left (qubit 0 is the rightmost character).

        Args:
            observable_str: String identifier (e.g., "Z1Z2", "Z0").
            num_qubits: The total number of qubits in the system.

        Returns:
            A SparsePauliOp representing the observable.

        Raises:
            ValueError: For unsupported observable strings or qubit mismatches.
        """
        pauli_list = ['I'] * num_qubits

        if observable_str == "Z1Z2":
             # Measures <Z_0 Z_1> (Pauli Z on qubit 0 and qubit 1)
            if num_qubits < 2:
                raise ValueError(f"Need at least 2 qubits for 'Z1Z2', but got {num_qubits}")
            pauli_list[0] = 'Z'
            pauli_list[1] = 'Z'
        elif observable_str == "Z0":
            # Measures <Z_0> (Pauli Z on qubit 0)
            if num_qubits < 1:
                 raise ValueError(f"Need at least 1 qubit for 'Z0', but got {num_qubits}")
            pauli_list[0] = 'Z'
        # Add more observables here as needed (e.g., "X0", "Y1", "Z2")
        # elif observable_str == "Z2":
        #     if num_qubits < 3:
        #          raise ValueError(f"Need at least 3 qubits for 'Z2', but got {num_qubits}")
        #     pauli_list[2] = 'Z'
        else:
            raise ValueError(f"Unsupported observable string identifier: '{observable_str}'")

        # Convert list to Qiskit's string format (right-to-left)
        qiskit_pauli_str = "".join(pauli_list[::-1])
        return SparsePauliOp(qiskit_pauli_str)