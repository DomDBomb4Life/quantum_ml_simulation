# quantum_ml_simulation/quantum_runner/simulator.py
# Runs quantum circuits and calculates results (Refactored for Vector Output)

import qiskit
from qiskit_aer import AerSimulator
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.result import Counts
import numpy as np
from collections import Counter, defaultdict
from typing import Union, Optional, List, Dict, Tuple

# Use relative import for config
from ..config import simulation_params as cfg # Keep for backend settings like N_SHOTS

class QuantumSimulator:
    """
    Handles the execution of quantum circuits and calculation of expectation values.
    Now supports calculating vectors of expectation values.
    """

    def __init__(self, simulation_mode: str = cfg.SIMULATION_MODE, n_shots: int = cfg.N_SHOTS):
        """Initializes the simulator based on the desired mode."""
        self.simulation_mode = simulation_mode
        self.n_shots = n_shots
        self._operator_cache: Dict[Tuple[str, int], SparsePauliOp] = {} # Cache for SparsePauliOp

        if self.simulation_mode == "statevector":
            try:
                self.backend = AerSimulator(method='statevector')
                print("QuantumSimulator initialized with AerSimulator (statevector mode).")
            except qiskit.exceptions.QiskitError:
                print("Warning: AerSimulator (statevector) failed. Falling back to BasicSimulator.")
                self.backend = BasicSimulator(mode='statevector')
        elif self.simulation_mode == "shots":
            # ... (shots mode initialization remains the same) ...
            try:
                self.backend = AerSimulator()
                print(f"QuantumSimulator initialized with AerSimulator (shots mode, n_shots={self.n_shots}).")
            except qiskit.exceptions.QiskitError:
                 print(f"Warning: AerSimulator (shots) failed. Falling back to BasicSimulator qasm.")
                 self.backend = BasicSimulator(mode='qasm')
        else:
            raise ValueError(f"Unsupported simulation_mode: {self.simulation_mode}")

    def run_circuit_and_get_statevector(self, circuit: qiskit.QuantumCircuit) -> Optional[Statevector]:
        """Executes the circuit and returns the final statevector (statevector mode only)."""
        if self.simulation_mode != "statevector":
            print("Warning: run_circuit_and_get_statevector called but not in statevector mode.")
            return None
        # No need to check backend type again, assuming constructor worked

        # Create a temporary circuit *just* for saving the statevector
        # Avoids modifying the original circuit which might be evolved further
        temp_circuit = circuit.copy()
        temp_circuit.save_statevector()
        try:
            result = self.backend.run(temp_circuit).result()
            statevector = result.get_statevector(temp_circuit)
            return statevector
        except Exception as e:
            print(f"Error running statevector simulation: {e}")
            return None

    # --- NEW: Calculate Vector of Expectation Values ---
    def calculate_expectation_vector_statevector(self,
                                               statevector: Statevector,
                                               observable_strings: List[str]) -> List[float]:
        """
        Calculates <psi|O|psi> for a list of observables using the statevector.

        Args:
            statevector: The Statevector object.
            observable_strings: A list of string identifiers for the observables (e.g., ["Z0", "Z1Z2"]).

        Returns:
            A list of corresponding expectation values (floats). Returns NaN for failed calculations.
        """
        if statevector is None:
            print("Error: Cannot calculate expectation vector from None statevector.")
            return [np.nan] * len(observable_strings)

        num_qubits = statevector.num_qubits
        results = []
        for obs_str in observable_strings:
            try:
                # Use cached operator if available, otherwise create and cache
                op_key = (obs_str, num_qubits)
                if op_key not in self._operator_cache:
                    self._operator_cache[op_key] = self._get_qiskit_operator(obs_str, num_qubits)
                observable_op = self._operator_cache[op_key]

                expected_value = statevector.expectation_value(observable_op)
                # Check imaginary part for Hermitian operators
                if not np.isclose(expected_value.imag, 0.0, atol=1e-8):
                    print(f"Warning: Statevector expectation value had non-negligible imaginary part ({expected_value.imag:.3e}) for {obs_str}.")
                results.append(expected_value.real)
            except Exception as e:
                print(f"Error calculating statevector expectation value for {obs_str} on {num_qubits} qubits: {e}")
                results.append(np.nan)
        return results

    def _get_qiskit_operator(self, observable_str: str, num_qubits: int) -> SparsePauliOp:
        """
        Converts string identifier (e.g., "Z0", "X1", "Z0Z1") to SparsePauliOp.
        Handles single-qubit and specified two-qubit operators.
        """
        # Cache check is done in the calling function now
        pauli_list = ['I'] * num_qubits
        op_char = observable_str[0] # e.g., 'Z', 'X', 'Y'
        qubit_indices_str = observable_str[1:] # e.g., '0', '1', '2', '01', '12'

        if not op_char in ('I', 'X', 'Y', 'Z'):
            raise ValueError(f"Unsupported Pauli operator type '{op_char}' in '{observable_str}'")

        try:
            # Handle single qubit ops like Z0, X1, Y2
            if len(qubit_indices_str) == 1:
                q_idx = int(qubit_indices_str)
                if not (0 <= q_idx < num_qubits):
                    raise ValueError(f"Qubit index {q_idx} out of bounds for {num_qubits} qubits in '{observable_str}'")
                pauli_list[q_idx] = op_char
            # Handle two qubit ops like Z0Z1, X1X2 (assuming same Pauli type for now)
            elif len(qubit_indices_str) == 2:
                q_idx1 = int(qubit_indices_str[0])
                q_idx2 = int(qubit_indices_str[1])
                if not (0 <= q_idx1 < num_qubits and 0 <= q_idx2 < num_qubits and q_idx1 != q_idx2):
                     raise ValueError(f"Invalid qubit indices {q_idx1}, {q_idx2} for {num_qubits} qubits in '{observable_str}'")
                # Assume ZZ, XX, YY structure for now
                pauli_list[q_idx1] = op_char
                pauli_list[q_idx2] = op_char
                # Add more complex parsing here if needed (e.g., "X0Y1") later
            elif len(qubit_indices_str) > 2:
                 # Example: Z0Z1Z2 - Currently just takes the first char as Pauli type
                 indices = [int(idx) for idx in qubit_indices_str]
                 if any(not (0 <= i < num_qubits) for i in indices):
                      raise ValueError(f"Invalid qubit indices in '{observable_str}'")
                 if len(set(indices)) != len(indices):
                      raise ValueError(f"Duplicate qubit indices in '{observable_str}'")
                 for q_idx in indices:
                     pauli_list[q_idx] = op_char
            else: # Just "I", "X", "Y", "Z"? Apply to qubit 0 by convention? Or reject? Let's reject for now.
                raise ValueError(f"Invalid observable string format: '{observable_str}'. Needs qubit index(es).")

        except ValueError as e:
             raise ValueError(f"Cannot parse observable string '{observable_str}': {e}")

        # Qiskit Pauli string format: 'qN-1 ... q1 q0'
        qiskit_pauli_str = "".join(pauli_list[::-1])
        return SparsePauliOp(qiskit_pauli_str)


    def calculate_expectation_value_counts(self, counts: Counts, observable_str: str, num_qubits: int) -> float:
         """ Calculates <O> from shot counts. """
         if counts is None:
             print("Error: Cannot calculate expectation value from None counts.")
             return np.nan
         if sum(counts.values()) == 0:
             print("Warning: Cannot calculate expectation value from zero total shots.")
             return np.nan

         total_shots = sum(counts.values())
         expected_value = 0.0

         # Get the Pauli operator definition
         try:
             observable_op = self._get_qiskit_operator(observable_str, num_qubits)
         except Exception as e:
              print(f"Error getting operator for expectation from counts: {e}")
              return np.nan

         # --- Efficient Calculation for Pauli Strings ---
         # Assumes observable_op is a *single* Pauli string (like 'ZI', 'XZ', 'ZZI')
         # For sums of Paulis, would need to average results for each term.
         # Current implementation handles single terms like Z0, Z1Z2 correctly.
         if len(observable_op) > 1:
              print(f"Warning: Expectation from counts currently assumes a single Pauli string in the operator. Found {len(observable_op)} terms for {observable_str}. Using first term.")
              # In future, handle sums: iterate observable_op.items() -> (Pauli, coeff)
              pass # Proceed using the logic for the first (and likely only) term

         pauli_str = observable_op.paulis[0].to_label() # Get 'IZZ', 'ZIZ', etc. (Qiskit order: qN-1...q0)

         for bitstring_qiskit_order, count in counts.items():
             # bitstring_qiskit_order is 'qN-1...q0'
             # pauli_str is also 'qN-1...q0'

             eigenvalue = 1.0
             for i in range(num_qubits):
                 pauli_char = pauli_str[i]
                 bit_char = bitstring_qiskit_order[i]
                 # qubit index = num_qubits - 1 - i

                 if pauli_char == 'I':
                     continue # Identity doesn't change eigenvalue
                 elif pauli_char in ('Z', 'X', 'Y'):
                     bit_value = int(bit_char)
                     if pauli_char == 'Z':
                         # Z |0> = +1|0>, Z |1> = -1|1>
                         if bit_value == 1: eigenvalue *= -1
                     elif pauli_char == 'X':
                         # X measurement needs basis change simulation or different calculation
                         # For expectation: <X> = Prob(0) - Prob(1) in X basis
                         # This simple loop only works for diagonal operators (like Z, ZZ) in Z-basis measurements
                         print(f"ERROR: Expectation value calculation from Z-basis counts for non-diagonal Pauli '{pauli_char}' in {observable_str} is not implemented correctly here.")
                         return np.nan # Needs more sophisticated handling
                     elif pauli_char == 'Y':
                          print(f"ERROR: Expectation value calculation from Z-basis counts for non-diagonal Pauli '{pauli_char}' in {observable_str} is not implemented correctly here.")
                          return np.nan
                 else:
                     print(f"ERROR: Unexpected character '{pauli_char}' in Pauli string {pauli_str}")
                     return np.nan

             expected_value += eigenvalue * count

         return expected_value / total_shots


