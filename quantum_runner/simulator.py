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
import re

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

    # Inside class QuantumSimulator:

    def _get_qiskit_operator(self, observable_str: str, num_qubits: int) -> SparsePauliOp:
        """
        Converts string identifier (e.g., "Z0", "X1", "ZZ01", "XYZ012") to SparsePauliOp.
        Format: PauliLetters followed by qubit indices. 'I' is ignored in PauliLetters.
        """
        op_key = (observable_str, num_qubits)
        if op_key in self._operator_cache:
            return self._operator_cache[op_key]

        # Regex to capture Pauli letters (X, Y, Z) and indices (digits)
        match = re.match(r"([XYZ]+)(\d+)$", observable_str)
        if not match:
             # Handle single qubit case like "Z0" separately if regex fails?
             # Or handle Identity separately? Let's check for Identity first.
             if observable_str == 'I': # Allow just "I" for identity
                 return SparsePauliOp('I'*num_qubits)
             # Check single Pauli + single digit, e.g., Z0, X1
             match_single = re.match(r"([XYZ])(\d+)$", observable_str)
             if match_single:
                  pauli_char = match_single.group(1)
                  indices_str = match_single.group(2)
                  if len(indices_str) == 1: # Ensure only one index for single Pauli
                       pauli_chars = pauli_char
                  else: # Treat Z01 as Z on 0 and Z on 1
                       pauli_chars = pauli_char * len(indices_str)
             else:
                  raise ValueError(f"Invalid observable string format: '{observable_str}'. Expected format like 'Z0', 'XX12', 'XYZ012'.")
        else:
             pauli_chars = match.group(1)
             indices_str = match.group(2)

        if len(pauli_chars) != len(indices_str):
            raise ValueError(f"Mismatch between number of Paulis ({len(pauli_chars)}) and indices ({len(indices_str)}) in '{observable_str}'. Use format like ZZ01 or XYZ012.")

        pauli_list = ['I'] * num_qubits
        parsed_indices = set()

        for i in range(len(pauli_chars)):
            pauli_char = pauli_chars[i]
            try:
                q_idx = int(indices_str[i])
            except ValueError:
                raise ValueError(f"Invalid index character in '{observable_str}'")

            if not (0 <= q_idx < num_qubits):
                raise ValueError(f"Qubit index {q_idx} out of bounds for {num_qubits} qubits in '{observable_str}'")
            if q_idx in parsed_indices:
                raise ValueError(f"Duplicate qubit index {q_idx} specified in '{observable_str}'")

            pauli_list[q_idx] = pauli_char
            parsed_indices.add(q_idx)

        # Qiskit Pauli string format: 'qN-1 ... q1 q0'
        qiskit_pauli_str = "".join(pauli_list[::-1])
        operator = SparsePauliOp(qiskit_pauli_str)
        self._operator_cache[op_key] = operator # Cache the result
        return operator


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


