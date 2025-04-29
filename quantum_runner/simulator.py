# quantum_ml_simulation/quantum_runner/simulator.py
# Runs quantum circuits and calculates results using Qiskit Aer
# Now supports both statevector and shot-based simulation

import qiskit
from qiskit_aer import AerSimulator
    # Optional: Import noise models if needed later
# from qiskit_aer.noise import NoiseModel

from qiskit.providers.basic_provider import BasicSimulator # For simple sampling if Aer isn't needed/available
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.result import Counts
import numpy as np
from collections import Counter
from typing import Union, Optional

# Use relative import for config
from ..config import simulation_params as cfg

class QuantumSimulator:
    """Handles the execution of quantum circuits and calculation of expectation values."""

    def __init__(self, simulation_mode: str = cfg.SIMULATION_MODE, n_shots: int = cfg.N_SHOTS):
        """
        Initializes the simulator based on the desired mode.

        Args:
            simulation_mode: 'statevector' or 'shots'.
            n_shots: Number of shots for 'shots' mode.
        """
        self.simulation_mode = simulation_mode
        self.n_shots = n_shots
        # self.noise_model = None # Placeholder for future noise model addition

        if self.simulation_mode == "statevector":
            try:
                # Use Aer's statevector simulator for high performance
                self.backend = AerSimulator(method='statevector')
                print("QuantumSimulator initialized with AerSimulator (statevector mode).")
            except qiskit.exceptions.QiskitError as e:
                print(f"Warning: AerSimulator (statevector) failed ({e}). Falling back to BasicSimulator statevector.")
                self.backend = BasicSimulator(mode='statevector') # Fallback
        elif self.simulation_mode == "shots":
            try:
                 # Use Aer's default simulator (handles shots efficiently)
                self.backend = AerSimulator() # Can add method='automatic' or specific like 'qasm_simulator'
                # Add noise model here if configured:
                # self.backend.options.set(noise_model=self.noise_model)
                print(f"QuantumSimulator initialized with AerSimulator (shots mode, n_shots={self.n_shots}).")
            except qiskit.exceptions.QiskitError as e:
                 print(f"Warning: AerSimulator (shots) failed ({e}). Falling back to BasicSimulator qasm.")
                 self.backend = BasicSimulator(mode='qasm') # Fallback
        else:
            raise ValueError(f"Unsupported simulation_mode: {self.simulation_mode}")

    def run_circuit_and_get_statevector(self, circuit: qiskit.QuantumCircuit) -> Optional[Statevector]:
        """
        Executes the circuit on a statevector simulator and returns the final statevector.
        Returns None if not in statevector mode.
        """
        if self.simulation_mode != "statevector":
            print("Warning: run_circuit_and_get_statevector called but not in statevector mode.")
            return None
        if not isinstance(self.backend, (AerSimulator, BasicSimulator)) or \
           (hasattr(self.backend, 'options') and self.backend.options.get('method') != 'statevector'):
             print("Warning: Backend is not configured for statevector simulation.")
             # Try to run anyway, might fail
             pass


        circuit_to_run = circuit.copy()
        circuit_to_run.save_statevector() # Crucial for statevector method
        try:
            result = self.backend.run(circuit_to_run).result()
            statevector = result.get_statevector(circuit_to_run)
            return statevector
        except Exception as e:
            print(f"Error running statevector simulation: {e}")
            return None

    def run_circuit_and_get_counts(self, circuit: qiskit.QuantumCircuit) -> Optional[Counts]:
         """
         Executes the circuit, performs measurements, and returns the shot counts.
         Requires the circuit to have measurements added.
         Returns None if not in shots mode.
         """
         if self.simulation_mode != "shots":
             print("Warning: run_circuit_and_get_counts called but not in shots mode.")
             return None

         circuit_to_run = circuit.copy()
         # Ensure measurements exist - add if missing? Or rely on simulation class?
         # Assuming simulation class adds measurements if needed for shots mode.
         if not any(isinstance(instr.operation, qiskit.circuit.Measure) for instr in circuit_to_run.data):
              print("Warning: Circuit submitted for shots mode has no measurement instructions. Adding measurement to all qubits.")
              circuit_to_run.measure_all() # Measure all qubits to classical bits

         try:
             job = self.backend.run(circuit_to_run, shots=self.n_shots)
             result = job.result()
             counts = result.get_counts(circuit_to_run)
             # Qiskit Counts is dict-like: {'001': 100, '101': 924, ...}
             # Keys are strings of classical bit outcomes (ordered cN-1 ... c0)
             return Counts(counts) # Return official Counts object
         except Exception as e:
             print(f"Error running shots simulation: {e}")
             return None


    def calculate_expectation_value_statevector(self, statevector: Statevector, observable_str: str) -> float:
        """ Calculates <psi|O|psi> using the statevector. """
        if statevector is None:
             print("Error: Cannot calculate expectation value from None statevector.")
             return np.nan
        num_qubits = statevector.num_qubits
        try:
            observable_op = self._get_qiskit_operator(observable_str, num_qubits)
            expected_value = statevector.expectation_value(observable_op).real
            # Check imaginary part
            imag_part = statevector.expectation_value(observable_op).imag
            if not np.isclose(imag_part, 0.0, atol=1e-8): # Stricter tolerance
                 print(f"Warning: Statevector expectation value had non-negligible imaginary part ({imag_part:.3e}) for {observable_str}.")
            return expected_value
        except Exception as e:
            print(f"Error calculating statevector expectation value for {observable_str} on {num_qubits} qubits: {e}")
            return np.nan


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


    def _get_qiskit_operator(self, observable_str: str, num_qubits: int) -> SparsePauliOp:
        """ Converts string identifier (e.g., "Z1Z2") to SparsePauliOp. """
        # Qiskit Pauli string format: 'qN-1 ... q1 q0'
        pauli_list = ['I'] * num_qubits

        if observable_str == "Z1Z2":
            if num_qubits < 2: raise ValueError(f"'Z1Z2' needs >= 2 qubits, got {num_qubits}")
            pauli_list[0] = 'Z' # Corresponds to qubit 0
            pauli_list[1] = 'Z' # Corresponds to qubit 1
        elif observable_str == "Z0":
            if num_qubits < 1: raise ValueError(f"'Z0' needs >= 1 qubit, got {num_qubits}")
            pauli_list[0] = 'Z' # Corresponds to qubit 0
        # Add more observables if needed, e.g., 'Z_avg' for <1/N Sum Zi>
        # elif observable_str == "Z_avg":
        #     ops = []
        #     for i in range(num_qubits):
        #          p_list = ['I'] * num_qubits
        #          p_list[i] = 'Z'
        #          ops.append(SparsePauliOp("".join(p_list[::-1]), coeffs=[1.0/num_qubits]))
        #     return sum(ops) if ops else SparsePauliOp('I'*num_qubits, coeffs=[0.0])

        else:
            raise ValueError(f"Unsupported observable string identifier: '{observable_str}'")

        qiskit_pauli_str = "".join(pauli_list[::-1])
        return SparsePauliOp(qiskit_pauli_str)