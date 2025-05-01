# quantum_ml_simulation/simulations/base_simulation.py
# Abstract base class for quantum simulations (Refactored Init)

from abc import ABC, abstractmethod
import os
import qiskit
import time
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Optional # For type hinting

# Use relative imports within the package
from ..quantum_runner import simulator as qs
from ..config import simulation_params as cfg # Keep for flags/defaults

class BaseSimulation(ABC):
    """Abstract base class for defining and running quantum simulations."""

    # --- MODIFIED __init__ ---
    def __init__(self,
                 simulation_name: str,
                 n_qubits: int,
                 measurement_operator: str,
                 delta_t: float,
                 n_steps_range: List[int],
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Optional[Dict[str, List]] = None): # Added param_ranges
        """
        Initializes the base simulation with specific parameters.

        Args:
            simulation_name: Name of the simulation (e.g., "IsingModel").
            n_qubits: Number of qubits for the simulation.
            measurement_operator: String identifier for the observable to measure.
            delta_t: Time step duration.
            n_steps_range: List or range of the number of time steps (n) to simulate.
            initial_state_type: 'zero' or 'superposition'.
            simulation_mode: 'statevector' or 'shots'.
            param_ranges: Dictionary containing ranges for simulation-specific parameters
                          (e.g., {'J': [0.1, ...], 'B': [0.0, ...]} for Ising).
        """
        self.simulation_name = simulation_name
        self.n_qubits = n_qubits
        self.measurement_operator = measurement_operator
        self.delta_t = delta_t
        self.n_steps_range = list(n_steps_range) # Ensure it's a list
        self.initial_state_type = initial_state_type
        self.simulation_mode = simulation_mode
        self.param_ranges = param_ranges if param_ranges is not None else {}

        # Instantiate the quantum runner based on passed mode
        self.q_simulator = qs.QuantumSimulator(
            simulation_mode=self.simulation_mode,
            n_shots=cfg.N_SHOTS # Can keep N_SHOTS from global config or pass it too
        )

        # Flags from config (can still be used to enable/disable features)
        # Note: Pass n_qubits to the check
        self.validate_trotter = cfg.VALIDATE_TROTTER and self.n_qubits <= cfg.VALIDATION_MAX_QUBITS
        self.compute_entanglement = cfg.COMPUTE_ENTANGLEMENT
        # Ensure partition is valid for current n_qubits
        self.entanglement_partition = min(cfg.ENTANGLEMENT_PARTITION, self.n_qubits // 2)

        print(f"Initialized BaseSimulation: {self.simulation_name}")
        print(f"  Mode: {self.simulation_mode}, Qubits: {self.n_qubits}, Measure: {self.measurement_operator}")
        print(f"  dt: {self.delta_t}, n_steps: {self.n_steps_range[0]}-{self.n_steps_range[-1]}, Initial State: {self.initial_state_type}")
        if self.validate_trotter:
             print(f"  Trotter validation ENABLED (max qubits: {cfg.VALIDATION_MAX_QUBITS}, Tolerance: {cfg.VALIDATION_TOLERANCE})")
        else:
             print(f"  Trotter validation DISABLED (N > {cfg.VALIDATION_MAX_QUBITS} or config flag False)")
        if self.compute_entanglement:
             print(f"  Entanglement computation ENABLED (Partition: {self.entanglement_partition})")
        else:
             print(f"  Entanglement computation DISABLED")
        # Optionally print parameter ranges passed
        # if self.param_ranges:
        #      print(f"  Parameter Ranges: {self.param_ranges}")


    # --- Abstract Methods ---
    @abstractmethod
    def get_parameter_space(self) -> list[tuple]:
        """Generates the list of all parameter combinations to simulate based on initialized ranges."""
        pass

    @abstractmethod
    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """Builds the Qiskit QuantumCircuit for a given set of simulation parameters."""
        pass

    @abstractmethod
    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """Converts the simulation parameters tuple into a flat list for ML input."""
        pass

    @abstractmethod
    def get_ml_input_feature_names(self) -> list[str]:
        """Returns the names of the ML input features in the order matching _format_params_for_ml."""
        pass

    @abstractmethod
    def get_hamiltonian_operator(self, params: tuple) -> SparsePauliOp:
        """Constructs the Hamiltonian for the given parameters as a SparsePauliOp."""
        pass

    # --- Core Simulation Logic (run_single_simulation, _validate_trotter_step, compute_entanglement_entropy) ---
    # These methods remain largely the same, but use self.delta_t, self.n_qubits etc. which are now set during __init__

    def run_single_simulation(self, params: tuple) -> dict:
        """
        Runs one simulation instance, calculates expectation value,
        optionally validates Trotter error and computes entanglement.
        (Implementation is unchanged, relies on self attributes set in __init__)
        """
        start_time = time.time()
        results = {
            'params': params,
            'ml_input': self._format_params_for_ml(params),
            'expectation_value': np.nan,
            'validation_diff': np.nan,
            'entanglement_entropy': np.nan
        }

        try:
            circuit = self.build_circuit_for_params(params)
            final_statevector = None
            counts = None

            if self.simulation_mode == "statevector":
                final_statevector = self.q_simulator.run_circuit_and_get_statevector(circuit)
                if final_statevector:
                     results['expectation_value'] = self.q_simulator.calculate_expectation_value_statevector(
                         final_statevector, self.measurement_operator
                     )
            elif self.simulation_mode == "shots":
                counts = self.q_simulator.run_circuit_and_get_counts(circuit)
                if counts:
                    results['expectation_value'] = self.q_simulator.calculate_expectation_value_counts(
                        counts, self.measurement_operator, self.n_qubits
                    )
            else:
                 raise RuntimeError(f"Unknown simulation mode: {self.simulation_mode}")

            if self.validate_trotter and self.simulation_mode == "statevector" and final_statevector:
                validation_diff = self._validate_trotter_step(params, final_statevector)
                results['validation_diff'] = validation_diff
                if abs(validation_diff) > cfg.VALIDATION_TOLERANCE:
                     # Reduce verbosity, maybe log this instead?
                     # print(f"  WARNING: Trotter validation failed for {params}. Diff: {validation_diff:.4f} > {cfg.VALIDATION_TOLERANCE}")
                     pass

            if self.compute_entanglement and self.simulation_mode == "statevector" and final_statevector:
                 results['entanglement_entropy'] = self.compute_entanglement_entropy(
                     final_statevector, list(range(self.entanglement_partition)) # Use self.entanglement_partition
                 )

            # end_time = time.time() # No need to store end time here

        except Exception as e:
            print(f"\nERROR during single simulation run for params {params}: {e}")
            # import traceback; traceback.print_exc() # Uncomment for debugging

        return results

    def _validate_trotter_step(self, params: tuple, trotter_statevector: Statevector) -> float:
        """Compares Trotter result to exact diagonalization. (Unchanged)"""
        try:
            H_op = self.get_hamiltonian_operator(params)
            H_matrix = H_op.to_matrix(sparse=False) # Ensure dense for expm

            initial_circuit = qiskit.QuantumCircuit(self.n_qubits)
            if self.initial_state_type == 'superposition':
                initial_circuit.h(range(self.n_qubits))
            psi_initial = Statevector(initial_circuit)

            n_steps = params[0]
            total_time = n_steps * self.delta_t
            U_exact = expm(-1j * H_matrix * total_time)

            # Ensure statevector data is compatible for matmul
            psi_exact_data = psi_initial.data @ U_exact.T
            psi_exact = Statevector(psi_exact_data)


            exact_exp_val = self.q_simulator.calculate_expectation_value_statevector(
                psi_exact, self.measurement_operator
            )
            trotter_exp_val = self.q_simulator.calculate_expectation_value_statevector(
                trotter_statevector, self.measurement_operator
            )

            difference = trotter_exp_val - exact_exp_val
            return float(np.real(difference)) # Ensure result is real float

        except Exception as e:
            print(f"ERROR during Trotter validation for params {params}: {e}")
            # import traceback; traceback.print_exc() # Uncomment for debugging
            return np.nan


    def compute_entanglement_entropy(self, statevector: Statevector, partition: list[int]) -> float:
         """Computes the entanglement entropy. (Unchanged)"""
         try:
             rho_subsystem = partial_trace(statevector, list(range(partition[0], self.n_qubits))) # Qiskit traces out qubits NOT specified
             ent_entropy = entropy(rho_subsystem, base=np.exp(1))
             return float(ent_entropy)
         except Exception as e:
              print(f"ERROR computing entanglement entropy: {e}")
              return np.nan


    # --- generate_dataset method remains the same ---
    def generate_dataset(self) -> list[dict]:
        """
        Generates the full dataset by running simulations for all parameter combinations.
        (Implementation is unchanged, calls methods using self attributes)
        """
        dataset = []
        parameter_space = self.get_parameter_space()
        if not parameter_space:
            print("Warning: Parameter space is empty. No data generated.")
            return []

        print(f"Generating dataset for {self.simulation_name} ({self.simulation_mode} mode)")
        print(f"  Total parameter sets: {len(parameter_space)}")

        # Use tqdm for progress bar
        # Disable progress bar if too verbose or for CI
        disable_tqdm = os.getenv("CI", "false").lower() == "true"
        for params in tqdm(parameter_space, desc=f"Simulating {self.simulation_name}", disable=disable_tqdm):
            simulation_results = self.run_single_simulation(params)
            data_point = {
                'input': simulation_results['ml_input'],
                'output': simulation_results['expectation_value'],
                'validation_diff': simulation_results['validation_diff'],
                'entanglement': simulation_results['entanglement_entropy']
            }
            # Optional: Include raw params if needed later (already in results['params'])
            # data_point['raw_params'] = simulation_results['params']
            dataset.append(data_point)

        # Report stats after generation
        num_generated = len(dataset)
        num_valid_exp_val = sum(1 for d in dataset if not np.isnan(d.get('output', np.nan))) # Safer get
        print(f"\nDataset generation complete for {self.simulation_name}.")
        print(f"  Total data points generated: {num_generated}")
        print(f"  Points with valid expectation value: {num_valid_exp_val}")
        if num_generated > num_valid_exp_val:
             failed_count = num_generated - num_valid_exp_val
             print(f"  Points failed (NaN expectation value): {failed_count} ({failed_count/num_generated:.1%} failure rate)")

        if self.validate_trotter:
             valid_diffs = [d['validation_diff'] for d in dataset if not np.isnan(d.get('validation_diff', np.nan))]
             if valid_diffs:
                  avg_abs_diff = np.mean(np.abs(valid_diffs))
                  max_abs_diff = np.max(np.abs(valid_diffs))
                  print(f"  Avg/Max validation diff (where computed): {avg_abs_diff:.4e} / {max_abs_diff:.4e}")
             else:
                  print("  No valid Trotter validation differences computed.")

        if self.compute_entanglement:
            valid_ents = [d['entanglement'] for d in dataset if not np.isnan(d.get('entanglement', np.nan))]
            if valid_ents:
                 avg_ent = np.mean(valid_ents)
                 max_ent = np.max(valid_ents)
                 print(f"  Avg/Max entanglement entropy (where computed): {avg_ent:.4f} / {max_ent:.4f}")
            else:
                 print("  No valid entanglement entropy values computed.")

        return dataset