# quantum_ml_simulation/simulations/base_simulation.py
# Abstract base class for quantum simulations (Enhanced)

from abc import ABC, abstractmethod
import qiskit
import time
import numpy as np
from scipy.linalg import expm # For exact diagonalization validation
from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from tqdm.auto import tqdm

# Use relative imports within the package
# from ..quantum_runner import circuit_builder as cb # Maybe less needed now
from ..quantum_runner import simulator as qs
from ..config import simulation_params as cfg

class BaseSimulation(ABC):
    """Abstract base class for defining and running quantum simulations."""

    def __init__(self, simulation_name: str, n_qubits: int, measurement_operator: str):
        """Initializes the base simulation."""
        self.simulation_name = simulation_name
        self.n_qubits = n_qubits
        self.measurement_operator = measurement_operator

        # Instantiate the quantum runner based on config mode
        self.q_simulator = qs.QuantumSimulator(
            simulation_mode=cfg.SIMULATION_MODE,
            n_shots=cfg.N_SHOTS
        )
        self.delta_t = cfg.DELTA_T
        self.n_steps_range = cfg.N_STEPS_RANGE
        self.initial_state_type = cfg.INITIAL_STATE_TYPE

        # Flags from config
        self.validate_trotter = cfg.VALIDATE_TROTTER and self.n_qubits <= cfg.VALIDATION_MAX_QUBITS
        self.compute_entanglement = cfg.COMPUTE_ENTANGLEMENT

        print(f"Initialized BaseSimulation: {self.simulation_name}")
        print(f"  Mode: {self.q_simulator.simulation_mode}, Qubits: {self.n_qubits}, Measure: {self.measurement_operator}")
        if self.validate_trotter:
             print(f"  Trotter validation ENABLED (max qubits: {cfg.VALIDATION_MAX_QUBITS})")
        if self.compute_entanglement:
             print(f"  Entanglement computation ENABLED (partition: {cfg.ENTANGLEMENT_PARTITION})")

    # --- Abstract Methods ---
    @abstractmethod
    def get_parameter_space(self) -> list[tuple]:
        """Generates the list of all parameter combinations to simulate."""
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
        """Returns the names of the ML input features."""
        pass

    @abstractmethod
    def get_hamiltonian_operator(self, params: tuple) -> SparsePauliOp:
        """
        Constructs the Hamiltonian for the given parameters as a SparsePauliOp.
        Needed for exact diagonalization validation.

        Args:
            params: The simulation parameters tuple.

        Returns:
            A Qiskit SparsePauliOp representing the system's Hamiltonian.
        """
        pass

    # --- Core Simulation Logic ---
    def run_single_simulation(self, params: tuple) -> dict:
        """
        Runs one simulation instance, calculates expectation value,
        optionally validates Trotter error and computes entanglement.

        Args:
            params: The parameter tuple for this run.

        Returns:
            A dictionary containing results:
            {
                'params': params,
                'ml_input': list[float],
                'expectation_value': float, # or NaN on failure
                'validation_diff': float, # or NaN if not validated
                'entanglement_entropy': float # or NaN if not computed
            }
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
            # 1. Build Circuit
            circuit = self.build_circuit_for_params(params)
            final_statevector = None
            counts = None

            # 2. Execute Circuit (Statevector or Shots)
            if self.q_simulator.simulation_mode == "statevector":
                final_statevector = self.q_simulator.run_circuit_and_get_statevector(circuit)
                if final_statevector:
                     results['expectation_value'] = self.q_simulator.calculate_expectation_value_statevector(
                         final_statevector, self.measurement_operator
                     )
            elif self.q_simulator.simulation_mode == "shots":
                # Measurement should be added by build_circuit_for_params if needed,
                # but simulator adds measure_all as fallback.
                counts = self.q_simulator.run_circuit_and_get_counts(circuit)
                if counts:
                    results['expectation_value'] = self.q_simulator.calculate_expectation_value_counts(
                        counts, self.measurement_operator, self.n_qubits
                    )
            else:
                 raise RuntimeError(f"Unknown simulation mode: {self.q_simulator.simulation_mode}")


            # 3. Optional: Validate Trotter Error (only in statevector mode and for small N)
            if self.validate_trotter and self.q_simulator.simulation_mode == "statevector" and final_statevector:
                validation_diff = self._validate_trotter_step(params, final_statevector)
                results['validation_diff'] = validation_diff
                if abs(validation_diff) > cfg.VALIDATION_TOLERANCE:
                     print(f"  WARNING: Trotter validation failed for {params}. Diff: {validation_diff:.4f} > {cfg.VALIDATION_TOLERANCE}")


            # 4. Optional: Compute Entanglement (only in statevector mode)
            if self.compute_entanglement and self.q_simulator.simulation_mode == "statevector" and final_statevector:
                 results['entanglement_entropy'] = self.compute_entanglement_entropy(
                     final_statevector, list(range(cfg.ENTANGLEMENT_PARTITION))
                 )

            end_time = time.time()
            # Optional print: print(f"Sim time: {end_time-start_time:.3f}s")

        except Exception as e:
            print(f"\nERROR during single simulation run for params {params}: {e}")
            import traceback
            # traceback.print_exc() # Uncomment for full traceback if needed
            # Results dict already has NaNs initialized

        return results

    def _validate_trotter_step(self, params: tuple, trotter_statevector: Statevector) -> float:
        """Compares Trotter result to exact diagonalization."""
        try:
            # 1. Get Hamiltonian Matrix
            H_op = self.get_hamiltonian_operator(params)
            H_matrix = H_op.to_matrix()

            # 2. Get Initial State Vector
            initial_circuit = qiskit.QuantumCircuit(self.n_qubits)
            if self.initial_state_type == 'superposition':
                initial_circuit.h(range(self.n_qubits))
            # Else: starts in |0...0> by default
            psi_initial = Statevector(initial_circuit)

            # 3. Calculate Exact Evolution Operator
            n_steps = params[0] # Assuming n_steps is always the first parameter
            total_time = n_steps * self.delta_t
            # Use scipy's expm for matrix exponential: U = exp(-iHt)
            U_exact = expm(-1j * H_matrix * total_time)

            # 4. Calculate Exact Final State
            psi_exact = Statevector(psi_initial.data @ U_exact.T) # Apply U to |psi>

            # 5. Calculate Exact Expectation Value
            exact_exp_val = self.q_simulator.calculate_expectation_value_statevector(
                psi_exact, self.measurement_operator
            )

            # 6. Get Trotter Expectation Value
            trotter_exp_val = self.q_simulator.calculate_expectation_value_statevector(
                trotter_statevector, self.measurement_operator
            )

            # 7. Calculate Difference
            difference = trotter_exp_val - exact_exp_val
            return float(difference)

        except Exception as e:
            print(f"ERROR during Trotter validation for params {params}: {e}")
            return np.nan


    def compute_entanglement_entropy(self, statevector: Statevector, partition: list[int]) -> float:
         """Computes the entanglement entropy for a given partition."""
         try:
             # Calculate density matrix after tracing out the specified partition
             rho_subsystem = partial_trace(statevector, partition)
             # Compute Von Neumann entropy (base e)
             ent_entropy = entropy(rho_subsystem, base=np.exp(1))
             return float(ent_entropy)
         except Exception as e:
              print(f"ERROR computing entanglement entropy: {e}")
              return np.nan


    def generate_dataset(self) -> list[dict]:
        """
        Generates the full dataset by running simulations for all parameter combinations.
        Returns a list of dictionaries, each containing results from run_single_simulation.
        """
        dataset = []
        parameter_space = self.get_parameter_space()
        if not parameter_space:
            print("Warning: Parameter space is empty. No data generated.")
            return []

        print(f"\nGenerating dataset for {self.simulation_name} ({self.q_simulator.simulation_mode} mode)")
        print(f"  Total parameter sets: {len(parameter_space)}")

        # Use tqdm for progress bar
        for params in tqdm(parameter_space, desc=f"Simulating {self.simulation_name}"):
            simulation_results = self.run_single_simulation(params)
            # Format for ML (input) and store raw results
            data_point = {
                 # 'raw_params': simulation_results['params'], # Redundant? Already in results
                 'input': simulation_results['ml_input'],
                 'output': simulation_results['expectation_value'], # Primary output for ML
                 'validation_diff': simulation_results['validation_diff'],
                 'entanglement': simulation_results['entanglement_entropy']
            }
            dataset.append(data_point)

        # Report stats after generation
        num_generated = len(dataset)
        num_valid_exp_val = sum(1 for d in dataset if not np.isnan(d['output']))
        print(f"\nDataset generation complete for {self.simulation_name}.")
        print(f"  Total data points generated: {num_generated}")
        print(f"  Points with valid expectation value: {num_valid_exp_val}")
        if num_generated > num_valid_exp_val:
             print(f"  Points failed (NaN expectation value): {num_generated - num_valid_exp_val}")

        if self.validate_trotter:
             valid_diffs = [d['validation_diff'] for d in dataset if not np.isnan(d['validation_diff'])]
             if valid_diffs:
                  avg_abs_diff = np.mean(np.abs(valid_diffs))
                  max_abs_diff = np.max(np.abs(valid_diffs))
                  print(f"  Avg/Max validation diff (where computed): {avg_abs_diff:.4e} / {max_abs_diff:.4e}")

        if self.compute_entanglement:
            valid_ents = [d['entanglement'] for d in dataset if not np.isnan(d['entanglement'])]
            if valid_ents:
                 avg_ent = np.mean(valid_ents)
                 max_ent = np.max(valid_ents)
                 print(f"  Avg/Max entanglement entropy (where computed): {avg_ent:.4f} / {max_ent:.4f}")

        return dataset