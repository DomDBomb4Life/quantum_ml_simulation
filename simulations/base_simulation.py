# quantum_ml_simulation/simulations/base_simulation.py
# Abstract base class for quantum simulations (Refactored for Time Series)

from abc import ABC, abstractmethod
import os
import qiskit
import time
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from tqdm.auto import tqdm # Use auto for better notebook/terminal detection
from typing import List, Tuple, Dict, Optional, Iterator, Any

# Use relative imports within the package
from ..quantum_runner import simulator as qs
from ..config import simulation_params as cfg # Keep for flags/defaults

class BaseSimulation(ABC):
    """
    Abstract base class for defining and running quantum simulations
    that generate time-series data of expectation value vectors.
    """

    def __init__(self,
                 simulation_name: str,
                 n_qubits: int,
                 # measurement_operator: str, # No longer needed? Output is defined by observables_to_track
                 delta_t: float,
                 n_steps_range: List[int], # This now defines the time points to record
                 initial_state_type: str,
                 simulation_mode: str,
                 param_ranges: Optional[Dict[str, List]] = None):
        """
        Initializes the base simulation for time-series generation.

        Args:
            simulation_name: Name of the simulation (e.g., "IsingModel").
            n_qubits: Number of qubits for the simulation.
            # measurement_operator: (Removed) Main output is vector, single observable can be tracked via get_observables_to_track.
            delta_t: Time step duration for Trotter evolution.
            n_steps_range: List of total step numbers (n) at which to record data (e.g., [1, 2, ..., max_n]).
            initial_state_type: 'zero' or 'superposition'.
            simulation_mode: 'statevector' or 'shots'.
            param_ranges: Dictionary containing ranges for simulation-specific parameters
                          (e.g., {'J': [0.1, ...], 'B': [0.0, ...]} for Ising).
        """
        self.simulation_name = simulation_name
        self.n_qubits = n_qubits
        self.delta_t = delta_t
        # Ensure n_steps_range starts from 1 and is sorted
        self.n_steps_record_points = sorted([n for n in n_steps_range if n > 0])
        if not self.n_steps_record_points:
             raise ValueError("n_steps_range must contain positive step numbers.")
        self.max_n_steps = self.n_steps_record_points[-1] # Max steps needed for simulation
        self.initial_state_type = initial_state_type
        self.simulation_mode = simulation_mode
        self.param_ranges = param_ranges if param_ranges is not None else {}

        # Instantiate the quantum runner
        self.q_simulator = qs.QuantumSimulator(
            simulation_mode=self.simulation_mode,
            n_shots=cfg.N_SHOTS
        )

        # Advanced Feature Flags
        self.validate_trotter = cfg.VALIDATE_TROTTER and self.n_qubits <= cfg.VALIDATION_MAX_QUBITS
        self.compute_entanglement = cfg.COMPUTE_ENTANGLEMENT
        self.entanglement_partition = list(range(min(cfg.ENTANGLEMENT_PARTITION, self.n_qubits // 2))) # Ensure list for qiskit

        # Store tracked observable names (including formatted names for ML output)
        self._observables_to_track = self.get_observables_to_track()
        self._output_observable_names = [f"exp_{obs}" for obs in self._observables_to_track]

        print(f"Initialized BaseSimulation: {self.simulation_name} (Time-Series Mode)")
        print(f"  Mode: {self.simulation_mode}, Qubits: {self.n_qubits}")
        print(f"  dt: {self.delta_t}, Recording steps: {self.n_steps_record_points[0]} to {self.max_n_steps}, Initial State: {self.initial_state_type}")
        print(f"  Tracked Observables: {self._observables_to_track}")
        print(f"  ML Output Columns: {self._output_observable_names}")
        # ... (print flags for validation/entanglement as before) ...
        if self.validate_trotter: print(f"  Trotter validation ENABLED (Tolerance: {cfg.VALIDATION_TOLERANCE})")
        else: print(f"  Trotter validation DISABLED")
        if self.compute_entanglement: print(f"  Entanglement computation ENABLED (Partition: {self.entanglement_partition})")
        else: print(f"  Entanglement computation DISABLED")


    # --- Abstract Methods ---
    @abstractmethod
    def get_parameter_space_without_n(self) -> Iterator[tuple]:
        """
        Generates an iterator over unique simulation parameter combinations,
        *excluding* the number of steps 'n'.
        Example: yields (J, B) for Ising, (k,) for Potential, (J1, J2) for Dimerized.
        """
        pass

    @abstractmethod
    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, params_without_n: tuple):
        """
        Appends gates corresponding to *one* Trotter step (evolution by delta_t)
        to the given circuit, based on the simulation parameters.

        Args:
            circuit: The qiskit.QuantumCircuit to modify.
            params_without_n: Tuple of simulation parameters (e.g., (J, B)).
        """
        pass

    @abstractmethod
    def _format_params_for_ml_input(self, n_step: int, params_without_n: tuple) -> List[float]:
        """
        Formats the current step number and simulation parameters into a flat list for ML input.
        The order *must* match get_ml_input_feature_names().

        Args:
            n_step: The current step number (from 1 to max_n_steps).
            params_without_n: Tuple of simulation parameters (e.g., (J, B)).

        Returns:
            List of floats for ML model input. Example: [n_step, J, B].
        """
        pass

    @abstractmethod
    def get_ml_input_feature_names(self) -> List[str]:
        """
        Returns the names of the ML input features in the order matching
        _format_params_for_ml_input(). Example: ['n_steps', 'J', 'B'].
        """
        pass

    @abstractmethod
    def get_observables_to_track(self) -> List[str]:
        """
        Returns a list of observable strings (e.g., "Z0", "X1", "Z0Z1")
        whose expectation values will form the ML output vector.
        This list defines the order of the output vector.
        """
        pass

    @abstractmethod
    def get_hamiltonian_operator(self, params_without_n: tuple) -> SparsePauliOp:
        """
        Constructs the Hamiltonian SparsePauliOp for the given parameters (excluding n_steps).
        Needed for exact diagonalization validation.
        """
        pass

    # --- Concrete Methods ---
    def get_output_observable_names(self) -> List[str]:
        """Returns formatted names for the ML output columns (e.g., ['exp_Z0', 'exp_X1'])."""
        return self._output_observable_names

    def _prepare_initial_circuit(self) -> qiskit.QuantumCircuit:
        """Creates a circuit and applies the initial state preparation."""
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"{self.simulation_name}_base")
        if self.initial_state_type == 'superposition':
            circuit.h(range(self.n_qubits))
            circuit.barrier(label="Init") # Separate initial state
        # 'zero' state is default
        return circuit

    # --- Core Simulation Logic ---
    def _run_simulation_trajectory(self, params_without_n: tuple) -> Iterator[Dict[str, Any]]:
        """
        Runs the simulation step-by-step for a single parameter set,
        yielding data points at specified recording steps.
        """
        circuit = self._prepare_initial_circuit()
        current_step = 0
        last_recorded_statevector = None # To avoid recalculating if dt is very small

        # Prepare exact evolution stuff if validation is enabled
        psi_exact = None
        H_matrix = None
        if self.validate_trotter:
            try:
                H_op = self.get_hamiltonian_operator(params_without_n)
                H_matrix = H_op.to_matrix(sparse=False)
                initial_state_circuit = self._prepare_initial_circuit()
                psi_initial_exact = Statevector(initial_state_circuit)
                psi_exact = psi_initial_exact.copy() # Start with initial state
            except Exception as e:
                print(f"Warning: Failed to setup exact evolution for validation: {e}")
                self.validate_trotter = False # Disable validation for this run


        # Iterate through ALL steps up to the maximum required
        for n in range(1, self.max_n_steps + 1):
            # Apply the n-th Trotter step
            self._add_trotter_step(circuit, params_without_n)
            current_step = n

            # Record data ONLY if this step 'n' is in our recording points
            if n in self.n_steps_record_points:
                statevector = None
                output_vector = [np.nan] * len(self._observables_to_track)
                entanglement = np.nan
                validation_diff = np.nan

                # Run simulator to get statevector (only if statevector mode)
                if self.simulation_mode == "statevector":
                    statevector = self.q_simulator.run_circuit_and_get_statevector(circuit)
                    if statevector:
                        # Calculate vector of expectation values
                        output_vector = self.q_simulator.calculate_expectation_vector_statevector(
                            statevector, self._observables_to_track
                        )
                        # Calculate entanglement
                        if self.compute_entanglement:
                            entanglement = self.compute_entanglement_entropy(
                                statevector, self.entanglement_partition
                            )
                        # Calculate validation diff
                        if self.validate_trotter and H_matrix is not None and psi_exact is not None:
                             try:
                                 # Evolve exact state up to time t = n * dt
                                 # Calculate U_exact for one step: exp(-i H dt)
                                 # If we already validated step n-1, just evolve by one more step
                                 # Recompute U_exact(t=n*dt) each time for simplicity now
                                 total_time = n * self.delta_t
                                 U_exact_t = expm(-1j * H_matrix * total_time)
                                 psi_exact_t_data = psi_initial_exact.data @ U_exact_t.T
                                 psi_exact_t = Statevector(psi_exact_t_data)

                                 # Calculate exact and trotter expectation vectors
                                 exact_exp_vec = self.q_simulator.calculate_expectation_vector_statevector(
                                      psi_exact_t, self._observables_to_track
                                 )
                                 trotter_exp_vec = output_vector # Already calculated

                                 # Calculate difference (e.g., L2 norm of difference vector)
                                 valid_indices = ~np.isnan(exact_exp_vec) & ~np.isnan(trotter_exp_vec)
                                 if np.any(valid_indices):
                                      diff_vec = np.array(trotter_exp_vec)[valid_indices] - np.array(exact_exp_vec)[valid_indices]
                                      validation_diff = np.linalg.norm(diff_vec)
                                 else:
                                      validation_diff = np.nan

                             except Exception as e:
                                 print(f"Error during validation at step {n}: {e}")
                                 validation_diff = np.nan
                # --- TODO: Add logic for 'shots' mode here ---
                # else: # Shots mode
                #    counts = self.q_simulator.run_circuit_and_get_counts(circuit)
                #    if counts:
                #         output_vector = self.q_simulator.calculate_expectation_vector_counts(...)
                #    # Need to adapt entanglement/validation for shots mode if desired

                # Format ML input
                ml_input = self._format_params_for_ml_input(n, params_without_n)

                # Yield data point for this step n
                yield {
                    'input': ml_input,
                    'output_vector': output_vector,
                    'entanglement': entanglement,
                    'validation_diff': validation_diff,
                    # Optional: include raw params_without_n if needed downstream
                    # 'params_set': params_without_n
                }

    def generate_dataset(self) -> List[Dict[str, Any]]:
        """
        Generates the full time-series dataset by running trajectories
        for all parameter combinations.
        """
        full_dataset = []
        parameter_space_no_n = list(self.get_parameter_space_without_n()) # Materialize iterator
        if not parameter_space_no_n:
            print("Warning: Parameter space (excluding n_steps) is empty. No data generated.")
            return []

        num_param_sets = len(parameter_space_no_n)
        num_time_points = len(self.n_steps_record_points)
        print(f"\nGenerating dataset for {self.simulation_name} (Time-Series Mode)")
        print(f"  Parameter sets (excluding n): {num_param_sets}")
        print(f"  Time points per set: {num_time_points} (up to n={self.max_n_steps})")
        print(f"  Estimated total data points: {num_param_sets * num_time_points}")

        # Use tqdm for progress bar over parameter sets
        disable_tqdm = os.getenv("CI", "false").lower() == "true"
        for params_no_n in tqdm(parameter_space_no_n, desc=f"Simulating {self.simulation_name} Trajectories", disable=disable_tqdm):
            try:
                # _run_simulation_trajectory yields results for each recorded step 'n'
                for step_result in self._run_simulation_trajectory(params_no_n):
                    full_dataset.append(step_result)
            except Exception as e:
                print(f"\nERROR during trajectory simulation for params {params_no_n}: {e}")
                # Optionally append partial results or skip this parameter set

        # --- Report Stats After Generation ---
        num_generated = len(full_dataset)
        print(f"\nDataset generation complete for {self.simulation_name}.")
        print(f"  Total data points generated: {num_generated}")

        # Check validity of output vectors
        num_valid_output = 0
        if full_dataset:
            num_outputs = len(full_dataset[0].get('output_vector', []))
            for dp in full_dataset:
                 # Check if output_vector exists and has no NaNs
                 if 'output_vector' in dp and isinstance(dp['output_vector'], list) and not np.isnan(dp['output_vector']).any():
                      num_valid_output += 1

            print(f"  Points with valid output vector: {num_valid_output}")
            if num_generated > num_valid_output:
                 failed_count = num_generated - num_valid_output
                 failure_rate = failed_count / num_generated if num_generated > 0 else 0
                 print(f"  Points failed (NaN in output vector): {failed_count} ({failure_rate:.1%} failure rate)")

            # Report on entanglement and validation if computed
            if self.compute_entanglement:
                 valid_ents = [d['entanglement'] for d in full_dataset if not np.isnan(d.get('entanglement', np.nan))]
                 if valid_ents: print(f"  Avg/Max entanglement (where computed): {np.mean(valid_ents):.4f} / {np.max(valid_ents):.4f}")
                 else: print("  No valid entanglement values computed.")
            if self.validate_trotter:
                 valid_diffs = [d['validation_diff'] for d in full_dataset if not np.isnan(d.get('validation_diff', np.nan))]
                 if valid_diffs: print(f"  Avg/Max validation diff (L2 Norm, where computed): {np.mean(valid_diffs):.4e} / {np.max(valid_diffs):.4e}")
                 else: print("  No valid Trotter validation differences computed.")

        return full_dataset
    
    def compute_entanglement_entropy(self, statevector: Statevector, partition: list[int]) -> float:
        """Computes the entanglement entropy. (Unchanged)"""
        try:
            rho_subsystem = partial_trace(statevector, list(range(partition[0], self.n_qubits))) # Qiskit traces out qubits NOT specified
            ent_entropy = entropy(rho_subsystem, base=np.exp(1))
            return float(ent_entropy)
        except Exception as e:
            print(f"ERROR computing entanglement entropy: {e}")
            return np.nan

    # --- Methods to be implemented by subclasses ---
    # get_parameter_space_without_n
    # _add_trotter_step
    # _format_params_for_ml_input
    # get_ml_input_feature_names
    # get_observables_to_track
    # get_hamiltonian_operator

    # --- Helper methods (already implemented in previous version) ---
    # compute_entanglement_entropy (signature correct)