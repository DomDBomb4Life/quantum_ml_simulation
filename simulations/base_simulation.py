# quantum_ml_simulation/simulations/base_simulation.py
# Abstract base class (Hybrid Sampling & Initial State Variation)

from abc import ABC, abstractmethod
import os
import qiskit
import time
import numpy as np
import itertools
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, partial_trace, entropy, SparsePauliOp
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Optional, Iterator, Any, Union

# Use relative imports
from ..quantum_runner import simulator as qs
from ..config import simulation_params as cfg

class BaseSimulation(ABC):
    """
    Abstract base class for quantum simulations generating time-series data,
    supporting grid/random parameter sampling and varying initial states.
    """

    def __init__(self,
                 simulation_name: str,
                 n_qubits: int,
                 delta_t: float,
                 n_steps_range: List[int], # Points to record
                 initial_state_config: Dict, # Config dict for initial state
                 simulation_mode: str,
                 sampling_method: str, # 'grid' or 'random'
                 num_parameter_sets: Optional[int], # Number for random sampling
                 parameter_config: Dict, # Sampling ranges/values for system params
                 **extra_args): # For fixed parameters like fixed_J
        """Initializes the base simulation with comprehensive configuration."""

        self.simulation_name = simulation_name
        self.n_qubits = n_qubits
        self.delta_t = delta_t
        self.n_steps_record_points = sorted([n for n in n_steps_range if n > 0])
        if not self.n_steps_record_points:
            raise ValueError("n_steps_range must contain positive step numbers.")
        self.max_n_steps = self.n_steps_record_points[-1]
        self.initial_state_config = initial_state_config
        self.simulation_mode = simulation_mode
        self.sampling_method = sampling_method
        self.num_parameter_sets = num_parameter_sets # Target for random sampling
        self.parameter_config = parameter_config # Stores ranges/values for system params
        self.extra_args = extra_args # Stores fixed parameters like fixed_J

        # Instantiate simulator
        self.q_simulator = qs.QuantumSimulator(
            simulation_mode=self.simulation_mode, n_shots=cfg.N_SHOTS
        )

        # Advanced Feature Flags
        self.validate_trotter = cfg.VALIDATE_TROTTER and self.n_qubits <= cfg.VALIDATION_MAX_QUBITS
        self.compute_entanglement = cfg.COMPUTE_ENTANGLEMENT
        self.entanglement_partition = list(range(min(cfg.ENTANGLEMENT_PARTITION, self.n_qubits // 2)))

        # Determine initial state parameter names (if any)
        self._initial_state_param_names = self.get_initial_state_param_names()
        self._system_param_names = self.get_system_parameter_names() # Get system param names

        # Store tracked observable names
        self._observables_to_track = self.get_observables_to_track()
        self._output_observable_names = [f"exp_{obs}" for obs in self._observables_to_track]

        # Store full ML input feature names
        self._ml_input_feature_names = self.get_ml_input_feature_names()

        # Print Initialization Info
        print(f"Initialized BaseSimulation: {self.simulation_name} (Hybrid Sampling Mode)")
        print(f"  Mode: {self.simulation_mode}, Qubits: {self.n_qubits}")
        print(f"  dt: {self.delta_t}, Record Steps: {self.n_steps_record_points[0]}-{self.max_n_steps}")
        print(f"  Initial State Type: {self.initial_state_config['type']}")
        if self._initial_state_param_names:
            print(f"  Initial State Params: {self._initial_state_param_names}")
        print(f"  Sampling Method: {self.sampling_method}" + (f" (Target Sets: {self.num_parameter_sets})" if self.sampling_method == 'random' else ""))
        print(f"  System Params: {self._system_param_names}")
        print(f"  Tracked Observables: {self._observables_to_track}")
        print(f"  ML Input Features: {self._ml_input_feature_names}")
        print(f"  ML Output Columns: {self._output_observable_names}")
        if self.extra_args: print(f"  Fixed Args: {self.extra_args}")
        if self.validate_trotter: print(f"  Trotter validation ENABLED (Tolerance: {cfg.VALIDATION_TOLERANCE})")
        else: print(f"  Trotter validation DISABLED")
        if self.compute_entanglement: print(f"  Entanglement computation ENABLED (Partition: {self.entanglement_partition})")
        else: print(f"  Entanglement computation DISABLED")


    # --- Abstract Methods (to be implemented by subclasses) ---

    @abstractmethod
    def get_system_parameter_names(self) -> List[str]:
        """Returns the list of system parameter names that vary (e.g., ['J', 'B'])."""
        pass

    @abstractmethod
    def get_param_sampling_config(self) -> Dict[str, Union[Tuple[float, float], List]]:
        """
        Returns the configuration for sampling system parameters.
        Example: {'J': (0.1, 1.0), 'B': (0.0, 0.5)} for random sampling,
                 {'J': [0.1, 0.5, 1.0], 'B': [0.0, 0.25, 0.5]} for grid sampling.
        """
        pass

    @abstractmethod
    def _add_trotter_step(self, circuit: qiskit.QuantumCircuit, system_params: tuple):
        """Appends gates for one Trotter step based on system parameters."""
        pass

    @abstractmethod
    def get_observables_to_track(self) -> List[str]:
        """Returns a list of observable strings for the ML output vector."""
        pass

    @abstractmethod
    def get_hamiltonian_operator(self, system_params: tuple) -> SparsePauliOp:
        """Constructs the Hamiltonian using only system parameters."""
        pass

    # --- Concrete Methods ---

    def get_initial_state_param_names(self) -> List[str]:
        """Determines initial state parameter names based on config."""
        config = self.initial_state_config
        if config.get('type') == 'random_rotations':
            # Expect 3 angles (theta, phi, lambda) per qubit
            names = []
            for i in range(self.n_qubits):
                names.extend([f"theta_{i}", f"phi_{i}", f"lambda_{i}"])
            return names
        else: # 'zero', 'superposition' don't have varying parameters
            return []

    def _prepare_initial_circuit(self, initial_state_params: Optional[Dict[str, float]] = None) -> qiskit.QuantumCircuit:
        """Creates a circuit and applies initial state based on config and params."""
        circuit = qiskit.QuantumCircuit(self.n_qubits, name=f"{self.simulation_name}_base")
        state_type = self.initial_state_config.get('type', 'zero')

        if state_type == 'superposition':
            circuit.h(range(self.n_qubits))
        elif state_type == 'random_rotations':
            if initial_state_params is None:
                raise ValueError("Initial state parameters are required for 'random_rotations' type.")
            # Apply U(theta, phi, lambda) gate which is Rz(phi+pi)Rx(theta+pi)Rz(lambda) approx?
            # Easier: Apply Rz(lambda) Ry(theta) Rz(phi) or Rx, Ry, Rz sequentially
            for i in range(self.n_qubits):
                theta = initial_state_params.get(f"theta_{i}", 0.0)
                phi = initial_state_params.get(f"phi_{i}", 0.0)
                lambd = initial_state_params.get(f"lambda_{i}", 0.0) # Use lambda without 'a'
                # Apply U3 gate: U(theta, phi, lambda) = Rz(phi)Ry(theta)Rz(lambda) up to global phase
                # Qiskit's U gate is U(theta, phi, lambda)
                circuit.u(theta, phi, lambd, i)
        # 'zero' state is the default, no operation needed

        circuit.barrier(label="Init")
        return circuit

    def get_ml_input_feature_names(self) -> List[str]:
         """Combines step, system, and initial state parameter names."""
         return ['n_steps'] + self.get_system_parameter_names() + self.get_initial_state_param_names()

    def _format_params_for_ml_input(self, n_step: int, combined_params: tuple) -> List[float]:
        """Formats the input vector [n_step, sys_params..., init_state_params...]."""
        # Assumes combined_params tuple contains system params followed by initial state params
        return [float(n_step)] + [float(p) for p in combined_params]

    def get_output_observable_names(self) -> List[str]:
        """Returns formatted names for the ML output columns."""
        return self._output_observable_names

    def _generate_parameter_sets(self) -> Iterator[tuple]:
        """Generates combined parameter sets (system + initial state) using the specified sampling method."""
        system_param_names = self.get_system_parameter_names()
        initial_state_param_names = self.get_initial_state_param_names()
        sampling_config = self.get_param_sampling_config()

        # --- Random Sampling ---
        if self.sampling_method == 'random':
            if self.num_parameter_sets is None or self.num_parameter_sets <= 0:
                 raise ValueError("Positive 'num_parameter_sets' required for random sampling.")
            print(f"Generating {self.num_parameter_sets} random parameter sets...")

            # Get system parameter ranges
            sys_param_ranges = {}
            for name in system_param_names:
                range_val = sampling_config.get(name)
                if not isinstance(range_val, tuple) or len(range_val) != 2:
                    raise ValueError(f"Expected (min, max) tuple for '{name}' in random sampling config.")
                sys_param_ranges[name] = range_val

            # Get initial state parameter ranges (if any)
            init_param_range = self.initial_state_config.get("rotation_angle_range")
            if initial_state_param_names and (not isinstance(init_param_range, tuple) or len(init_param_range) != 2):
                 raise ValueError("Expected (min, max) tuple for 'rotation_angle_range' in initial state config.")

            # Generate samples
            for _ in range(self.num_parameter_sets):
                system_params = [np.random.uniform(low=mn, high=mx) for mn, mx in sys_param_ranges.values()]
                initial_state_params = []
                if initial_state_param_names:
                     mn, mx = init_param_range
                     initial_state_params = np.random.uniform(low=mn, high=mx, size=len(initial_state_param_names)).tolist()

                yield tuple(system_params + initial_state_params) # Yield combined flattened tuple


        # --- Grid Sampling ---
        elif self.sampling_method == 'grid':
            print("Generating parameter sets using grid sampling...")
            system_param_value_lists = []
            for name in system_param_names:
                values = sampling_config.get(name)
                if not isinstance(values, list):
                     raise ValueError(f"Expected list of values for '{name}' in grid sampling config.")
                system_param_value_lists.append(values)

            initial_state_param_value_lists = []
            if initial_state_param_names:
                # Grid sampling initial state angles? Less common, but possible.
                # For simplicity, let's stick to random for initial state even in grid mode,
                # OR only allow grid sampling if initial state type is 'zero'/'superposition'.
                if self.initial_state_config.get('type') == 'random_rotations':
                     raise NotImplementedError("Grid sampling for random initial state rotations not implemented. Use random sampling method or fixed initial state type ('zero', 'superposition') for grid system param sampling.")
                # Otherwise, init state params list is empty.

            # Generate grid product
            if not system_param_value_lists: # Handle case with no varying system params
                 if not initial_state_param_value_lists: yield tuple() # No varying params at all
                 else: yield from itertools.product(*initial_state_param_value_lists)
            else:
                 if not initial_state_param_value_lists: yield from itertools.product(*system_param_value_lists)
                 else: yield from itertools.product(*(system_param_value_lists + initial_state_param_value_lists))
        else:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")


    def _run_simulation_trajectory(self, combined_params: tuple) -> Iterator[Dict[str, Any]]:
        """Runs step-by-step simulation for one combined parameter set."""

        # --- Unpack Combined Parameters ---
        num_sys_params = len(self.get_system_parameter_names())
        num_init_params = len(self.get_initial_state_param_names())

        if len(combined_params) != num_sys_params + num_init_params:
             raise ValueError(f"Parameter tuple length mismatch. Expected {num_sys_params + num_init_params}, got {len(combined_params)}.")

        system_params = combined_params[:num_sys_params]
        initial_state_params_tuple = combined_params[num_sys_params:]
        # Convert initial state params tuple to dict for easier access in _prepare_initial_circuit
        initial_state_params_dict = dict(zip(self._initial_state_param_names, initial_state_params_tuple))

        # --- Prepare Initial Circuit ---
        circuit = self._prepare_initial_circuit(initial_state_params_dict)

        # --- Prepare for Validation (if enabled) ---
        psi_initial_exact = None
        H_matrix = None
        validation_enabled_for_run = self.validate_trotter # Store initial flag state
        if validation_enabled_for_run:
            try:
                H_op = self.get_hamiltonian_operator(system_params) # Use system params only
                H_matrix = H_op.to_matrix(sparse=False)
                # Important: Use the *actual* initial state for exact evolution
                initial_state_circuit_exact = self._prepare_initial_circuit(initial_state_params_dict)
                psi_initial_exact = Statevector(initial_state_circuit_exact)
            except Exception as e:
                print(f"Warning: Failed to setup exact evolution for validation (params: {combined_params}): {e}")
                validation_enabled_for_run = False # Disable for this specific trajectory

        # --- Evolve Step-by-Step ---
        for n in range(1, self.max_n_steps + 1):
            # Apply n-th Trotter step using system params
            self._add_trotter_step(circuit, system_params)

            # Record data if this step 'n' is requested
            if n in self.n_steps_record_points:
                statevector = None
                output_vector = [np.nan] * len(self._observables_to_track)
                entanglement = np.nan
                validation_diff = np.nan

                # --- Statevector Mode ---
                if self.simulation_mode == "statevector":
                    statevector = self.q_simulator.run_circuit_and_get_statevector(circuit)
                    if statevector:
                        output_vector = self.q_simulator.calculate_expectation_vector_statevector(
                            statevector, self._observables_to_track
                        )
                        if self.compute_entanglement:
                            entanglement = self.compute_entanglement_entropy(
                                statevector, self.entanglement_partition
                            )
                        # Validation
                        if validation_enabled_for_run and H_matrix is not None and psi_initial_exact is not None:
                             try:
                                 total_time = n * self.delta_t
                                 U_exact_t = expm(-1j * H_matrix * total_time)
                                 psi_exact_t_data = psi_initial_exact.data @ U_exact_t.T
                                 psi_exact_t = Statevector(psi_exact_t_data)
                                 exact_exp_vec = self.q_simulator.calculate_expectation_vector_statevector(
                                      psi_exact_t, self._observables_to_track
                                 )
                                 trotter_exp_vec = output_vector
                                 valid_idx = ~np.isnan(exact_exp_vec) & ~np.isnan(trotter_exp_vec)
                                 if np.any(valid_idx):
                                      diff_vec = np.array(trotter_exp_vec)[valid_idx] - np.array(exact_exp_vec)[valid_idx]
                                      validation_diff = np.linalg.norm(diff_vec)
                                 else: validation_diff = np.nan
                             except Exception as e: print(f"Error validation step {n}: {e}"); validation_diff = np.nan
                # --- TODO: Shots Mode ---
                # else: ...

                # Format ML input using combined params
                ml_input = self._format_params_for_ml_input(n, combined_params)

                yield {
                    'input': ml_input,
                    'output_vector': output_vector,
                    'entanglement': entanglement,
                    'validation_diff': validation_diff,
                    # 'raw_params': combined_params # Optional: If needed for detailed analysis later
                }

    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generates the full dataset using configured sampling method."""
        full_dataset = []
        parameter_sets_iterator = self._generate_parameter_sets() # Get iterator/list

        # Estimate total parameter sets for progress bar (difficult for pure iterator)
        # If random sampling, we know the number. If grid, we can calculate it.
        num_param_sets_estimate = self.num_parameter_sets # For random
        if self.sampling_method == 'grid':
            try: # Try calculating grid size
                 grid_lists = [self.get_param_sampling_config().get(name, []) for name in self.get_system_parameter_names()]
                 if self.get_initial_state_param_names(): # Add grid for initial state if applicable (currently not for random rotations)
                      pass # Add logic if grid sampling initial state is implemented
                 num_param_sets_estimate = np.prod([len(lst) for lst in grid_lists if lst]) if grid_lists else 1
            except: num_param_sets_estimate = None # Fallback if calculation fails


        num_time_points = len(self.n_steps_record_points)
        print(f"\nGenerating dataset for {self.simulation_name} (Hybrid Sampling Mode)")
        if num_param_sets_estimate:
            print(f"  Parameter sets generated/sampled: {num_param_sets_estimate}")
            print(f"  Time points per set: {num_time_points} (up to n={self.max_n_steps})")
            print(f"  Estimated total data points: {num_param_sets_estimate * num_time_points}")
        else:
             print(f"  Generating parameter sets via iterator...")
             print(f"  Time points per set: {num_time_points} (up to n={self.max_n_steps})")


        disable_tqdm = os.getenv("CI", "false").lower() == "true"
        # Use total=num_param_sets_estimate if available for tqdm progress
        for combined_params in tqdm(parameter_sets_iterator, desc=f"Simulating {self.simulation_name} Trajectories", total=num_param_sets_estimate, disable=disable_tqdm):
            try:
                for step_result in self._run_simulation_trajectory(combined_params):
                    full_dataset.append(step_result)
            except Exception as e:
                print(f"\nERROR during trajectory simulation for params {combined_params}: {e}")

        # --- Reporting (same as before) ---
        num_generated = len(full_dataset)
        print(f"\nDataset generation complete for {self.simulation_name}.")
        print(f"  Total data points generated: {num_generated}")
        # ... (rest of reporting logic is the same) ...
        num_valid_output = 0
        if full_dataset:
            num_outputs = len(full_dataset[0].get('output_vector', []))
            for dp in full_dataset:
                 if 'output_vector' in dp and isinstance(dp['output_vector'], list) and not np.isnan(dp['output_vector']).any():
                      num_valid_output += 1
            print(f"  Points with valid output vector: {num_valid_output}")
            if num_generated > num_valid_output:
                 failed_count = num_generated - num_valid_output
                 failure_rate = failed_count / num_generated if num_generated > 0 else 0
                 print(f"  Points failed (NaN in output vector): {failed_count} ({failure_rate:.1%} failure rate)")
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
        """Computes the entanglement entropy."""
        # Corrected implementation for Qiskit >= 1.0
        if not partition or partition[-1] >= self.n_qubits:
            print(f"Warning: Invalid entanglement partition {partition} for {self.n_qubits} qubits. Skipping.")
            return np.nan
        try:
            # Qiskit's partial_trace traces out the specified qubits.
            # To keep partition `p`, we need to trace out qubits NOT in `p`.
            qubits_to_trace = list(set(range(self.n_qubits)) - set(partition))
            if not qubits_to_trace: # Cannot trace out empty set or all qubits if partition is full
                 print("Warning: Cannot compute entanglement for trivial partition (all or no qubits).")
                 return 0.0 # Entropy of full pure state is 0
            rho_subsystem = partial_trace(statevector, qubits_to_trace)
            ent_entropy = entropy(rho_subsystem.data, base=np.exp(1)) # Use .data for matrix
            return float(np.real(ent_entropy)) # Ensure real float
        except Exception as e:
            print(f"ERROR computing entanglement entropy for partition {partition}: {e}")
            # import traceback; traceback.print_exc() # Debugging
            return np.nan