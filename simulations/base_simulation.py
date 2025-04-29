# quantum_ml_simulation/simulations/base_simulation.py
# Abstract base class for quantum simulations

from abc import ABC, abstractmethod
import qiskit
import time
import numpy as np
from tqdm.auto import tqdm # Use auto version for better notebook compatibility

# Use relative imports within the package
from ..quantum_runner import circuit_builder as cb
from ..quantum_runner import simulator as qs
from ..config import simulation_params as cfg

class BaseSimulation(ABC):
    """Abstract base class for defining and running quantum simulations."""

    def __init__(self, simulation_name: str, n_qubits: int, measurement_operator: str):
        """
        Initializes the base simulation.

        Args:
            simulation_name: A unique name for this simulation type (e.g., "IsingModel").
            n_qubits: The number of qubits required for this simulation.
            measurement_operator: String identifier for the observable to measure (e.g., "Z1Z2").
        """
        self.simulation_name = simulation_name
        self.n_qubits = n_qubits
        self.measurement_operator = measurement_operator
        self.q_simulator = qs.QuantumSimulator() # Instantiate the quantum runner
        self.delta_t = cfg.DELTA_T
        self.n_steps_range = cfg.N_STEPS_RANGE
        print(f"Initialized BaseSimulation: {self.simulation_name} ({self.n_qubits} qubits, measure {self.measurement_operator})")

    @abstractmethod
    def get_parameter_space(self) -> list[tuple]:
        """
        Generates the list of all parameter combinations to simulate.

        Returns:
            A list of tuples, where each tuple contains the parameters for one simulation run
            (e.g., [(n_steps, J, B), (n_steps, J, B), ...]).
            The order within the tuple must match the order expected by build_circuit_for_params
            and _format_params_for_ml.
        """
        pass

    @abstractmethod
    def build_circuit_for_params(self, params: tuple) -> qiskit.QuantumCircuit:
        """
        Builds the Qiskit QuantumCircuit for a given set of simulation parameters.

        Args:
            params: A tuple containing the parameters for this specific run
                    (e.g., (n_steps, J, B)). Must match the order from get_parameter_space.

        Returns:
            A Qiskit QuantumCircuit object ready for simulation.
        """
        pass

    @abstractmethod
    def _format_params_for_ml(self, params: tuple) -> list[float]:
        """
        Converts the simulation parameters tuple into a flat list of floats suitable for ML model input.

        Args:
            params: A tuple containing the parameters for this specific run
                    (e.g., (n_steps, J, B)). Must match the order from get_parameter_space.

        Returns:
            A list of floating-point numbers representing the input features for the ML model.
            The order must be consistent and match get_ml_input_feature_names.
        """
        pass

    @abstractmethod
    def get_ml_input_feature_names(self) -> list[str]:
        """
        Returns the names of the input features provided by _format_params_for_ml.

        Returns:
            A list of strings, e.g., ['n_steps', 'J', 'B']. The order must match
            the output of _format_params_for_ml.
        """
        pass

    def run_single_simulation(self, params: tuple) -> float:
        """
        Runs one simulation instance for the given parameters.

        Args:
            params: The parameter tuple for this run.

        Returns:
            The calculated expectation value (float).
        """
        start_time = time.time()
        try:
            circuit = self.build_circuit_for_params(params)
            # Optional: Print circuit depth for debugging
            # print(f"Circuit depth for params {params}: {circuit.depth()}")
            statevector = self.q_simulator.run_circuit_and_get_statevector(circuit)
            expectation_value = self.q_simulator.calculate_expectation_value(
                statevector, self.measurement_operator
            )
            end_time = time.time()
            # Optional: Print simulation time
            # print(f"Sim duration for {params}: {end_time - start_time:.3f}s")
            return float(expectation_value) # Ensure float output

        except Exception as e:
            print(f"ERROR during single simulation run for params {params}: {e}")
            # Decide how to handle errors: re-raise, return NaN, etc.
            # Returning NaN allows processing to continue but flags bad points.
            return np.nan # Return Not-a-Number to indicate failure

    def generate_dataset(self) -> list[dict]:
        """
        Generates the full dataset by running simulations for all parameter combinations.

        Returns:
            A list of dictionaries, where each dictionary represents a data point:
            {'input': [ml_feature_1, ...], 'output': expectation_value, 'raw_params': (param1, ...)}
            Points where simulation failed will have 'output': np.nan.
        """
        dataset = []
        parameter_space = self.get_parameter_space()
        if not parameter_space:
            print("Warning: Parameter space is empty. No data generated.")
            return []

        print(f"Generating dataset for {self.simulation_name} with {len(parameter_space)} parameter sets...")

        # Use tqdm for progress bar
        for params in tqdm(parameter_space, desc=f"Simulating {self.simulation_name}"):
            expectation_value = self.run_single_simulation(params)
            ml_input_features = self._format_params_for_ml(params)
            dataset.append({
                "input": ml_input_features,
                "output": expectation_value,
                "raw_params": params
            })

        # Count valid vs failed simulations
        num_generated = len(dataset)
        num_failed = sum(1 for d in dataset if np.isnan(d['output']))
        print(f"Dataset generation complete for {self.simulation_name}.")
        print(f"  Total points attempted: {num_generated}")
        if num_failed > 0:
            print(f"  Points failed (output is NaN): {num_failed}")
        print(f"  Points successful: {num_generated - num_failed}")

        return dataset