# quantum_ml_simulation/quantum_runner/circuit_builder.py
# Helper functions for building quantum circuits using Qiskit

import qiskit
from qiskit.circuit.library import RXGate, RZGate
import numpy as np

# Note: Qiskit uses 0-based indexing for qubits.

def create_rx_gate(angle: float) -> RXGate:
    """Creates a Qiskit RX gate."""
    return RXGate(angle)

def create_rz_gate(angle: float) -> RZGate:
    """Creates a Qiskit RZ gate."""
    return RZGate(angle)

def create_rzz_gate(angle: float) -> qiskit.circuit.Gate:
    """
    Creates a Qiskit Rzz gate implementing exp(-i * angle * Z_i Z_j).
    Decomposition: Rzz(theta) = CNOT(i,j) Rz(theta, j) CNOT(i,j)
    Note: Qiskit's RZZGate implements exp(-i * angle/2 * Z_i Z_j).
          If using Qiskit's built-in RZZ, the angle would need adjustment (multiply by 2).
          This implementation matches the direct decomposition formula often used.
    Args:
        angle: The rotation angle (theta).
    Returns:
        A Qiskit Gate object representing Rzz.
    """
    qc = qiskit.QuantumCircuit(2, name=f'Rzz({angle:.3f})')
    qc.cx(0, 1)
    qc.rz(angle, 1) # Apply Rz(angle) to the target qubit
    qc.cx(0, 1)
    return qc.to_gate()

def create_rxx_gate(angle: float) -> qiskit.circuit.Gate:
    """
    Creates a Qiskit Rxx gate implementing exp(-i * angle * X_i X_j).
    Decomposition: Rxx(theta) = H(i)H(j) CNOT(i,j) Rz(theta, j) CNOT(i,j) H(i)H(j)
    Note: Similar angle considerations apply as for Rzz if comparing to Qiskit's RXXGate.
    Args:
        angle: The rotation angle (theta).
    Returns:
        A Qiskit Gate object representing Rxx.
    """
    qc = qiskit.QuantumCircuit(2, name=f'Rxx({angle:.3f})')
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(angle, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    return qc.to_gate()

def create_ryy_gate(angle: float) -> qiskit.circuit.Gate:
    """
    Creates a Qiskit Ryy gate implementing exp(-i * angle * Y_i Y_j).
    Decomposition: Ryy(theta) = Rx(pi/2,i) Rx(pi/2,j) CNOT(i,j) Rz(theta, j) CNOT(i,j) Rx(-pi/2,i) Rx(-pi/2,j)
    Note: Similar angle considerations apply as for Rzz if comparing to Qiskit's RYYGate.
    Args:
        angle: The rotation angle (theta).
    Returns:
        A Qiskit Gate object representing Ryy.
    """
    qc = qiskit.QuantumCircuit(2, name=f'Ryy({angle:.3f})')
    pi_half = np.pi / 2.0
    qc.rx(pi_half, 0)
    qc.rx(pi_half, 1)
    qc.cx(0, 1)
    qc.rz(angle, 1)
    qc.cx(0, 1)
    qc.rx(-pi_half, 0)
    qc.rx(-pi_half, 1)
    return qc.to_gate()