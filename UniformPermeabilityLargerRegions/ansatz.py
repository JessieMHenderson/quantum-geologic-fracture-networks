from typing import List
from qiskit import QuantumCircuit
from typing import Sequence

__import__ = ["get_number_of_parameters", "apply_ansatz"]


def get_number_of_parameters(n_qubits: int, n_layers: int) -> int:
    """
    Calculates number of parameters for LANL ansatz

    Parameters
    ----------
    n_qubits : int
        The number of qubits in a cirquit
    n_layers : int
        The number of layers for the ansatz

    Returns
    -------
    number of parameters for LANL ansatz
    """
    return n_qubits + n_layers * (n_qubits // 2 + (n_qubits - 1) // 2) * 2


def apply_ansatz(
    qc: QuantumCircuit, qubits: List[int], n_layers: int, parameters: Sequence[float]
) -> None:
    """
    Applying LANL ansatz (sequence of gates) to specific wires

    Parameters
    ----------
    qc: QuantumCircuit
        The quantum cirquit to which we apply ansatz
    qubits: List[int]
        The indexes of qubits in the cirquit
    n_layers: int
        The number of ansatz's layers
    parameters: List[float]
        The parameters for the R_y gates

    """
    ip = 0  # parameter index
    for i in range(len(qubits)):
        qc.ry(parameters[ip], qubits[i])
        ip += 1
    for _ in range(n_layers):
        for i in range(1, len(qubits), 2):
            qc.cz(qubits[i - 1], qubits[i])
            qc.ry(parameters[ip], qubits[i - 1])
            ip += 1
            qc.ry(parameters[ip], qubits[i])
            ip += 1
        for i in range(2, len(qubits), 2):
            qc.cz(qubits[i - 1], qubits[i])
            qc.ry(parameters[ip], qubits[i - 1])
            ip += 1
            qc.ry(parameters[ip], qubits[i])
            ip += 1
