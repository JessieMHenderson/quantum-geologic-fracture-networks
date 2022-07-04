from typing import Sequence
from pitch_fork import ansatz
from qiskit import Aer, execute, QuantumCircuit

N_DECIMALS = 11

__import__ = ["get_statevector"]


def get_statevector(
    parameters: Sequence[float], n_qubits: int, n_layers: int
) -> Sequence[float]:
    """
    Simulate state vector for set of anstanz parameters

    Parameters
    ----------
    parameters: Sequence[float]
        The ansatz parameters
    n_qubits : int
        The number of qubits in a cirquit
    n_layers : int
        The number of layers for the ansatz

    Returns
    -------
    State vector
    """
    if len(parameters) != ansatz.get_number_of_parameters(n_qubits, n_layers):
        raise ValueError("Incorrect number of parameters")
    qubits = list(range(n_qubits))
    circ = QuantumCircuit(n_qubits, n_qubits)
    ansatz.apply_ansatz(circ, qubits, n_layers, parameters)
    job = execute(circ, Aer.get_backend("statevector_simulator"))
    state_vector = job.result().get_statevector(circ, decimals=N_DECIMALS)
    return state_vector
