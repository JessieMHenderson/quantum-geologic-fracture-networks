from multiprocessing import Value
import numpy as np
from typing import Optional, Dict, Any
from typing import Sequence
from pitch_fork import simulation, ansatz
from qiskit import Aer, execute, QuantumCircuit
from scipy.optimize import minimize
import json
import logging
import time
import uuid
from pathlib import Path, PurePath
import pickle

N_DECIMALS = 11

__import__ = ["cost_function", "gradient", "optimize"]


def cost_function(
    parameters: Sequence[float], x_true: np.ndarray, n_qubits: int, n_layers: int
) -> float:
    """
    Compute 1-|<x_true|theta(parameters)>|^2
    where |theta(parameters)> = ansatz(parameters)|0>

    Parameters
    ----------
    parameters: Sequence[float]
        The ansatz parameters
    x_true: np.ndarray
        The vector
    n_qubits : int
        The number of qubits in a cirquit
    n_layers : int
        The number of layers for the ansatz

    Returns
    -------
    Cost function
    """
    state_vector = simulation.get_statevector(parameters, n_qubits, n_layers)
    return 1 - np.abs(np.sum(state_vector * x_true)) ** 2


def gradient(
    parameters: Sequence[float], x_true: np.ndarray, n_qubits: int, n_layers: int
) -> Sequence[float]:
    """
    Compute gradient of <theta(parameters)|H|theta(parameters)>
    where |theta(parameters)> = ansatz(parameters)|0>

    Parameters
    ----------
    parameters: Sequence[float]
        The ansatz parameters
    x_true: np.ndarray
        The vector
    n_qubits : int
        The number of qubits in a cirquit
    n_layers : int
        The number of layers for the ansatz

    Returns
    -------
    Gradient
    """
    if len(parameters) != ansatz.get_number_of_parameters(n_qubits, n_layers):
        raise ValueError("Incorrect number of parameters")
    shift_p = np.array(parameters)
    shift_n = np.array(parameters)
    grad = []
    for i in range(len(parameters)):
        shift_p[i] = parameters[i] + np.pi / 2
        shift_n[i] = parameters[i] - np.pi / 2

        cost_p = cost_function(shift_p, x_true, n_qubits, n_layers)
        cost_n = cost_function(shift_n, x_true, n_qubits, n_layers)

        grad.append((cost_p - cost_n) / 2)

        shift_p[i] = parameters[i]
        shift_n[i] = parameters[i]
    return np.array(grad)


def optimize(
    x_true: Optional[np.ndarray] = None,
    n_qubits: Optional[int] = None,
    n_layers: Optional[int] = None,
    max_iter: int = 200,
    x0: Optional[np.ndarray] = None,
    result_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    continue_: bool = False,
    extra_iters: int = 0,
) -> Dict[str, Any]:
    if result_dir is None:
        result_dir = "./"
    status = [time.time(), 0]
    value_history = {
        "parameters": [],
        "fidelity": [],
    }

    if not continue_:
        if x_true is None or n_qubits is None or n_layers is None:
            raise ValueError("x_true, n_qubits, n_layers are required variables")
    else:
        if run_id is None:
            raise ValueError("run_id is required to continue optimization")
        result_path = PurePath(result_dir).joinpath(run_id)
        if (
            extra_iters == 0
            and Path(result_path.joinpath("optimizer_out.dat")).exists()
        ):
            raise ValueError("Optimization already completed")
        with open(str(result_path.joinpath("optimization.dat")), "rb") as f:
            optimization_parameters = pickle.load(f)
            x_true = optimization_parameters["x_true"]
            n_qubits = optimization_parameters["n_qubits"]
            n_layers = optimization_parameters["n_layers"]
            max_iter = optimization_parameters["max_iter"]
            x0 = optimization_parameters["x0"]
        iterations_completed = []
        for fpath in Path(result_path).iterdir():
            if fpath.match("iteration_*.dat"):
                iterations_completed.append(
                    int(fpath.name.replace("iteration_", "").replace(".dat", ""))
                )
        iterations_completed.sort()
        for iter in iterations_completed:
            with open(str(result_path.joinpath(f"iteration_{iter}.dat")), "rb") as f:
                state = pickle.load(f)
                value_history["parameters"].append(np.array(state["parameters"]))
                value_history["fidelity"].append(state["fidelity"])
        if len(iterations_completed) > 0:
            status[1] = iterations_completed[-1] + 1
            x0 = value_history["parameters"][-1]

    if run_id is None:
        run_id = str(uuid.uuid4())
    result_path = PurePath(result_dir).joinpath(run_id)
    if not Path(result_path).exists():
        Path(result_path).mkdir(parents=True)

    def c(parameters):
        return cost_function(parameters, x_true, n_qubits, n_layers)

    def g(parameters):
        return gradient(parameters, x_true, n_qubits, n_layers)

    def callback(parameters):
        state_vector = simulation.get_statevector(parameters, n_qubits, n_layers)
        value_history["parameters"].append(np.copy(parameters))

        fidelity = np.abs(np.sum(state_vector * x_true)) ** 2
        value_history["fidelity"].append(fidelity)

        state = {
            "parameters": parameters.tolist(),
            "fidelity": fidelity,
            "iteration_time": time.time() - status[0],
        }
        Path(result_path.joinpath(f"iteration_{status[1]}.json")).write_text(
            json.dumps(
                state,
                sort_keys=True,
                indent=4,
            )
        )
        with open(str(result_path.joinpath(f"iteration_{status[1]}.dat")), "wb") as f:
            pickle.dump(state, f)

        logging.warn(f"Iteration[{time.time() - status[0]:0.4}s]: fidelity={fidelity}")
        status[0] = time.time()
        status[1] += 1

    n_parameters = ansatz.get_number_of_parameters(n_qubits, n_layers)
    if x0 is None:
        x0 = np.random.rand(n_parameters) * 2 * np.pi
    else:
        if len(x0) != n_parameters:
            raise ValueError("Incorrect number of parameters")
    if not continue_:
        optimization_parameters = {
            "x_true": x_true.tolist(),
            "x0": x0.tolist(),
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "max_iter": max_iter,
        }
        Path(result_path.joinpath("optimization.json")).write_text(
            json.dumps(optimization_parameters, sort_keys=True, indent=4)
        )
        with open(str(result_path.joinpath("optimization.dat")), "wb") as f:
            pickle.dump(optimization_parameters, f)

    status[0] = time.time()
    out = minimize(
        c,
        x0,
        jac=g,
        method="CG",
        options={"maxiter": max_iter + extra_iters},
        callback=callback,
    )

    final_state = {
        "x": out["x"].tolist(),
        "success": out["success"],
        "status": out["status"],
        "message": out["message"],
        "fun": out["fun"],
        "jac": out["jac"].tolist(),
        "nit": out["nit"],
        "nfev": out["nfev"],
    }
    Path(result_path.joinpath("optimizer_out.json")).write_text(
        json.dumps(
            final_state,
            sort_keys=True,
            indent=4,
        )
    )
    with open(str(result_path.joinpath("optimizer_out.dat")), "wb") as f:
        pickle.dump(final_state, f)

    return {"optimizer_out": out, "history": value_history}
