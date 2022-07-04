from typing import Sequence, Optional, List, Union
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Job, Backend, JobStatus
from qiskit import Aer, transpile, assemble, execute
from collections import defaultdict
import numpy as np
from qiskit.pulse import configuration
from scipy.optimize import minimize
from typing import List, Tuple
import json
from pitch_fork import ansatz
from pathlib import Path, PurePath
from json import JSONEncoder
import datetime


__import__ = ["get_backend", "run", "run_with_qiskit_runtime", "get_coupling_layouts"]


class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


def get_backend(
    backend_name: str,
) -> Backend:
    provider = qiskit.IBMQ.get_provider('Fill In Provider Here')
    backend = provider.get_backend(backend_name)
    return backend


def run(
    parameters: Sequence[float],
    n_qubits: int,
    n_layers: int,
    backend: Union[Backend, str],
    n_shots: int = 8192,
    initial_layout: Optional[List] = None,
    optimization_level: int = 3,
    result_dir: Optional[str] = None,
) -> Job:
    if len(parameters) != ansatz.get_number_of_parameters(n_qubits, n_layers):
        raise ValueError("Incorrect number of parameters")
    qubits = list(range(n_qubits))
    circuit = QuantumCircuit(n_qubits, n_qubits)
    ansatz.apply_ansatz(circuit, qubits, n_layers, parameters)
    circuit.measure(qubits, qubits)
    if isinstance(backend, str):
        backend = get_backend(backend)
    job = execute(
        circuit,
        backend,
        shots=n_shots,
        initial_layout=initial_layout,
        optimization_level=optimization_level,
    )
    job.wait_for_final_state()
    if result_dir is not None:
        rdir = PurePath(result_dir)
        Path(rdir).mkdir(exist_ok=True)
        status = job.status()
        if status == JobStatus.DONE:
            result = job.result()
            unified = defaultdict(int)
            for k, v in result.get_counts().items():
                unified[f"0x{int(k, 2):x}"] += v
            unified = dict(unified)
        else:
            result = ""
            unified = ""

        if status != JobStatus.DONE:
            Path(rdir).joinpath(job.job_id() + ".json").write_text(
                json.dumps(
                    {
                        "status": status.name,
                        "error_message": job.error_message(),
                    },
                    sort_keys=True,
                    indent=4,
                ),
            )
        else:
            properties = backend.properties(refresh=True)
            configuration = backend.configuration()
            Path(rdir).joinpath(job.job_id() + ".json").write_text(
                json.dumps(
                    {
                        "result": result.get_counts(),
                        "status": status.name,
                        "initial_layout": initial_layout,
                        "n_shots": n_shots,
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "parameters": list(parameters),
                        "backend_name": backend.name(),
                        "optimization_level": optimization_level,
                        "backend_properties": properties.to_dict(),
                        "backend_configuration": configuration.to_dict(),
                        "unified_result": unified,
                    },
                    sort_keys=True,
                    indent=4,
                    cls=DateTimeEncoder,
                )
            )
    return job


def run_with_qiskit_runtime(
    parameters: Sequence[float],
    n_qubits: int,
    n_layers: int,
    backend_name: str,
    n_shots: int = 8192,
    initial_layout: Optional[List] = None,
    optimization_level: int = 3,
    result_dir: Optional[str] = None,
) -> Job:
    if len(parameters) != ansatz.get_number_of_parameters(n_qubits, n_layers):
        raise ValueError("Incorrect number of parameters")
    backend = get_backend(backend_name)
    qubits = list(range(n_qubits))
    circuit = QuantumCircuit(n_qubits, n_qubits)
    ansatz.apply_ansatz(circuit, qubits, n_layers, parameters)
    circuit.measure(qubits, qubits)
    n_multiple = (n_shots + 8191) // 8192
    program_inputs = {
        "circuits": [circuit] * n_multiple,
        "optimization_level": optimization_level,
        "shots": 8192,
        "initial_layout": initial_layout,
    }
    options = {"backend_name": backend_name}
    provider = qiskit.IBMQ.get_provider('Fill In Provider Here')
    job = provider.runtime.run(
        program_id="circuit-runner",
        options=options,
        inputs=program_inputs,
    )
    job.wait_for_final_state()
    if result_dir is not None:
        rdir = PurePath(result_dir)
        Path(rdir).mkdir(exist_ok=True)
        status = job.status()

        if status != JobStatus.DONE:
            Path(rdir).joinpath(job.job_id() + ".json").write_text(
                json.dumps(
                    {
                        "status": status.name,
                        "error_message": job.error_message(),
                    },
                    sort_keys=True,
                    indent=4,
                ),
            )
        else:
            properties = backend.properties(refresh=True)
            configuration = backend.configuration()
            result = job.result()
            combined = defaultdict(int)
            for r in result["results"]:
                for k, v in r["data"]["counts"].items():
                    combined[k] += v
            combined = dict(combined)
            Path(rdir).joinpath(job.job_id() + ".json").write_text(
                json.dumps(
                    {
                        "result": result,
                        "status": status.name,
                        "initial_layout": initial_layout,
                        "n_multiple": n_multiple,
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "parameters": list(parameters),
                        "backend_name": backend_name,
                        "optimization_level": optimization_level,
                        "backend_properties": properties.to_dict(),
                        "combined_result": combined,
                        "backend_configuration": configuration.to_dict(),
                    },
                    sort_keys=True,
                    indent=4,
                    cls=DateTimeEncoder,
                ),
            )
    return job


def get_coupling_layouts(
    backend: Union[Backend, str],
    n_qubits: int,
    exclude_qubits: Optional[List] = None,
) -> List[List[int]]:
    if isinstance(backend, str):
        backend = get_backend(backend)
    configuration = backend.configuration()
    adj = defaultdict(list)
    for i, j in configuration.coupling_map:
        adj[i].append(j)

    layouts = []
    current_path = []
    visited = set()
    if exclude_qubits is not None:
        for q in exclude_qubits:
            visited.add(q)

    def dfs(q):
        if q in visited:
            return
        visited.add(q)
        current_path.append(q)
        if len(current_path) == n_qubits:
            layouts.append(list(current_path))
        else:
            for n in adj[q]:
                dfs(n)
        current_path.pop()
        visited.remove(q)

    for s in range(configuration.n_qubits):
        dfs(s)

    return layouts
