from pitch_fork import ibmq
import numpy as np
import qiskit
import pathlib
import argparse


def do(
    x0: np.ndarray,
    backend: str,
    n_qubits: int,
    n_layers: int,
    n_shots: int,
    working_dir: pathlib.Path,
) -> None:
    if working_dir.exists():
        assert working_dir.is_dir()
    working_dir.mkdir(parents=True, exist_ok=True)
    qiskit.IBMQ.load_account()
    for layout in ibmq.get_coupling_layouts(backend, n_qubits):
        ibmq.run_with_qiskit_runtime(
            x0,
            n_qubits,
            n_layers,
            backend,
            n_shots,
            result_dir=str(working_dir),
            initial_layout=layout,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run on IBM Q using Runtime all coupling layouts of backend"
    )
    parser.add_argument(
        "--ansatz",
        dest="ansatz_params",
        required=True,
        type=argparse.FileType("rb"),
        help="file with ansatz params",
    )
    parser.add_argument(
        "--layers",
        required=True,
        type=int,
        help="number of ansatz layers",
    )
    parser.add_argument(
        "--qubits",
        required=True,
        type=int,
        help="number of qubits",
    )
    parser.add_argument(
        "--shots",
        default=10 ** 5,
        type=int,
        help="number of shots",
    )
    parser.add_argument(
        "--backend",
        required=True,
        type=str,
        help="backend to run on",
    )
    parser.add_argument(
        "--working-dir",
        required=True,
        type=pathlib.Path,
        help="working directory to save results",
    )
    args = parser.parse_args()
    x0 = np.load(args.ansatz_params)
    args.ansatz_params.close()
    do(
        x0,
        args.backend,
        args.qubits,
        args.layers,
        args.shots,
        args.working_dir,
    )
