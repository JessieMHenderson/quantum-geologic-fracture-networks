import logging
import argparse
import numpy as np
import pathlib
from pitch_fork import fidelity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fidelity optimization: argmin 1-|<x(theta)|x_true>|^2"
    )
    parser.add_argument(
        "--x-true",
        dest="x_true_file",
        default=None,
        type=argparse.FileType("rb"),
        help="npy file with x_true value",
    )
    parser.add_argument(
        "--layers",
        default=None,
        type=int,
        help="number of ansatz layers",
    )
    parser.add_argument(
        "--max-iter",
        default=200,
        type=int,
        help="number of optimization iterations",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        type=pathlib.Path,
        help="working directory to save results",
    )
    parser.add_argument("--run-id", default=None, type=str, help="run-id")
    parser.add_argument(
        "--extra-iters",
        default=0,
        type=int,
        help="extra iteration used in continue mode",
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        action="store_const",
        const=True,
        default=False,
        help="continue run-id in working-dir",
    )
    parser.add_argument("--log", default="warning", help="logging level")

    args = parser.parse_args()
    logging.basicConfig(level=args.log.upper())

    if args.continue_:
        fidelity.optimize(
            result_dir=args.working_dir,
            run_id=args.run_id,
            continue_=args.continue_,
        )
    else:
        x_true = np.load(args.x_true_file)
        x_true = x_true / np.sqrt(np.abs(np.sum(x_true * x_true)))
        args.x_true_file.close()
        n_qubits = int(np.round(np.log2(len(x_true))))
        if args.layers is None:
            n_layers = n_qubits
        else:
            n_layers = args.layers

        fidelity.optimize(
            x_true,
            n_qubits=n_qubits,
            n_layers=n_layers,
            max_iter=args.max_iter,
            result_dir=str(args.working_dir),
            run_id=args.run_id,
        )
