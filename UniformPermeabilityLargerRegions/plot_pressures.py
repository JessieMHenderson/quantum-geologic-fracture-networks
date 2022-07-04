from pitch_fork.ibmq import get_backend
from pitch_fork import visualization
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import pathlib


def get_vector(r):
    quantum_vector = np.zeros(2 ** r["n_qubits"])
    if "combined_result" in r:
        res = r["combined_result"]
    else:
        res = r["unified_result"]
    for k, v in res.items():
        quantum_vector[int(k.replace("0x", ""), 16)] += v
    quantum_vector = quantum_vector / np.sum(quantum_vector)
    return np.sqrt(quantum_vector)


def do(
    x_true: np.ndarray,
    source_dir: pathlib.Path,
    target_dir: pathlib.Path,
    add_fidelity: bool,
    add_borders: bool,
) -> None:
    scale = np.sqrt(np.sum(x_true * x_true))
    x_true = x_true / scale
    target_dir.mkdir(parents=True, exist_ok=True)

    for r in source_dir.glob("*.json"):
        with r.open() as f:
            res = json.load(f)
        v = get_vector(res)
        f = np.sum(x_true * v) ** 2
        title = None
        if add_fidelity:
            title = f"Fidelity: {f:0.4}"
        fig = visualization.plot_pressures(
            v * scale,
            title=title,
            add_border=add_borders,
        )
        fig_path = target_dir.joinpath(r.name.replace(".json", ".pdf"))
        fig.savefig(
            str(fig_path),
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pressures")
    parser.add_argument(
        "--x-true",
        dest="x_true_file",
        required=True,
        type=argparse.FileType("rb"),
        help="npy file with x_true value",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=pathlib.Path,
        help="source directory of results",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=pathlib.Path,
        help="target directory for plots",
    )
    parser.add_argument(
        "--add-fidelity",
        default=False,
        const=True,
        action="store_const",
        help="add fidelity to title",
    )
    parser.add_argument(
        "--add-borders",
        default=False,
        const=True,
        action="store_const",
        help="add borders 0 and 1",
    )
    args = parser.parse_args()
    x_true = np.load(args.x_true_file)
    args.x_true_file.close()
    assert args.source_dir.exists()
    assert args.source_dir.is_dir()
    if args.target_dir.exists():
        assert args.target_dir.is_dir()
    do(
        x_true,
        args.source_dir,
        args.target_dir,
        args.add_fidelity,
        args.add_borders,
    )
