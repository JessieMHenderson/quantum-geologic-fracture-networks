import os
import matplotlib.pyplot as plt
from pitch_fork import visualization
import numpy as np
import argparse
import pathlib


def do(
    x_true: np.ndarray,
    source_dir: pathlib.Path,
    target_dir: pathlib.Path,
    add_fidelity: bool,
) -> None:
    x_true = x_true / np.sqrt(np.sum(x_true * x_true))
    target_dir.mkdir(parents=True, exist_ok=True)
    if add_fidelity:
        title_format = "Fidelity: {fidelity:0.4f}"
    else:
        title_format = None

    for r in source_dir.glob("*.json"):
        fig = visualization.plot_coupling_layout_with_errors(
            str(r),
            x_true,
            title_format=title_format,
            figsize=(30, 4),
        )
        fig_path = target_dir.joinpath(r.name.replace(".json", ".pdf"))
        fig.savefig(
            str(fig_path),
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot errors")
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
    )
