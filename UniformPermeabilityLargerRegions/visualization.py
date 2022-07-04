from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.gridspec as gridspec


def plot_pressures(
    x: np.ndarray,
    add_border: bool = False,
    title: Optional[str] = None,
    cmap: str = "plasma",
    style: str = "default",
    add_colorbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: float = 100.0,
) -> mpl.figure.Figure:
    side_length = int(np.round(np.sqrt(len(x) / 2)))
    x = x.reshape(side_length, side_length * 2).T
    if add_border:
        x = np.concatenate(
            (np.ones((side_length * 2, 1)), x, np.zeros((side_length * 2, 1))), axis=1
        )
    with mpl.style.context(style):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(x, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        if add_colorbar:
            fig.colorbar(im)
        ax.invert_yaxis()
    return fig


def plot_coupling_layout_with_errors(
    result_path: str,
    true: Optional[np.ndarray] = None,
    title_format: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 2),
    dpi: float = 100.0,
) -> mpl.figure.Figure:
    # following implementation of qiskit plot_error_map
    with open(result_path, "r") as f:
        r = json.load(f)

    n_qubits = r["n_qubits"]
    if r["initial_layout"] is None:
        r["initial_layout"] = n_qubits

    read_err = [0 for _ in r["initial_layout"]]
    for i, qubit in enumerate(r["initial_layout"]):
        for item in r["backend_properties"]["qubits"][qubit]:
            if item["name"] == "readout_error":
                read_err[i] = item["value"]
    read_err = np.asarray(read_err) * 100
    if len(read_err) != n_qubits:
        raise ValueError()

    cx_errors = []
    for line in zip(r["initial_layout"][:-1], r["initial_layout"][1:]):
        for item in r["backend_properties"]["gates"]:
            if item["qubits"][0] == line[0] and item["qubits"][-1] == line[-1]:
                cx_errors.append(item["parameters"][0]["value"])
                break
        else:
            continue
    cx_errors = 100 * np.asarray(cx_errors)
    if len(cx_errors) != n_qubits - 1:
        raise ValueError()

    color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)
    cx_norm = mpl.colors.Normalize(vmin=min(cx_errors), vmax=max(cx_errors))
    line_colors = [color_map(cx_norm(err)) for err in cx_errors]
    read_error_norm = mpl.colors.Normalize(vmin=min(read_err), vmax=max(read_err))
    q_colors = [color_map(read_error_norm(err)) for err in read_err]

    fig, _ = plt.subplots(figsize=figsize, dpi=dpi)
    gridspec.GridSpec(nrows=2, ncols=1)
    grid_spec = gridspec.GridSpec(2, 3, height_ratios=[9, 1], width_ratios=[15, 1, 15])
    main_ax = plt.subplot(grid_spec[:1, :])
    for ind, (i, j) in enumerate(zip(range(n_qubits - 1), range(1, n_qubits))):
        main_ax.add_artist(
            plt.Line2D(
                [i, j],
                [0.5, 0.5],
                color=line_colors[ind],
                linewidth=7,
                zorder=0,
            )
        )
        for i in range(n_qubits):
            _idx = (i, 0.5)
            main_ax.add_artist(
                mpatches.Ellipse(
                    _idx,
                    25 / 48,
                    25 / 48,
                    color=q_colors[i],
                    zorder=1,
                )
            )
            main_ax.text(
                *_idx,
                s=str(r["initial_layout"][i]),
                horizontalalignment="center",
                verticalalignment="center",
                color="w",
                size=12,
                weight="bold",
            )
    main_ax.set_xlim([-1, n_qubits + 1])
    main_ax.set_ylim([-1, 1])
    main_ax.set_aspect("equal")
    main_ax.axis("off")
    bright_ax = plt.subplot(grid_spec[-1, 2])

    cx_cb = mpl.colorbar.ColorbarBase(
        bright_ax, cmap=color_map, norm=cx_norm, orientation="horizontal"
    )
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cx_cb.locator = tick_locator
    cx_cb.update_ticks()
    bright_ax.set_title(f"CNOT error rate (%) [Avg. = {round(np.mean(cx_errors), 3)}]")

    bleft_ax = plt.subplot(grid_spec[-1, 0])
    readout_error_cb = mpl.colorbar.ColorbarBase(
        bleft_ax, cmap=color_map, norm=read_error_norm, orientation="horizontal"
    )
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    readout_error_cb.locator = tick_locator
    readout_error_cb.update_ticks()
    bleft_ax.set_title(f"Readout error (%) [Avg. = {round(np.mean(read_err), 3)}]")

    if true is not None:
        quantum_vector = np.zeros(2 ** n_qubits)
        if "combined_result" in r:
            res = r["combined_result"]
        else:
            res = r["unified_result"]
        for k, v in res.items():
            quantum_vector[int(k.replace("0x", ""), 16)] += v
        quantum_vector = quantum_vector / np.sum(quantum_vector)
        fidelity = np.abs(np.sum(np.sqrt(quantum_vector) * true)) ** 2
    if title_format is not None:
        title = title_format.format(fidelity=fidelity)
        main_ax.set_title(title)
    return fig
