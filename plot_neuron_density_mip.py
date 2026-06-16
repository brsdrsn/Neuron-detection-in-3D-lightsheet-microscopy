"""
Pipeline stage: 7 — analysis / reporting
========================================
Plot a top-down XY neuron density map (MIP over Z) from inference output CSVs.

Auto-detects column schema (axis-0/1/2 or x/y/z) and histograms neuron centers
into a 2D density image with optional log scaling.

Inputs:  neuron_centers_*_raw_axes.csv (or global_neuron_centers_*.csv)
Outputs: matplotlib figure (interactive show)

Run:
  python plot_neuron_density_mip.py
  # or import plot_neuron_density_mip_xy_from_raw_axes() from another script

Suggested rename: plot_neuron_density_xy_mip.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_neuron_density_mip_xy_from_raw_axes(
    csv_path: str | Path,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
    bins: tuple[int, int] = (512, 512),
    cmap: str = "inferno",
    log_scale: bool = True,
    title: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build and display a top-down XY neuron density map by collapsing all Z planes.

    Parameters
    ----------
    csv_path
        Path to neuron centers CSV (e.g. neuron_centers_chunk_1_all_tiles_raw_axes.csv).
    x_col, y_col, z_col
        Optional column names for X, Y and Z. If omitted, the function auto-detects:
        - for axis-* CSVs, Z is the smallest-range axis; X/Y are the other two
        - for x/y/z CSVs, uses x and y directly
    bins
        Number of bins for (X, Y) histogram.
    cmap
        Matplotlib colormap.
    log_scale
        If True, display log1p(density) for better contrast.
    title
        Optional plot title.

    Returns
    -------
    density_xy, x_edges, y_edges
        2D density map and histogram edges from numpy.histogram2d.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Auto-detect whether CSV uses Napari axis-* columns or x/y/z columns
    if x_col is None or y_col is None:
        axis_cols = ["axis-0", "axis-1", "axis-2"]
        if set(axis_cols).issubset(df.columns):
            # Determine which raw axis is Z by choosing the smallest coordinate range.
            axis_ranges = {}
            for col in axis_cols:
                c = pd.to_numeric(df[col], errors="coerce").dropna()
                if c.empty:
                    raise ValueError(f"Column '{col}' has no numeric values.")
                axis_ranges[col] = float(c.max() - c.min())

            # Z is the axis with smallest range (depth is usually thinnest in lightsheet data)
            if z_col is None:
                z_col = min(axis_ranges, key=axis_ranges.get)

            xy_cols = [c for c in axis_cols if c != z_col]
            # Assign X as the wider in-plane dimension.
            xy_cols_sorted = sorted(xy_cols, key=lambda c: axis_ranges[c], reverse=True)
            x_col, y_col = xy_cols_sorted[0], xy_cols_sorted[1]
        elif {"x", "y"}.issubset(df.columns):
            x_col = "x"
            y_col = "y"
        else:
            raise ValueError(
                "Could not auto-detect XY columns. Expected either "
                "['axis-0','axis-1','axis-2'] or ['x','y']. "
                f"Available columns: {list(df.columns)}"
            )

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{x_col}' and '{y_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    xy = df[[x_col, y_col]].dropna()
    if xy.empty:
        raise ValueError("No valid (x, y) points found in CSV after dropping NaNs.")

    x = xy[x_col].to_numpy(dtype=float)
    y = xy[y_col].to_numpy(dtype=float)

    # 2D histogram of all neuron centers projected onto XY (MIP over Z)
    density_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    image = np.log1p(density_xy.T) if log_scale else density_xy.T

    plt.figure(figsize=(8, 7))
    plt.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        title
        or f"Neuron Density Top-Down MIP (XY, collapsed over Z)\n{csv_path.name}"
    )
    cbar = plt.colorbar()
    cbar.set_label("log1p(count)" if log_scale else "count")
    plt.tight_layout()
    plt.show()

    return density_xy, x_edges, y_edges


if __name__ == "__main__":
    plot_neuron_density_mip_xy_from_raw_axes(
        r"C:\Users\pc\Documents\neurons\Napari\outputs\chunk 2\global_coords\neuron_centers_chunk_1_all_tiles_raw_axes.csv"
    )