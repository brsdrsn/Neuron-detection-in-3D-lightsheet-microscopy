"""
Pipeline stage: 6 — visualize inference results
=================================================
Load a processed Zarr chunk and precomputed global neuron centers (.npy) in Napari
for manual inspection of infer_chunk_global_coords.py output.

Inputs:  --zarr-path (required)
         --points-npy (required, shape N×3 in Z,Y,X)
Outputs: interactive Napari window

Run:
  python view_global_points_on_chunk.py --zarr-path "C:\\...\\1" --points-npy outputs/chunk_1/global_coords/global_neuron_centers_chunk_1_all_tiles.npy

Suggested rename: view_neuron_centers_on_chunk_napari.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import napari
import numpy as np

# ---- make local plugin importable (editable install may not be on PYTHONPATH) ----
_REPO_ROOT = Path(__file__).resolve().parent
_PLUGIN_SRC = _REPO_ROOT / "napari-large-tiler" / "src"
if _PLUGIN_SRC.is_dir():
    sys.path.insert(0, str(_PLUGIN_SRC))

from napari_large_tiler._tiling import (  # type: ignore  # noqa: E402
    load_zarr,
    process_data,
)


def load_zarr_volume(path_with_optional_key: str) -> np.ndarray:
    """Load and preprocess a Zarr chunk to (Z,Y,X) numpy array for Napari display."""
    vol = process_data(load_zarr(path_with_optional_key))
    if getattr(vol, "ndim", 0) != 3:
        raise ValueError(f"Expected processed 3D volume (Z,Y,X). Got shape {vol.shape}")
    return np.asarray(vol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View global neuron center points on the original chunk in Napari."
    )
    parser.add_argument(
        "--zarr-path",
        required=True,
        help=r'Chunk Zarr path (can include dataset key), e.g. "...fused.zarr\...\fused.zarr\1".',
    )
    parser.add_argument(
        "--points-npy",
        required=True,
        help="Path to .npy points array with shape (N,3) in Z,Y,X order.",
    )
    parser.add_argument("--points-size", type=float, default=3.0, help="Napari point size.")
    parser.add_argument("--image-name", default="chunk", help="Napari image layer name.")
    parser.add_argument("--points-name", default="global neuron centers", help="Napari points layer name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vol = load_zarr_volume(args.zarr_path)

    points_path = Path(args.points_npy)
    if not points_path.exists():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    points = np.load(points_path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points array shape (N,3). Got {points.shape}")

    # Points must be in the same processed (Z,Y,X) axis order as the displayed volume
    points = points.astype(np.float32, copy=False)

    viewer = napari.Viewer()
    viewer.add_image(vol, name=args.image_name, rendering="mip")
    viewer.add_points(points, size=args.points_size, name=args.points_name)

    print(f"Loaded chunk shape: {vol.shape}")
    print(f"Loaded points: {points.shape[0]}")
    napari.run()


if __name__ == "__main__":
    main()
