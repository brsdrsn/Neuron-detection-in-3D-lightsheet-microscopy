# Core tiling utilities shared by the Napari plugin and pipeline scripts.
# All production code uses (Z, Y, X) axis order after process_data().

import dask.array as da
import numpy as np
from dask.array import Array
from pathlib import Path


def to_zyx(vol: np.ndarray) -> np.ndarray:
    """
    Ensure volume is (Z,Y,X). Heuristic: assume Z is the smallest dimension.
    Lightsheet volumes are typically much thinner in Z than in Y/X.
    """
    assert vol.ndim == 3
    z_axis = int(np.argmin(vol.shape))
    if z_axis == 0:
        return vol  # already (Z,Y,X)
    if z_axis == 1:
        return np.moveaxis(vol, 1, 0)  # (Y,Z,X) -> (Z,Y,X)
    return np.moveaxis(vol, 2, 0)      # (Y,X,Z) -> (Z,Y,X)


def load_zarr(path):
    """Load a Zarr store (or dataset key path) into a lazy Dask array."""
    return da.from_zarr(path)


def process_data(data):
    """
    Preprocess raw Zarr data for tiling and display.

    Steps:
      1. Index into the 5D array (drops leading singleton dims from multiscale layout)
      2. Fix byte order for cross-platform compatibility
      3. Reorder axes to canonical (Z, Y, X) via to_zyx()
    """
    data2 = data[0, 0, :, :, :]
    data_fixed = data2.map_blocks(lambda x: x.astype(x.dtype.newbyteorder("=")))
    # enforce canonical axis order
    data_fixed = to_zyx(data_fixed)
    return data_fixed


def select_scale(zarr_group, scale_level):
    """Return the dask array for a chosen multiscale pyramid level."""
    return da.from_zarr(zarr_group[scale_level])


def tile_array(arr: Array, tile_size=(256, 256, 256)):
    """
    Split a Dask array into a grid of fixed-size blocks.
    tile_size is (Z, Y, X); default 256³ in UI, 512³ in production scripts.
    """
    return arr.rechunk(tile_size)


def get_tile(arr: Array, index_tuple):
    """Extract one tile by block index (tz, ty, tx) without loading the full volume."""
    return arr.blocks[index_tuple]


def num_tiles(arr: Array):
    """Return (num_blocks_z, num_blocks_y, num_blocks_x) for the rechunked array."""
    return arr.numblocks


if __name__ == "__main__":
    # Quick smoke test: load a chunk and print one tile's shape
    zarr_path = str(
        Path.home()
        / "Documents"
        / "neurons"
        / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr"
        / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr"
        / "1"
    )
    data = load_zarr(zarr_path)
    data = process_data(data)
    tiles = tile_array(data)
    print(get_tile(tiles, (3, 3, 0)).shape)
