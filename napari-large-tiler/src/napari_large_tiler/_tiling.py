#Tiling utilities

import dask.array as da
import numpy as np
from dask.array import Array

def to_zyx(vol: np.ndarray) -> np.ndarray:
    """
    Ensure volume is (Z,Y,X). Heuristic: assume Z is the smallest dimension.
    """
    assert vol.ndim == 3
    z_axis = int(np.argmin(vol.shape))
    if z_axis == 0:
        return vol  # already (Z,Y,X)
    if z_axis == 1:
        return np.moveaxis(vol, 1, 0)  # (Y,Z,X) -> (Z,Y,X)
    return np.moveaxis(vol, 2, 0)      # (Y,X,Z) -> (Z,Y,X)

def load_zarr(path):
    """Load a zarr file into a Dask array."""
    return da.from_zarr(path)

def process_data(data):
    """Processes the Dask array. This makes the array shape same as the native nninteractive array shape(4 dim)."""
    data2 = data[0, 0, :, :, :]
    data_fixed = data2.map_blocks(lambda x: x.astype(x.dtype.newbyteorder("=")))
    # enforce canonical axis order
    data_fixed = to_zyx(data_fixed)
    return data_fixed

def select_scale(zarr_group, scale_level):
    """Return the dask array for the chosen multiscale level."""
    return da.from_zarr(zarr_group[scale_level])

def tile_array(arr: Array, tile_size=(256, 256, 256)):
    """Return a tiled dask array."""
    return arr.rechunk(tile_size)

def get_tile(arr: Array, index_tuple):
    """Get a single tile using block indexing."""
    return arr.blocks[index_tuple]

def num_tiles(arr: Array):
    return arr.numblocks


if __name__ == "__main__":
    zarr_path = r"C:\Users\brsdr\Documents\neurons\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\1"
    data = load_zarr(zarr_path)
    data = process_data(data)
    tiles = tile_array(data)
    print(get_tile(tiles, (3, 3, 0)).shape)