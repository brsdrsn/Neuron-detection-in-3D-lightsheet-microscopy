# Dock widget
from pathlib import Path
from typing import Annotated

import napari
import dask.array as da
from magicgui import magic_factory
from napari.types import ImageData
from zarr.errors import ArrayNotFoundError as ZarrArrayNotFoundError
from ._tiling import load_zarr, tile_array, get_tile, num_tiles, process_data


class TileManager:
    """Stores tiling state so the widget can navigate tiles."""
    def __init__(self):
        self.data = None          # dask array of full dataset
        self.tiles = None         # rechunked array
        self.current_idx = (0,0,0)
        self.tile_shape = None
        self.numblocks = None
        self.layer = None

tile_state = TileManager()

def tile_name_from_idx(idx):
    z, y, x = idx
    return f"Tile {z}-{y}-{x}"

@magic_factory(call_button="Start Tiling")
def tiler_widget(
    zarr_path: Annotated[Path, {"mode": "d", "label": "Zarr directory"}] = Path(),
    tile_x: int = 256,
    tile_y: int = 256,
    tile_z: int = 256,
    viewer: "napari.viewer.Viewer" = None,
):
    """
    Napari UI widget for tiling and navigating large Zarr images.
    """
    path = Path(zarr_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    try:
        full_data = process_data(load_zarr(str(path)))
    except ZarrArrayNotFoundError:
        raise ValueError(
            f"No zarr array found at {path}. Please select a zarr directory "
            "(e.g. your_dataset.zarr or a scale level like …/dataset.zarr/0)."
        ) from None

    # ---- load data ----
    tile_state.data = full_data
    tile_state.tiles = tile_array(full_data, (tile_z, tile_y, tile_x))
    tile_state.numblocks = num_tiles(tile_state.tiles)

    # ---- first tile ----
    tile_state.current_idx = (0, 0, 0)
    first_tile = get_tile(tile_state.tiles, tile_state.current_idx)

    viewer.layers.clear()
    layer = viewer.add_image(
        first_tile,
        name=tile_name_from_idx(tile_state.current_idx),
    )

    tile_state.layer = layer

    return None

@magic_factory(call_button="Next Tile")
def next_tile(viewer: "napari.viewer.Viewer"):
    z, y, x = tile_state.current_idx
    bz, by, bx = tile_state.numblocks

    # move in X direction first
    x = x + 1
    if x >= bx:
        x = 0
        y += 1
    if y >= by:
        y = 0
        z += 1
    if z >= bz:
        z = 0  # wraparound

    tile_state.current_idx = (z, y, x)
    tile = get_tile(tile_state.tiles, tile_state.current_idx)

    layer = tile_state.layer or viewer.layers[0]
    layer.data = tile
    layer.name = tile_name_from_idx(tile_state.current_idx)

    return None

@magic_factory(call_button="Previous Tile")
def prev_tile(viewer: "napari.viewer.Viewer"):
    z, y, x = tile_state.current_idx
    bz, by, bx = tile_state.numblocks

    # inverse navigation
    x -= 1
    if x < 0:
        x = bx - 1
        y -= 1
    if y < 0:
        y = by - 1
        z -= 1
    if z < 0:
        z = bz - 1

    tile_state.current_idx = (z, y, x)
    tile = get_tile(tile_state.tiles, tile_state.current_idx)

    layer = tile_state.layer or viewer.layers[0]
    layer.data = tile
    layer.name = tile_name_from_idx(tile_state.current_idx)

    return None
