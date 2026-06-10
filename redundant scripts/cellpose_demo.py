"""
Pipeline stage: — experiment (alternate segmentation path)
==========================================================
Prototype 3D Cellpose segmentation on Zarr tiles. Explored before settling on the
YOLO bounding-box pipeline. Requires ~8 GB+ GPU; imports helpers from main.py.

Not part of the production neuron-detection workflow. Keep for reference or delete
once YOLO results are satisfactory.

Inputs:  hard-coded Zarr path (same as main.py)
Outputs: optional cellpose_masks.npy; optional Napari viewer

Run:     python cellpose_demo.py
Suggested rename: experiment_cellpose_segmentation.py
"""

import napari
import dask.array as da
import numpy as np
import os
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from cellpose import models
from main import open_in_napari, get_dimentionality, process_data, tile

zarr_path = str(
    Path.home()
    / "Documents"
    / "neurons"
    / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr"
    / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr"
    / "1"
)
data = da.from_zarr(zarr_path)
# tile: numpy array shaped (Z, Y, X), dtype float32/uint16


#tile = data  # <-- get one tile from your plugin / loader


def cellpose_demo(tile, show=True, show_progress=True, save_path=None):
    t0 = perf_counter()
    if show_progress:
        print("cellpose: preparing input...")
    # Ensure numpy array (cellpose expects numpy, not dask)
    if hasattr(tile, "compute"):
        tile = tile.compute()
    else:
        tile = np.asarray(tile)

    # Basic normalization (lightweight)
    tile_f = tile.astype(np.float32)
    tile_f -= tile_f.min()
    if tile_f.max() > 0:
        tile_f /= tile_f.max()

    # Create model
    # Cellpose v3 exposes CellposeModel (not Cellpose) in cellpose.models
    model = models.CellposeModel(gpu=True, model_type="cyto")  # try "nuclei" too

    # Key params you’ll tune later:
    # - diameter: soma diameter in pixels in XY; use None first if unsure
    # - anisotropy: dz/dx if z spacing is larger than x/y spacing
    if show_progress:
        print("cellpose: running model...")
    eval_out = model.eval(
        tile_f,
        channel_axis=None,  # grayscale volume: (Z, Y, X)
        z_axis=0,
        do_3D=True,
        diameter=None,      # set later
        anisotropy=4        # set later (e.g., 3–6 for lightsheet often)
    )
    # Cellpose v4 returns 3 values; older versions returned 4
    if len(eval_out) == 4:
        masks, flows, styles, diams = eval_out
    else:
        masks, flows, styles = eval_out

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, masks.astype(np.int32))
        if show_progress:
            print(f"cellpose: masks saved to {save_path}")

    if show:
        if show_progress:
            print("cellpose: displaying in napari...")
        # Visualize in napari
        viewer = napari.Viewer()
        viewer.add_image(tile, name="tile", contrast_limits=[np.percentile(tile, 1), np.percentile(tile, 99)])
        viewer.add_labels(masks.astype(np.int32), name="cellpose_masks")
        napari.run()

    if show_progress:
        print(f"cellpose: done in {perf_counter() - t0:.1f}s")
    return masks


def run_cellpose_on_tiles(tiles):
    total_tiles = int(np.prod(tiles.numblocks))
    results = {}
    for index in tqdm(np.ndindex(*tiles.numblocks), total=total_tiles, desc="Cellpose tiles"):
        tile = tiles.blocks[index]
        masks = cellpose_demo(tile, show=False)
        results[index] = masks
    return results

processed_data = process_data(data)
get_dimentionality(processed_data)#Array shape: (2103, 1441, 124), Chunk size: (256, 256, 124)
#mip(processed_data)
tiles = tile(processed_data)
print("Number of tiles:", tiles.numblocks)
# first_tile = tiles.blocks[3, 3, 0]
first_tile = tiles.blocks[2, 2, 0]
print("tile shape:", first_tile.shape) #tile shape: (128, 512, 124)
PROCESS_ALL_TILES = False
MASKS_SAVE_PATH = "cellpose_masks.npy"
#open_in_napari(first_tile)
print("starting cellpose demo...")
if PROCESS_ALL_TILES:
    run_cellpose_on_tiles(tiles)
else:
    cellpose_demo(first_tile, save_path=MASKS_SAVE_PATH)