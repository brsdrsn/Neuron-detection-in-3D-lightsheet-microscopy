"""
Pipeline stage: — one-off migration (likely obsolete)
=====================================================
Batch-convert early manual labels from (Y, X, Z) axis order to canonical (Z, Y, X).

Renames files so tile coordinates in filenames follow Z-Y-X instead of Y-X-Z,
and transposes array data with np.moveaxis.

Inputs:  ~/Documents/neurons/tagged tiles/All positives old/*.tif
Outputs: ~/Documents/neurons/tagged tiles/All positives new/*.tif

Safe to archive once all labels use the Z-Y-X convention used by napari-large-tiler.

Run:     python label_batch_converter.py
Suggested rename: migrate_labels_yxz_to_zyx.py
"""
import re
from pathlib import Path

import numpy as np
import tifffile as tiff


def yxz_to_zyx(arr):
    """Transpose array data from mistaken (Y,X,Z) labeling to canonical (Z,Y,X)."""
    # (Y,X,Z) -> (Z,Y,X)
    return np.moveaxis(arr, 2, 0)


def tile_name_yxz_to_zyx(name: str) -> str:
    """
    Convert filename tile coords (Y,X,Z) -> (Z,Y,X).
    e.g. "object 1 - Tile 0-5-1" -> "object 1 - Tile 1-0-5"
    """
    # Match "object N - Tile Y-X-Z" (Y, X, Z are digits)
    m = re.match(r"(.*\bTile\s+)(\d+)-(\d+)-(\d+)(.*)", name, re.IGNORECASE)
    if m is None:
        return name  # no change if pattern doesn't match
    prefix, y, x, z, suffix = m.groups()
    # (Z,Y,X) order in the new name
    new_name = f"{prefix}{z}-{y}-{x}{suffix}"
    return new_name


neurons_root = Path.home() / "Documents" / "neurons"
# One-time migration: read old-axis masks, write corrected Z-Y-X masks
in_dir = neurons_root / "tagged tiles" / "All positives old"
out_dir = neurons_root / "tagged tiles" / "All positives new"
out_dir.mkdir(parents=True, exist_ok=True)

for p in in_dir.glob("*.tif*"):
    lab = tiff.imread(p)
    if lab.ndim != 3:
        print("Skipping non-3D:", p.name, lab.shape)
        continue
    lab2 = yxz_to_zyx(lab)
    out_name = tile_name_yxz_to_zyx(p.name)
    tiff.imwrite(out_dir / out_name, lab2, compression="zlib")
    if out_name != p.name:
        print(f"  {p.name} -> {out_name}")
