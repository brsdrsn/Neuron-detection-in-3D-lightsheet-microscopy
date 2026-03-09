"""
Script to convert the labeled tiles to the new format (Y,X,Z) -> (Z,Y,X).
"""
import re
from pathlib import Path

import numpy as np
import tifffile as tiff


def yxz_to_zyx(arr):
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


in_dir = Path(r"C:\Users\brsdr\Documents\neurons\tagged tiles\All positives old")
out_dir = Path(r"C:\Users\brsdr\Documents\neurons\tagged tiles\All positives new")
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
