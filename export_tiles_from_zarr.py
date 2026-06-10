"""
Pipeline stage: 1 — tile export
===============================
Rebuild `Tile z-y-x.tif` volumes from original Zarr chunks for manual labeling.

Writes one multi-page TIFF per tagged-tile folder under ~/Documents/neurons/tagged
tiles/{1st,2nd} chunk/. Uses the same tiling logic as the napari-large-tiler plugin.

This matches the Large Image Tiler plugin semantics:
- folder name: `tile z-y-x`
- tile name: `Tile z-y-x.tif`
- indexing: `tiles.blocks[(z, y, x)]` after `tile_array(volume, (tile_z, tile_y, tile_x))`
- preprocessing: `process_data(load_zarr(zarr_path))` (includes `to_zyx` like the plugin)

Examples (PowerShell):

Dry-run (no zarr read, no writes):
  & .\\napari-env\\Scripts\\python.exe .\\rebuild_tiles.py --chunk first --dry-run

Rebuild one tile in 2nd chunk:
  & .\\napari-env\\Scripts\\python.exe .\\rebuild_tiles.py --chunk second --only-tile 0-1-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tifffile as tiff

# ---- make local plugin importable (editable install may not be on PYTHONPATH) ----
_REPO_ROOT = Path(__file__).resolve().parent
_PLUGIN_SRC = _REPO_ROOT / "napari-large-tiler" / "src"
if _PLUGIN_SRC.is_dir():
    sys.path.insert(0, str(_PLUGIN_SRC))

from napari_large_tiler._tiling import (  # type: ignore  # noqa: E402
    get_tile,
    load_zarr,
    num_tiles,
    process_data,
    tile_array,
)


DEFAULT_JOBS: list[tuple[str, Path, Path]] = [
    (
        "1st chunk",
        Path.home()
        / "Documents"
        / "neurons"
        / "tagged tiles"
        / "1st chunk",
        Path(
            r"C:\Users\pc\Documents\neurons\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr"
            r"\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\1"
        ),
    ),
    (
        "2nd chunk",
        Path.home()
        / "Documents"
        / "neurons"
        / "tagged tiles"
        / "2nd chunk",
        Path(r"C:\Users\pc\Documents\neurons\2025_12_11_DiI3_540_595_MFS2_2x2_50ms_fused.zarr\1"),
    ),
]


def parse_tile_id(folder_name: str) -> tuple[int, int, int] | None:
    if not folder_name.startswith("tile "):
        return None
    rest = folder_name[len("tile ") :].strip()
    parts = rest.split("-")
    if len(parts) != 3:
        return None
    try:
        z, y, x = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    return z, y, x


def discover_tile_dirs(tagged_root: Path) -> list[Path]:
    return sorted(p for p in tagged_root.iterdir() if p.is_dir() and p.name.startswith("tile "))


def validate_written_tiff(path: Path, expected_z: int) -> int:
    with tiff.TiffFile(str(path)) as tf:
        n = len(tf.pages)
    if n != expected_z:
        print(f"WARNING: {path.name} page count {n} != expected Z {expected_z}")
    return n


def rebuild_job(
    job_label: str,
    tagged_root: Path,
    zarr_path: Path,
    tile_shape: tuple[int, int, int],
    only_tile: str | None,
    dry_run: bool,
    compression: str | None,
) -> None:
    if not tagged_root.exists():
        raise FileNotFoundError(f"[{job_label}] tagged root missing: {tagged_root}")
    if not zarr_path.exists():
        raise FileNotFoundError(f"[{job_label}] zarr path missing: {zarr_path}")

    print(f"\n=== {job_label} ===")
    print(f"zarr: {zarr_path}")
    print(f"tagged: {tagged_root}")
    print(f"tile_shape (Z,Y,X): {tile_shape}")

    tile_dirs = discover_tile_dirs(tagged_root)
    if only_tile:
        want = f"tile {only_tile.strip()}"
        tile_dirs = [p for p in tile_dirs if p.name == want]
        if not tile_dirs:
            raise RuntimeError(f"No folder named '{want}' under {tagged_root}")

    if dry_run:
        print("dry-run: skipping zarr load + writes")
        for td in tile_dirs:
            idx = parse_tile_id(td.name)
            if idx is None:
                print(f"dry-run skip (unrecognized): {td.name}")
                continue
            z, y, x = idx
            out_path = td / f"Tile {z}-{y}-{x}.tif"
            print(f"dry-run: would rebuild {td.name} -> {out_path.name} using blocks[({z},{y},{x})]")
        return

    full = process_data(load_zarr(str(zarr_path)))
    tiles = tile_array(full, tile_shape)
    bz, by, bx = num_tiles(tiles)
    print(f"volume shape (Z,Y,X): {full.shape}")
    print(f"numblocks (Z,Y,X): {(bz, by, bx)}")

    for td in tile_dirs:
        idx = parse_tile_id(td.name)
        if idx is None:
            print(f"skip (unrecognized folder name): {td.name}")
            continue
        z, y, x = idx
        if z < 0 or y < 0 or x < 0 or z >= bz or y >= by or x >= bx:
            raise RuntimeError(
                f"Tile index out of range for {td.name}: (z,y,x)={(z, y, x)} "
                f"but numblocks={(bz, by, bx)}"
            )

        out_path = td / f"Tile {z}-{y}-{x}.tif"
        tmp_path = td / f"Tile {z}-{y}-{x}.tmp.tif"

        print(f"rebuilding {td.name} -> {out_path.name} ...")
        tile_da = get_tile(tiles, (z, y, x))
        tile_np = tile_da.compute()

        if tile_np.ndim != 3:
            raise ValueError(f"{td.name}: expected 3D tile, got shape {tile_np.shape}")

        kwargs = {}
        if compression:
            kwargs["compression"] = compression

        tiff.imwrite(str(tmp_path), tile_np, **kwargs)
        tmp_path.replace(out_path)

        pages = validate_written_tiff(out_path, expected_z=tile_np.shape[0])
        print(f"  wrote shape={tile_np.shape} dtype={tile_np.dtype} tiff_pages={pages}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild Tile z-y-x.tif from Zarr using Large Image Tiler logic.")
    p.add_argument(
        "--chunk",
        choices=["first", "second", "all"],
        default="all",
        help="Which default chunk mapping to rebuild.",
    )
    p.add_argument("--tile-z", type=int, default=512)
    p.add_argument("--tile-y", type=int, default=512)
    p.add_argument("--tile-x", type=int, default=512)
    p.add_argument(
        "--only-tile",
        type=str,
        default=None,
        help='Example: "0-1-2" (matches folder `tile 0-1-2`).',
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--compression",
        type=str,
        default="",
        help='Optional tifffile compression, e.g. "zlib". Leave empty for default writer behavior.',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tile_shape = (args.tile_z, args.tile_y, args.tile_x)
    compression = args.compression.strip() or None

    jobs = DEFAULT_JOBS
    if args.chunk == "first":
        jobs = [DEFAULT_JOBS[0]]
    elif args.chunk == "second":
        jobs = [DEFAULT_JOBS[1]]

    for job_label, tagged_root, zarr_path in jobs:
        rebuild_job(
            job_label=job_label,
            tagged_root=tagged_root,
            zarr_path=zarr_path,
            tile_shape=tile_shape,
            only_tile=args.only_tile,
            dry_run=args.dry_run,
            compression=compression,
        )


if __name__ == "__main__":
    main()
