"""
Pipeline stage: 5 — chunk-scale inference (production)
=====================================================
Tile a full Zarr chunk, run trained YOLO slice-by-slice, and export neuron
center coordinates in chunk-global (Z, Y, X) space.

Uses napari-large-tiler tiling (512³ default) and maps detections back to global
indices. Writes both processed-axis and raw-axis Napari-compatible point CSVs.

Inputs:  --zarr-path (required)
         --weights (default: runs/detect/train_yolo_tagged_tiles_v3/weights/best.pt)
Outputs: outputs/chunk_<id>/global_coords/
           global_neuron_centers_chunk_<id>_<tiles>.csv  (full metadata)
           global_neuron_centers_chunk_<id>_<tiles>.npy  (N×3 Z,Y,X)
           neuron_centers_chunk_<id>_<tiles>.csv         (Napari points, processed axes)
           neuron_centers_chunk_<id>_<tiles>_raw_axes.csv (Napari points, raw Zarr axes)

Run:
  python infer_chunk_global_coords.py --zarr-path "C:\\...\\fused.zarr\\...\\1"
  python infer_chunk_global_coords.py --zarr-path "..." --only-tile 0-1-3

Suggested rename: infer_neuron_centers_chunk.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

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


def _compute_processed_to_raw_axis_map(
    raw_shape: tuple[int, ...], processed_shape: tuple[int, ...]
) -> tuple[int, int, int] | None:
    """
    Return mapping from processed axes (Z,Y,X) to raw axes.
    Example:
      raw=(4207,2883,248), processed=(248,4207,2883) -> (2,0,1)
    """
    if len(raw_shape) != 3 or len(processed_shape) != 3:
        return None

    available_raw_axes = [0, 1, 2]
    mapping: list[int] = []
    for p_dim in processed_shape:
        matches = [ax for ax in available_raw_axes if raw_shape[ax] == p_dim]
        if len(matches) != 1:
            return None
        chosen = matches[0]
        mapping.append(chosen)
        available_raw_axes.remove(chosen)

    return tuple(mapping)  # type: ignore[return-value]


def _normalize_raw_shape_for_mapping(raw_shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    """
    Reduce raw shape to a comparable 3D shape for axis mapping.
    Handles singleton dims and common channel-first/channel-last layouts.
    """
    shape = tuple(int(v) for v in raw_shape if int(v) > 1)
    if len(shape) == 3:
        return shape
    if len(shape) == 4:
        if shape[0] <= 4:
            return (shape[1], shape[2], shape[3])
        if shape[3] <= 4:
            return (shape[0], shape[1], shape[2])
    return None


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


@dataclass
class DetectionRecord:
    z: float
    y: float
    x: float
    conf: float
    tile_z: int
    tile_y: int
    tile_x: int
    local_z: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer neurons over chunk tiles and export global center coordinates."
    )
    parser.add_argument(
        "--zarr-path",
        required=True,
        help=r'Input chunk path, e.g. "...fused.zarr\...\fused.zarr\1".',
    )
    parser.add_argument(
        "--weights",
        default=str(
            Path(__file__).resolve().parent
            / "runs"
            / "detect"
            / "train_yolo_tagged_tiles_v3"
            / "weights"
            / "best.pt"
        ),
        help="Path to trained YOLO weights (.pt).",
    )
    parser.add_argument("--tile-z", type=int, default=512, help="Tile depth (Z).")
    parser.add_argument("--tile-y", type=int, default=512, help="Tile height (Y).")
    parser.add_argument("--tile-x", type=int, default=512, help="Tile width (X).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.27, help="YOLO confidence threshold.")
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "Directory where outputs are written. "
            "Default auto-resolves to Napari/outputs/chunk_<id>/global_coords."
        ),
    )
    parser.add_argument(
        "--napari-points-csv",
        default="",
        help=(
            'Optional explicit Napari points CSV path with columns "axis-0,axis-1,axis-2" '
            "(global Z,Y,X). If omitted, it is written under --out-dir."
        ),
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Optional cap for quick validation (0 = process all tiles).",
    )
    parser.add_argument(
        "--only-tile",
        type=str,
        default="",
        help='Optional single tile index in "z-y-x" format, e.g. "0-1-3".',
    )
    return parser.parse_args()


def parse_tile_index(tile_text: str) -> tuple[int, int, int]:
    parts = tile_text.strip().split("-")
    if len(parts) != 3:
        raise ValueError(f'Invalid --only-tile "{tile_text}". Expected format "z-y-x".')
    try:
        tz, ty, tx = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f'Invalid --only-tile "{tile_text}". Expected integers in "z-y-x".'
        ) from exc
    return tz, ty, tx


def infer_chunk_id(zarr_path: Path) -> str:
    # Prefer terminal numeric component, e.g. ...\\...zarr\\1
    for part in reversed(zarr_path.parts):
        if part.isdigit():
            return part

    # Fallback: numeric suffix in path string
    m = re.search(r"(?:^|[\\/])(\d+)(?:$|[\\/])", str(zarr_path))
    if m:
        return m.group(1)
    return "unknown"


def export_csv(path: Path, records: list[DetectionRecord]) -> None:
    fieldnames = ["z", "y", "x", "conf", "tile_z", "tile_y", "tile_x", "local_z"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "z": f"{rec.z:.3f}",
                    "y": f"{rec.y:.3f}",
                    "x": f"{rec.x:.3f}",
                    "conf": f"{rec.conf:.6f}",
                    "tile_z": rec.tile_z,
                    "tile_y": rec.tile_y,
                    "tile_x": rec.tile_x,
                    "local_z": rec.local_z,
                }
            )


def export_npy(path: Path, records: list[DetectionRecord]) -> None:
    if not records:
        arr = np.empty((0, 3), dtype=np.float32)
    else:
        arr = np.asarray([[r.z, r.y, r.x] for r in records], dtype=np.float32)
    np.save(path, arr)


def export_napari_points_csv(path: Path, records: list[DetectionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["axis-0", "axis-1", "axis-2"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "axis-0": f"{rec.z:.3f}",
                    "axis-1": f"{rec.y:.3f}",
                    "axis-2": f"{rec.x:.3f}",
                }
            )


def export_napari_points_csv_raw_axes(
    path: Path, records: list[DetectionRecord], processed_to_raw_axis_map: tuple[int, int, int]
) -> None:
    """
    Save points in raw chunk axis order for direct overlay on raw-loaded arrays.
    Input records are always in processed Z,Y,X.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["axis-0", "axis-1", "axis-2"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            p_coords = [rec.z, rec.y, rec.x]  # processed (Z,Y,X)
            raw_coords = [0.0, 0.0, 0.0]
            for p_axis, raw_axis in enumerate(processed_to_raw_axis_map):
                raw_coords[raw_axis] = p_coords[p_axis]
            writer.writerow(
                {
                    "axis-0": f"{raw_coords[0]:.3f}",
                    "axis-1": f"{raw_coords[1]:.3f}",
                    "axis-2": f"{raw_coords[2]:.3f}",
                }
            )


def main() -> None:
    args = parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights}")

    tile_shape = (args.tile_z, args.tile_y, args.tile_x)
    chunk_id = infer_chunk_id(zarr_path)
    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parent / "outputs" / f"chunk_{chunk_id}" / "global_coords"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Loading chunk: {zarr_path}")
    raw = load_zarr(str(zarr_path))
    raw_shape_full = tuple(getattr(raw, "shape", ()))
    raw_shape = _normalize_raw_shape_for_mapping(raw_shape_full)
    full = process_data(raw)
    if full.ndim != 3:
        raise ValueError(f"Expected 3D processed chunk, got shape {full.shape}")
    full_z, full_y, full_x = full.shape

    tiles = tile_array(full, tile_shape)
    bz, by, bx = num_tiles(tiles)

    print(f"Chunk shape (Z,Y,X): {full.shape}")
    print(f"Raw chunk shape (full): {raw_shape_full}")
    if raw_shape is not None:
        print(f"Raw chunk shape (3D for mapping): {raw_shape}")
    print(f"Tile shape (Z,Y,X): {tile_shape}")
    print(f"Num tiles (Z,Y,X): {(bz, by, bx)}")

    records: list[DetectionRecord] = []
    processed_tiles = 0
    detection_count = 0
    max_tiles = args.max_tiles if args.max_tiles and args.max_tiles > 0 else None
    only_tile = parse_tile_index(args.only_tile) if args.only_tile.strip() else None
    if only_tile is not None:
        tz, ty, tx = only_tile
        if not (0 <= tz < bz and 0 <= ty < by and 0 <= tx < bx):
            raise ValueError(
                f"--only-tile {args.only_tile} is out of range for available tiles {(bz, by, bx)}"
            )
        print(f"Processing only tile: (z,y,x)=({tz},{ty},{tx})")
    run_label = (
        f"tile_{only_tile[0]}-{only_tile[1]}-{only_tile[2]}" if only_tile is not None else "all_tiles"
    )

    for tz in range(bz):
        for ty in range(by):
            for tx in range(bx):
                if only_tile is not None and (tz, ty, tx) != only_tile:
                    continue
                tile_np = get_tile(tiles, (tz, ty, tx)).compute()
                if tile_np.ndim != 3:
                    raise ValueError(
                        f"Tile {(tz, ty, tx)} expected 3D, got shape {tile_np.shape}"
                    )

                tile_detection_count = 0
                local_z_size = tile_np.shape[0]
                for lz in range(local_z_size):
                    u8 = to_uint8(tile_np[lz])
                    img3 = np.stack([u8, u8, u8], axis=-1)
                    result = model.predict(
                        img3, imgsz=args.imgsz, conf=args.conf, verbose=False
                    )[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()

                    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                        # Convert tile-local center to chunk-global center.
                        gx = tx * args.tile_x + 0.5 * (x1 + x2)
                        gy = ty * args.tile_y + 0.5 * (y1 + y2)
                        gz = tz * args.tile_z + lz

                        if not (0 <= gx < full_x and 0 <= gy < full_y and 0 <= gz < full_z):
                            continue

                        records.append(
                            DetectionRecord(
                                z=float(gz),
                                y=float(gy),
                                x=float(gx),
                                conf=float(conf),
                                tile_z=tz,
                                tile_y=ty,
                                tile_x=tx,
                                local_z=lz,
                            )
                        )
                        detection_count += 1
                        tile_detection_count += 1

                processed_tiles += 1
                print(
                    f"Processed tile {processed_tiles}/{bz*by*bx}: "
                    f"(z,y,x)=({tz},{ty},{tx}), tile detections={tile_detection_count}, "
                    f"cumulative detections={detection_count}"
                )

                if only_tile is not None:
                    break

                if max_tiles is not None and processed_tiles >= max_tiles:
                    print(f"Reached --max-tiles={max_tiles}, stopping early for validation.")
                    break
            if only_tile is not None and processed_tiles > 0:
                break
            if max_tiles is not None and processed_tiles >= max_tiles:
                break
        if only_tile is not None and processed_tiles > 0:
            break
        if max_tiles is not None and processed_tiles >= max_tiles:
            break

    stem = f"chunk_{chunk_id}_{run_label}"
    csv_path = out_dir / f"global_neuron_centers_{stem}.csv"
    npy_path = out_dir / f"global_neuron_centers_{stem}.npy"
    if args.napari_points_csv.strip():
        napari_points_csv_path = Path(args.napari_points_csv)
    else:
        napari_points_csv_path = out_dir / f"neuron_centers_{stem}.csv"
    napari_points_csv_raw_axes_path = napari_points_csv_path.with_name(
        f"{napari_points_csv_path.stem}_raw_axes{napari_points_csv_path.suffix}"
    )

    export_csv(csv_path, records)
    export_npy(npy_path, records)
    export_napari_points_csv(napari_points_csv_path, records)
    axis_map = (
        _compute_processed_to_raw_axis_map(raw_shape, tuple(full.shape))
        if raw_shape is not None
        else None
    )
    if axis_map is not None:
        export_napari_points_csv_raw_axes(
            napari_points_csv_raw_axes_path, records, processed_to_raw_axis_map=axis_map
        )
    else:
        print(
            "WARNING: Could not infer processed->raw axis mapping from shapes; "
            "raw-axis points CSV was not written."
        )

    print("\nDone.")
    print(f"Tiles processed: {processed_tiles}")
    print(f"Detections kept: {len(records)}")
    print(f"CSV: {csv_path}")
    print(f"NPY: {npy_path}")
    print(f"Napari points CSV (processed Z,Y,X): {napari_points_csv_path}")
    if axis_map is not None:
        print(f"Napari points CSV (raw axis order): {napari_points_csv_raw_axes_path}")


if __name__ == "__main__":
    main()
