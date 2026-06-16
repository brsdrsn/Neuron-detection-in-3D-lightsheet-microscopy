"""
Pipeline stage: 5b — single-tile QC visualization
=================================================
Run trained YOLO on one tile (TIFF or Zarr) and overlay detection boxes + centers
in Napari. Useful for tuning confidence threshold before full-chunk inference.

Inputs:  trained weights at runs/detect/train_yolo_tagged_tiles_v3/weights/best.pt
         hard-coded tile_path (edit before running)
Outputs: interactive Napari window (no files written)

Run:     edit tile_path, then python qc_yolo_detections_tile_napari.py
For whole-chunk inference + CSV export, use infer_chunk_global_coords.py instead.
"""
import numpy as np
import tifffile as tiff
import napari
import zarr
from ultralytics import YOLO
from pathlib import Path


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-normalize a 2D float slice to 8-bit for YOLO input."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99.8))  # robust contrast stretch
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def ensure_zyx(vol: np.ndarray) -> np.ndarray:
    """Return a 3D (Z, Y, X) volume; squeeze singleton dims and pick one channel if 4D."""
    vol = np.asarray(vol)
    vol = np.squeeze(vol)

    if vol.ndim == 3:
        return vol

    if vol.ndim == 4:
        # Common layouts:
        # - (C, Z, Y, X) -> take channel 0
        # - (Z, Y, X, C) -> take channel 0
        if vol.shape[0] <= 4:
            return vol[0]
        if vol.shape[-1] <= 4:
            return vol[..., 0]

    raise ValueError(
        f"Expected a 3D volume (Z,Y,X). Got shape {vol.shape}. "
        "Set tile_path to a 3D dataset/key or update dimension selection."
    )


# ---- paths and model ----
project_root = Path(__file__).resolve().parent
neurons_root = Path.home() / "Documents" / "neurons"
weights = str(project_root / "runs" / "detect" / "train_yolo_tagged_tiles_v3" / "weights" / "best.pt")
model = YOLO(weights)

# Edit this path to the tile you want to QC (TIFF or Zarr with optional sub-key)
tile_path = str(neurons_root / "tagged tiles" / "1st chunk" / "tile 0-1-3" / "Tile 0-1-3.tif")
#tile_path = str(neurons_root / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr" / "2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr" / "1")

# ---- load volume (TIFF or Zarr) ----
tile_path_lower = tile_path.lower()
if ".zarr" in tile_path_lower:
    # Handle Zarr group path like "...fused.zarr/...fused.zarr/1"
    zarr_suffix_idx = tile_path_lower.rfind(".zarr") + len(".zarr")
    zarr_root = Path(tile_path[:zarr_suffix_idx])
    zarr_key = tile_path[zarr_suffix_idx:].strip("\\/")
    arr = zarr.open(str(zarr_root), mode="r")
    vol = np.asarray(arr[zarr_key] if zarr_key else arr)
else:
    vol = tiff.imread(tile_path)  # (Z,Y,X) tile

vol = ensure_zyx(vol)
Z, Y, X = vol.shape

# ---- run YOLO slice-by-slice and collect detections ----
rects_zyx = []   # rectangle corners in (z, y, x) for Napari shapes layer
points_zyx = []  # box centers in (z, y, x) for Napari points layer
scores = []      # confidence per detection (shown as shape labels)

for z in range(Z):
    u8 = to_uint8(vol[z])
    # YOLO expects 3-channel RGB; replicate grayscale into R,G,B
    img3 = np.stack([u8, u8, u8], axis=-1)

    res = model.predict(img3, imgsz=640, conf=0.1, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        continue

    xyxy = res.boxes.xyxy.cpu().numpy()  # pixel coords in slice (x1,y1,x2,y2)
    conf = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        # Napari rectangle: 4 corners at fixed z, spanning the YOLO box in Y/X
        rects_zyx.append(np.array([[z, y1, x1],
                                   [z, y1, x2],
                                   [z, y2, x2],
                                   [z, y2, x1]], dtype=np.float32))
        points_zyx.append([z, 0.5*(y1+y2), 0.5*(x1+x2)])
        scores.append(float(c))

# ---- open Napari with image + overlays ----
viewer = napari.Viewer()
viewer.add_image(vol, name="tile", rendering="mip")

shapes = viewer.add_shapes(rects_zyx, shape_type="rectangle", edge_width=1.5, name="YOLO boxes")
points_arr = np.array(points_zyx, dtype=np.float32) if points_zyx else np.empty((0, 3), dtype=np.float32)
pts = viewer.add_points(points_arr, size=3, name="YOLO centers")

# Annotate each box with its confidence score
if scores:
    shapes.properties = {"conf": np.array(scores, dtype=np.float32)}
    shapes.text = {"string": "conf", "size": 8, "color": "white"}

napari.run()
