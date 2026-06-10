# Neuron detection pipeline (lightsheet Zarr → YOLO → global coordinates)

End-to-end workflow for detecting neurons in large fused lightsheet microscopy volumes stored as **Zarr**. The pipeline tiles volumes for manual labeling, trains a **YOLOv8** object detector on 2D slices, runs inference across full chunks, and exports **global neuron center coordinates** for visualization and analysis.

This repository contains:

- **`napari-large-tiler/`** — Napari plugin for browsing and tiling large Zarr volumes
- **Pipeline scripts** — dataset export, training, inference, QC, and plotting
- **`runs/detect/`** — trained YOLO weights (after training)
- **`outputs/`** — inference results (CSV / NPY point clouds)

Raw image data and manual labels live outside this repo under `~/Documents/neurons/` by convention.

---

## Pipeline overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Zarr chunk     │────▶│  napari-large-tiler  │────▶│  Tagged tile TIFFs  │
│  (lightsheet)   │     │  (browse / navigate) │     │  + manual masks     │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                │
                     ┌──────────────────────┐                   │
                     │  build_yolo_dataset  │◀──────────────────┘
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐     ┌─────────────────────────┐
                     │  train_yolo          │────▶│  best.pt weights        │
                     └──────────┬───────────┘     └──────────┬──────────────┘
                                │                            │
                     ┌──────────▼───────────────────────────▼──────────────┐
                     │  infer_chunk_global_coords  (production inference)  │
                     └──────────┬──────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     view_neuron_centers   plot_neuron_density   CSV / NPY exports
     _on_chunk_napari.py   _mip.py
```

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 — Tile export | `export_tiles_from_zarr.py` | Write `Tile z-y-x.tif` from Zarr for labeling |
| 2 — Labeling | Napari (+ nnInteractive) | Manual 3D neuron masks per tile |
| 3 — Dataset | `build_yolo_dataset.py` | Masks → YOLO labels; export PNG train/val set |
| 4 — Training | `train_yolo.py` | Fine-tune YOLOv8 on exported dataset |
| 5 — Inference | `infer_chunk_global_coords.py` | Detect neurons across a full Zarr chunk |
| 5b — QC | `qc_yolo_detections_tile_napari.py` | Visual check on a single tile |
| 6 — View results | `view_neuron_centers_on_chunk_napari.py` | Overlay points on full chunk in Napari |
| 7 — Analysis | `plot_neuron_density_mip.py` | Top-down XY density map from CSV |

Legacy / one-off scripts are in `redundant scripts/`.

---

## Prerequisites

### Hardware

- **GPU recommended** for YOLO training and inference (CUDA). CPU works but is slow on full chunks.
- **RAM**: full chunks are processed tile-by-tile; 16 GB+ is comfortable.
- Lightsheet volumes can be tens of GB on disk; ensure enough free space for TIFF exports and training data.

### Software

- Windows 10/11 (commands below use PowerShell; adapt paths for Linux/macOS)
- Python 3.10+ with a virtual environment
- [Napari](https://napari.org/) for interactive viewing and labeling
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) for training and inference

### Environment setup

From the repository root:

```powershell
# Create and activate the project venv (if not already present)
python -m venv napari-env
.\napari-env\Scripts\Activate.ps1

# Core dependencies (adjust versions to match your working env)
pip install napari magicgui dask zarr tifffile numpy pandas matplotlib pillow ultralytics

# Install the local tiling plugin (editable)
pip install -e .\napari-large-tiler
```

Optional for manual labeling: install **napari-nnInteractive** (interactive segmentation in Napari). GPU memory requirements can be high (~11 GB for default settings).

Verify the plugin:

```powershell
python -c "import napari_large_tiler; print('napari-large-tiler OK')"
```

---

## Data layout and conventions

### Zarr input

Volumes are multiscale Zarr stores from fused lightsheet acquisitions, e.g.:

```
C:\Users\<you>\Documents\neurons\
  2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\
    2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\
      1\                    ← dataset key passed to scripts
```

Scripts accept the full path including the trailing `\1` (or equivalent scale/key).

### Axis order

All production code uses **(Z, Y, X)** — depth, height, width. The tiling plugin reorders raw arrays so Z is the smallest dimension when needed.

### Tile naming

| Item | Format | Example |
|------|--------|---------|
| Tile folder | `tile <z>-<y>-<x>` | `tile 0-1-3` |
| Tile TIFF | `Tile <z>-<y>-<x>.tif` | `Tile 0-1-3.tif` |
| Object mask | `object N - Tile <z>-<y>-<x>.tif` | `object 1 - Tile 0-1-3.tif` |
| YOLO label (per slice) | `tile_<z>-<y>-<x>_z####.txt` | `tile_0-1-3_z0042.txt` |

Tile indices `(z, y, x)` match `tiles.blocks[(z, y, x)]` after rechunking — the same indexing used by `napari-large-tiler` and `infer_chunk_global_coords.py`.

### Tagged tiles directory

Manual labels and exported TIFFs are stored under:

```
~/Documents/neurons/tagged tiles/
  1st chunk/
    tile 0-1-3/
      Tile 0-1-3.tif
      positive/
        object 1 - Tile 0-1-3.tif
        object 2 - Tile 0-1-3.tif
        yolo_labels_per_slice/    ← auto-generated by build_yolo_dataset.py
  2nd chunk/
    tile ...
```

Alternative mask locations are also searched (`objects/`, `positives/`, etc.) — see `build_yolo_dataset.py`.

### Training dataset (generated)

```
~/Documents/neurons/yolo_tagged_tiles_v3/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

### Model weights (after training)

```
Napari/runs/detect/train_yolo_tagged_tiles_v3/weights/best.pt
```

### Inference outputs

```
Napari/outputs/chunk_<id>/global_coords/
  global_neuron_centers_chunk_<id>_all_tiles.csv   # full metadata
  global_neuron_centers_chunk_<id>_all_tiles.npy   # N×3 float32, Z,Y,X
  neuron_centers_chunk_<id>_all_tiles.csv          # Napari points (processed axes)
  neuron_centers_chunk_<id>_all_tiles_raw_axes.csv # Napari points (raw Zarr axes)
```

---

## Step 1 — Browse and tile volumes with napari-large-tiler

The **Large Image Tiler** plugin lets you load a Zarr chunk, split it into manageable 3D tiles, and step through them in Napari. This is the starting point for choosing which regions to label.

### Install and launch

```powershell
.\napari-env\Scripts\Activate.ps1
pip install -e .\napari-large-tiler
napari
```

In Napari: **Plugins → Large Image Tiler → Large Tiler** (dock widget).

### Using the widget

1. **Zarr directory** — browse to your dataset key, e.g.  
   `...\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\...\1`
2. Set **tile size** in Z, Y, X (default 256³ in the UI; production scripts use **512³** — keep these consistent).
3. Click **Start Tiling** — the first tile loads as an image layer named `Tile 0-0-0`.
4. Use **Next Tile** / **Previous Tile** widgets to navigate. Navigation order: X → Y → Z.

### What the plugin does internally

1. `load_zarr()` — opens the array as a Dask array
2. `process_data()` — drops leading singleton dims, fixes byte order, enforces **(Z, Y, X)**
3. `tile_array()` — rechunks into `(tile_z, tile_y, tile_x)` blocks
4. `get_tile()` — extracts one block for display

The same functions are used by `export_tiles_from_zarr.py` and `infer_chunk_global_coords.py`, so tile indices stay aligned across labeling, training, and inference.

---

## Step 2 — Export tile TIFFs for labeling

When you have created `tile z-y-x` folders under `tagged tiles/`, run the export script to write the corresponding multi-page TIFF from the source Zarr.

```powershell
.\napari-env\Scripts\python.exe .\export_tiles_from_zarr.py --chunk first --dry-run
.\napari-env\Scripts\python.exe .\export_tiles_from_zarr.py --chunk first
.\napari-env\Scripts\python.exe .\export_tiles_from_zarr.py --chunk second --only-tile 0-1-2
```

| Flag | Description |
|------|-------------|
| `--chunk {first,second,all}` | Which predefined Zarr ↔ tagged-folder mapping to use |
| `--only-tile 0-1-2` | Rebuild a single tile folder |
| `--tile-z/y/x` | Tile dimensions (default 512 each) |
| `--dry-run` | List what would be written without reading Zarr |

Edit `DEFAULT_JOBS` at the top of `export_tiles_from_zarr.py` to point at your Zarr paths and tagged-tile roots.

---

## Step 3 — Manual labeling (create training masks)

For each tile you want in the training set:

1. Open `Tile z-y-x.tif` in Napari (or navigate there with the tiler plugin).
2. Segment individual neurons using your preferred tool (e.g. **nnInteractive** point/bbox prompts).
3. Save each neuron as a separate binary 3D mask TIFF under the tile folder, e.g.  
   `positive/object 1 - Tile 0-1-3.tif`

**Tips:**

- Masks must be the same shape as the tile volume **(Z, Y, X)**.
- Include tiles with **no neurons** (negative examples) — empty label files are valid.
- Label diverse regions (different densities, depths, artifacts) for a robust detector.

You do not need to hand-write YOLO `.txt` files; `build_yolo_dataset.py` derives bounding boxes from the masks automatically.

---

## Step 4 — Build the YOLO training dataset

```powershell
.\napari-env\Scripts\python.exe .\build_yolo_dataset.py
```

This script:

1. Scans all tile folders under configured roots (`1st chunk`, `2nd chunk`, legacy `tile ...` dirs).
2. For each tile, finds object mask TIFFs and builds per-slice YOLO label files (`yolo_labels_per_slice/`).
3. Exports every Z slice as an 8-bit PNG (percentile-normalized) with matching `.txt` labels.
4. Splits slices 80/20 into train/val (seed=0).
5. Writes `~/Documents/neurons/yolo_tagged_tiles_v3/data.yaml`.

Edit the `# ---- CONFIG ----` section in `build_yolo_dataset.py` if your paths differ. Set `neg_per_pos = 3` to subsample negative slices (default uses all slices).

---

## Step 5 — Train YOLO

```powershell
.\napari-env\Scripts\python.exe .\train_yolo.py
```

Default settings (see `train_yolo.py`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Base model | `yolov8n.pt` | Small/fast; try `yolov8s.pt` for more capacity |
| Data | `~/Documents/neurons/yolo_tagged_tiles_v3/data.yaml` | |
| Epochs | 200 | Early stopping patience 30 |
| Image size | 640 | Must match inference |
| Batch | 8 | Lower if GPU OOM |
| Device | `0` | First CUDA GPU; use `cpu` if no GPU |
| Workers | 0 | **Required on Windows** for Ultralytics DataLoader |

Custom run:

```powershell
.\napari-env\Scripts\python.exe .\train_yolo.py --epochs 100 --batch 4 --device 0 --name my_run_v4
```

Weights are saved to:

```
runs/detect/<name>/weights/best.pt
runs/detect/<name>/weights/last.pt
```

Training curves and metrics: `runs/detect/<name>/results.csv`.

---

## Step 6 — Run inference on a full chunk (main production use)

This is the primary script for **applying the fine-tuned model** to new data.

```powershell
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "C:\Users\pc\Documents\neurons\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\1"
```

### Common options

```powershell
# Use specific weights
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "...\1" `
  --weights ".\runs\detect\train_yolo_tagged_tiles_v3\weights\best.pt"

# Tune detection sensitivity (default 0.27)
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "...\1" `
  --conf 0.35

# Process one tile only (fast validation)
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "...\1" `
  --only-tile 0-1-3

# Limit tiles for a quick smoke test
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "...\1" `
  --max-tiles 3

# Custom tile size (must match labeling/export)
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "...\1" `
  --tile-z 512 --tile-y 512 --tile-x 512
```

| Flag | Default | Description |
|------|---------|-------------|
| `--zarr-path` | *(required)* | Path to Zarr dataset key |
| `--weights` | `runs/detect/train_yolo_tagged_tiles_v3/weights/best.pt` | Trained `.pt` file |
| `--conf` | `0.27` | Confidence threshold; raise to reduce false positives |
| `--imgsz` | `640` | YOLO input size (match training) |
| `--tile-z/y/x` | `512` | Tile dimensions |
| `--out-dir` | `outputs/chunk_<id>/global_coords/` | Output directory |
| `--only-tile` | — | Single tile `z-y-x` |
| `--max-tiles` | `0` (all) | Stop after N tiles |

### What happens internally

1. Load and preprocess the Zarr volume to **(Z, Y, X)**.
2. Rechunk into tiles (same grid as labeling).
3. For each tile, iterate every Z slice:
   - Percentile-normalize to 8-bit, stack to pseudo-RGB
   - Run YOLO `predict()`
   - Convert each bounding box to a **center point** in chunk-global coordinates
4. Export CSV (with confidence + tile metadata), NPY `(N, 3)`, and Napari-compatible point CSVs.

Global coordinates:

```
global_z = tile_z * tile_depth  + local_z
global_y = tile_y * tile_height + box_center_y
global_x = tile_x * tile_width  + box_center_x
```

### Output files explained

| File | Contents |
|------|----------|
| `global_neuron_centers_*.csv` | `z, y, x, conf, tile_z, tile_y, tile_x, local_z` |
| `global_neuron_centers_*.npy` | `float32` array shape `(N, 3)` — columns are Z, Y, X |
| `neuron_centers_*.csv` | Napari format: `axis-0, axis-1, axis-2` in processed Z,Y,X |
| `neuron_centers_*_raw_axes.csv` | Same points remapped to raw Zarr axis order for direct overlay |

Use `*_raw_axes.csv` when loading the **unprocessed** Zarr array in Napari; use the plain CSV or NPY when using `process_data()`-style volumes.

---

## Step 7 — Quality control before full-chunk runs

Edit `tile_path` in `qc_yolo_detections_tile_napari.py`, then:

```powershell
.\napari-env\Scripts\python.exe .\qc_yolo_detections_tile_napari.py
```

Opens Napari with the tile volume, YOLO bounding boxes (shapes layer), and center points. Tune `--conf` in the script to decide a good threshold before running `infer_chunk_global_coords.py` on the full chunk.

---

## Step 8 — Visualize inference results

Overlay detected centers on the full processed chunk:

```powershell
.\napari-env\Scripts\python.exe .\view_neuron_centers_on_chunk_napari.py `
  --zarr-path "C:\Users\pc\Documents\neurons\...\1" `
  --points-npy ".\outputs\chunk 1\global_coords\global_neuron_centers_chunk_1_all_tiles.npy"
```

Optional: `--points-size 5` to enlarge markers.

To load points from Napari's GUI instead: **Layer → Add Points → Open file** and choose `neuron_centers_*_raw_axes.csv`.

---

## Step 9 — Density analysis

Top-down neuron density (MIP over Z):

```powershell
.\napari-env\Scripts\python.exe .\plot_neuron_density_mip.py
```

Edit the CSV path in the `if __name__ == "__main__"` block, or import `plot_neuron_density_mip_xy_from_raw_axes()` from another script. Works with `*_raw_axes.csv` or `global_neuron_centers_*.csv`.

---

## Quick start for new users (inference only)

If the model is already trained and you only need to detect neurons in a new chunk:

1. Activate the environment: `.\napari-env\Scripts\Activate.ps1`
2. Confirm weights exist: `runs/detect/train_yolo_tagged_tiles_v3/weights/best.pt`
3. Run inference:

```powershell
.\napari-env\Scripts\python.exe .\infer_chunk_global_coords.py `
  --zarr-path "PATH\TO\your\chunk\1" `
  --weights ".\runs\detect\train_yolo_tagged_tiles_v3\weights\best.pt" `
  --conf 0.27
```

4. Inspect results:

```powershell
.\napari-env\Scripts\python.exe .\view_neuron_centers_on_chunk_napari.py `
  --zarr-path "PATH\TO\your\chunk\1" `
  --points-npy ".\outputs\chunk_1\global_coords\global_neuron_centers_chunk_1_all_tiles.npy"
```

5. Optional: plot density with `plot_neuron_density_mip.py`.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `Dataset YAML not found` | Skipped `build_yolo_dataset.py` | Run dataset export first |
| CUDA OOM during training | Batch too large | `--batch 4` or `2`; use `yolov8n.pt` |
| CUDA OOM during inference | Full tile in memory | Reduce tile size or process with `--only-tile` |
| No detections | `conf` too high or domain shift | Lower `--conf`; QC on a labeled tile |
| Too many false positives | `conf` too low | Raise `--conf` (try 0.35–0.5) |
| Points misaligned in Napari | Wrong axis CSV | Use `*_raw_axes.csv` for raw Zarr, or NPY with processed volume |
| Tile index mismatch | Different tile sizes | Use same `--tile-z/y/x` everywhere (512 default) |
| Empty training set | No masks found | Check mask filenames match `object*Tile z-y-x.tif` pattern |
| Ultralytics hangs on Windows | `workers > 0` | Keep `--workers 0` in `train_yolo.py` |
| TIFF reads as 2D only | Broken TIFF metadata | `build_yolo_dataset.py` has a page-by-page fallback reader |

---

## Repository structure

```
Napari/
├── napari-large-tiler/          # Napari plugin (tiling utilities + UI)
├── build_yolo_dataset.py        # Stage 3: export YOLO dataset
├── train_yolo.py                # Stage 4: train YOLOv8
├── export_tiles_from_zarr.py    # Stage 1: Zarr → tile TIFF
├── infer_chunk_global_coords.py # Stage 5: production inference
├── qc_yolo_detections_tile_napari.py
├── view_neuron_centers_on_chunk_napari.py
├── plot_neuron_density_mip.py
├── migrate_labels_yxz_to_zyx.py # One-off axis migration (legacy)
├── redundant scripts/           # Archived experiments and superseded scripts
├── runs/detect/                 # Training runs and weights
└── outputs/                     # Inference CSV / NPY outputs
```

---

## Re-training after adding new labels

1. Label new tiles in Napari and save masks under `tagged tiles/`.
2. (Re-)export TIFFs if needed: `export_tiles_from_zarr.py`.
3. Rebuild dataset: `build_yolo_dataset.py`.
4. Train: `train_yolo.py --name train_yolo_tagged_tiles_v4` (new run name).
5. Point inference at new weights: `--weights runs/detect/train_yolo_tagged_tiles_v4/weights/best.pt`.

---

## Citation and context

This pipeline was built for **DiI-labeled neuron detection** in fused **MFS2 lightsheet** volumes. YOLO operates on individual 2D slices; 3D structure is recovered by stacking slice detections and mapping tile-local coordinates back to chunk-global space via the shared tiling grid from `napari-large-tiler`.
