"""
Pipeline stage: 3 — YOLO dataset export (does NOT train a model)
================================================================
Scan all tagged tile folders, generate per-slice YOLO labels from object masks
(if missing), export 2D PNG slices + label .txt files, and write data.yaml.

Despite the filename, this script only builds the dataset. Actual training is in
train_yolo_weights.py.

Inputs:  ~/Documents/neurons/tagged tiles/{1st chunk, 2nd chunk, tile ...}/
         Each tile folder needs Tile <z-y-x>.tif and object mask TIFFs.
Outputs: ~/Documents/neurons/yolo_tagged_tiles_v3/
           images/{train,val}/*.png
           labels/{train,val}/*.txt
           data.yaml

Run:     python train_yolo.py
Suggested rename: build_yolo_dataset.py
"""

import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def read_volume_tiff(tile_path: Path) -> np.ndarray:
    """
    Robustly read TIFF volumes.
    Some files in this dataset have broken shaped-series metadata, where
    tifffile.imread returns only a 2D plane although many pages exist.
    """
    vol = tiff.imread(str(tile_path))
    if getattr(vol, "ndim", 0) == 3:
        return vol

    with tiff.TiffFile(str(tile_path)) as tf:
        n_pages = len(tf.pages)
    if n_pages <= 1:
        return vol

    # First fallback: try loading all pages in one go.
    try:
        vol = tiff.imread(str(tile_path), key=range(n_pages))
        if getattr(vol, "ndim", 0) == 3:
            return vol
    except Exception as e:
        print(f"Bulk page read failed for {tile_path.name}: {e}")

    # Second fallback: read pages one-by-one and skip unreadable frames.
    frames = []
    with tiff.TiffFile(str(tile_path)) as tf:
        expected_shape = None
        for i, page in enumerate(tf.pages):
            try:
                frame = page.asarray()
            except Exception as e:
                print(f"Skipping unreadable frame {i} in {tile_path.name}: {e}")
                continue

            if getattr(frame, "ndim", 0) != 2:
                print(f"Skipping non-2D frame {i} in {tile_path.name}: shape={frame.shape}")
                continue
            if expected_shape is None:
                expected_shape = frame.shape
            if frame.shape != expected_shape:
                print(
                    f"Skipping mismatched frame {i} in {tile_path.name}: "
                    f"shape={frame.shape}, expected={expected_shape}"
                )
                continue
            frames.append(frame)

    if not frames:
        raise ValueError(f"Could not read any valid 2D frames from TIFF: {tile_path}")
    return np.stack(frames, axis=0)

# ---- CONFIG ----
neurons_root = Path.home() / "Documents" / "neurons"
tagged_tiles_root = neurons_root / "tagged tiles"
# Union dataset: new chunk folders + the original five-tile set used for yolo_five_tiles.
legacy_five_tile_ids = ["0-5-0", "1-3-0", "1-4-0", "2-2-0", "3-3-0"]
dataset_roots = [
    tagged_tiles_root / "1st chunk",
    tagged_tiles_root / "2nd chunk",
    *[tagged_tiles_root / f"tile {tid}" for tid in legacy_five_tile_ids],
    tagged_tiles_root / "1st tile",   # backward compatibility
    tagged_tiles_root / "2nd tile",   # backward compatibility
]

out_root = str(neurons_root / "yolo_tagged_tiles_v3")
train_ratio = 0.8
seed = 0

# If you want to enforce 1:3 pos:neg for this tile, set neg_per_pos=3
# If you want "use all slices", set neg_per_pos=None
neg_per_pos = None

# ---- OUTPUT DIRS ----
img_train = os.path.join(out_root, "images", "train")
img_val   = os.path.join(out_root, "images", "val")
lab_train = os.path.join(out_root, "labels", "train")
lab_val   = os.path.join(out_root, "labels", "val")
for d in [img_train, img_val, lab_train, lab_val]:
    os.makedirs(d, exist_ok=True)

random.seed(seed)
all_samples = []


def bbox2d_from_instance_slice(inst2d: np.ndarray, obj_id: int):
    ys, xs = np.where(inst2d == obj_id)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def yolo_line(x1, y1, x2, y2, W, H, cls=0):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def create_yolo_labels_from_object_masks(tile_dir: Path, tile_id: str) -> Path | None:
    """
    Convert object mask TIFFs into per-slice YOLO txt labels.
    """
    object_dirs = [
        tile_dir / "objects",
        tile_dir / "object",
        tile_dir / "positive" / "objects",
        tile_dir / "positives" / "objects",
        tile_dir / "positive",
        tile_dir / "positives",
        tile_dir,
    ]
    obj_paths = []
    for obj_dir in object_dirs:
        if not obj_dir.exists():
            continue
        obj_paths.extend(sorted(glob.glob(str(obj_dir / f"object*Tile {tile_id}.tif"))))
        obj_paths.extend(sorted(glob.glob(str(obj_dir / f"object*{tile_id}.tif"))))
    obj_paths = sorted(set(obj_paths))
    if not obj_paths:
        return None

    first = tiff.imread(obj_paths[0])
    if first.ndim != 3:
        raise ValueError(f"Expected 3D mask in {obj_paths[0]}, got shape {first.shape}")
    Z, Y, X = first.shape
    dtype = np.uint16 if len(obj_paths) < 65535 else np.uint32
    inst = np.zeros((Z, Y, X), dtype=dtype)

    next_id = 1
    for p in obj_paths:
        m = tiff.imread(p)
        if m.shape != inst.shape:
            raise ValueError(f"Shape mismatch: {p} has {m.shape}, expected {inst.shape}")
        if not np.any(m):
            continue
        inst[m > 0] = next_id
        next_id += 1

    labels_out = tile_dir / "yolo_labels_per_slice"
    labels_out.mkdir(parents=True, exist_ok=True)

    for z in range(Z):
        inst2d = inst[z]
        ids = np.unique(inst2d)
        ids = ids[ids != 0]

        lines = []
        for obj_id in ids:
            bb = bbox2d_from_instance_slice(inst2d, int(obj_id))
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            lines.append(yolo_line(x1, y1, x2, y2, W=X, H=Y, cls=0))

        out_txt = labels_out / f"tile_{tile_id}_z{z:04d}.txt"
        out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Generated YOLO labels from object masks: {labels_out}")
    return labels_out


def resolve_tile_paths(tile_dir: Path, tile_id: str) -> tuple[Path, Path]:
    expected_tif = tile_dir / f"Tile {tile_id}.tif"
    if expected_tif.exists():
        tile_path = expected_tif
    else:
        tif_candidates = sorted(tile_dir.glob("*.tif"))
        tile_path = tif_candidates[0] if tif_candidates else expected_tif
    labels_candidates = [
        tile_dir / "positive" / "yolo_labels_per_slice",
        tile_dir / "positives" / "yolo_labels_per_slice",
        tile_dir / "positive",
        tile_dir / "positives",
        tile_dir / "yolo_labels_per_slice",
    ]
    labels_src_dir = next((p for p in labels_candidates if p.exists()), None)
    if labels_src_dir is None:
        labels_src_dir = create_yolo_labels_from_object_masks(tile_dir, tile_id)
    if labels_src_dir is None:
        labels_src_dir = labels_candidates[0]
    return tile_path, labels_src_dir


all_tile_dirs = []
seen_roots = set()
for root in dataset_roots:
    root = root.resolve()
    # A dataset root may be either a chunk folder (contains `tile ...` dirs)
    # or a single `tile ...` directory (legacy layout).
    if root.is_dir() and root.name.startswith("tile "):
        if str(root) in seen_roots:
            continue
        seen_roots.add(str(root))
        all_tile_dirs.append(root)
        continue
    if not root.exists() and root.name == "2nd chunk":
        alt_root = root.parent / "2st chunk"
        if alt_root.exists():
            root = alt_root.resolve()
    if not root.exists() and root.name == "2nd tile":
        alt_root = root.parent / "2st tile"
        if alt_root.exists():
            root = alt_root.resolve()
    if not root.exists():
        print(f"Skipping missing dataset root: {root}")
        continue
    if str(root) in seen_roots:
        continue
    seen_roots.add(str(root))
    all_tile_dirs.extend(sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("tile ")]))

if not all_tile_dirs:
    raise RuntimeError("No tile directories found in configured dataset roots.")

print(f"Discovered {len(all_tile_dirs)} tile folders.")

for tile_dir in all_tile_dirs:
    tile_id = tile_dir.name.replace("tile ", "", 1).strip()
    tile_path, labels_src_dir = resolve_tile_paths(tile_dir, tile_id)
    if not tile_path.exists():
        raise FileNotFoundError(f"Missing tile TIFF: {tile_path}")
    if not labels_src_dir.exists():
        raise FileNotFoundError(f"Missing labels folder: {labels_src_dir}")

    vol = read_volume_tiff(tile_path)  # (Z,Y,X)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D TIFF for tile {tile_id}, got shape {vol.shape}")

    Z, _, _ = vol.shape
    sample_prefix = tile_dir.parent.name.replace(" ", "_")
    sample_id = f"{sample_prefix}_{tile_id}"
    print(f"\nLoaded tile {sample_id}: shape={vol.shape}, dtype={vol.dtype}")

    all_z = list(range(Z))
    pos_z, neg_z = [], []

    for z in all_z:
        txt = labels_src_dir / f"tile_{tile_id}_z{z:04d}.txt"
        if not txt.exists():
            # If missing, treat as negative.
            neg_z.append(z)
            continue
        content = txt.read_text(encoding="utf-8").strip()
        if content:
            pos_z.append(z)
        else:
            neg_z.append(z)

    print(f"tile {sample_id} -> pos slices: {len(pos_z)}, neg slices: {len(neg_z)}")

    if neg_per_pos is None:
        keep_z = all_z
    else:
        target_neg = min(len(neg_z), neg_per_pos * len(pos_z))
        neg_sample = random.sample(neg_z, k=target_neg) if target_neg > 0 else []
        keep_z = sorted(pos_z + neg_sample)

    print(
        f"tile {sample_id} -> keeping: {len(keep_z)} "
        f"(pos {len(pos_z)} neg {len(keep_z) - len(pos_z)})"
    )

    for z in keep_z:
        all_samples.append(
            {
                "tile_id": tile_id,
                "sample_id": sample_id,
                "z": z,
                "vol": vol,
                "labels_src_dir": labels_src_dir,
            }
        )

print(f"\nTotal samples across all tiles: {len(all_samples)}")
if not all_samples:
    raise RuntimeError("No samples were collected from the configured tiles.")

# ---- SPLIT INTO TRAIN/VAL ----
random.shuffle(all_samples)
n_train = int(len(all_samples) * train_ratio)
train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:]

print(f"train slices: {len(train_samples)} val slices: {len(val_samples)}")

# ---- EXPORT ----
def export_split(sample_list, img_dir, lab_dir):
    for sample in sample_list:
        tile_id = sample["tile_id"]
        sample_id = sample["sample_id"]
        z = sample["z"]
        vol = sample["vol"]
        labels_src_dir = sample["labels_src_dir"]

        # save image slice
        img_u8 = to_uint8(vol[z])
        img_name = f"tile_{sample_id}_z{z:04d}.png"
        Image.fromarray(img_u8).save(os.path.join(img_dir, img_name))

        # copy label txt (must match basename)
        src_txt = os.path.join(str(labels_src_dir), f"tile_{tile_id}_z{z:04d}.txt")
        dst_txt = os.path.join(lab_dir, f"tile_{sample_id}_z{z:04d}.txt")
        if os.path.exists(src_txt):
            shutil.copyfile(src_txt, dst_txt)
        else:
            Path(dst_txt).write_text("", encoding="utf-8")

export_split(train_samples, img_train, lab_train)
export_split(val_samples, img_val, lab_val)

data_yaml = Path(out_root) / "data.yaml"
data_yaml.write_text(
    "\n".join(
        [
            f"path: {out_root}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: neuron",
            "",
        ]
    ),
    encoding="utf-8",
)

print("Done exporting to:", out_root)
print("Wrote dataset yaml:", str(data_yaml))