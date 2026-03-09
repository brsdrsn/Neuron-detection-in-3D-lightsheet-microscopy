"""
Script to pre-process the tagged masks for the YOLO model.
"""

import os
import glob
import numpy as np
import tifffile as tiff

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

# ---- CONFIG ----
tile_id = "2-2-0"
folder = r"C:\Users\brsdr\Documents\neurons\tagged tiles\tile 2-2-0\positive"

# ---- 1) COMBINE PER-OBJECT 3D MASKS INTO ONE INSTANCE VOLUME ----
obj_paths = sorted(glob.glob(os.path.join(folder, f"object * {tile_id}.tif")))
if len(obj_paths) == 0:
    raise FileNotFoundError(f"No object masks found for tile_id={tile_id} in {folder}")

first = tiff.imread(obj_paths[0])
if first.ndim != 3:
    raise ValueError(f"Expected 3D mask, got shape {first.shape}")
Z, Y, X = first.shape

dtype = np.uint16 if len(obj_paths) < 65535 else np.uint32
inst = np.zeros((Z, Y, X), dtype=dtype)

next_id = 1
total_overlap = 0

for p in obj_paths:
    m = tiff.imread(p)
    if m.shape != inst.shape:
        raise ValueError(f"Shape mismatch: {p} has {m.shape}, expected {inst.shape}")
    if not np.any(m):
        continue  # skip empty masks

    # overlap diagnostics (optional)
    total_overlap += int(np.count_nonzero((inst > 0) & (m > 0)))

    # write instance id (overwrite policy)
    inst[m > 0] = next_id
    next_id += 1

out_path = os.path.join(folder, f"instances_{tile_id}.tif")
tiff.imwrite(out_path, inst, compression="zlib")

num_instances = next_id - 1
print(f"Wrote instance volume: {out_path}")
print(f"Instances written: {num_instances}")
print(f"Overlap voxels (diagnostic): {total_overlap}")

# ---- 2) CONVERT INSTANCE VOLUME TO YOLO TXT LABELS (PER SLICE) ----
Z, H, W = inst.shape

labels_out = os.path.join(folder, "yolo_labels_per_slice")
os.makedirs(labels_out, exist_ok=True)

for z in range(Z):
    inst2d = inst[z]

    # only consider ids present on this slice
    ids = np.unique(inst2d)
    ids = ids[ids != 0]  # remove background

    lines = []
    for obj_id in ids:
        bb = bbox2d_from_instance_slice(inst2d, int(obj_id))
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        lines.append(yolo_line(x1, y1, x2, y2, W=W, H=H, cls=0))

    out_txt = os.path.join(labels_out, f"tile_{tile_id}_z{z:04d}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("Wrote YOLO txt files to:", labels_out)