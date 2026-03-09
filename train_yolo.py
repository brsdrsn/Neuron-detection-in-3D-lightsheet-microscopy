"""
Script to train the YOLO model.
"""

# Train on only one tile for trial
import os
import shutil
import random
import numpy as np
import tifffile as tiff
from PIL import Image

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)

# ---- CONFIG ----
tile_id = "2-2-0"
tile_path = r"C:\Users\brsdr\Documents\neurons\tagged tiles\tile 2-2-0\Tile 2-2-0.tif"
labels_src_dir = r"C:\Users\brsdr\Documents\neurons\tagged tiles\tile 2-2-0\positive\yolo_labels_per_slice"

out_root = r"C:\Users\brsdr\Documents\neurons\yolo_one_tile"
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

# ---- LOAD TILE ----
vol = tiff.imread(tile_path)  # (Z,Y,X)
Z, H, W = vol.shape
print("Loaded tile:", vol.shape, vol.dtype)

# ---- COLLECT SLICES BASED ON LABEL FILE CONTENT ----
all_z = list(range(Z))
pos_z, neg_z = [], []

for z in all_z:
    txt = os.path.join(labels_src_dir, f"tile_{tile_id}_z{z:04d}.txt")
    if not os.path.exists(txt):
        # if missing, treat as negative (or raise error)
        neg_z.append(z)
        continue
    with open(txt, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content:
        pos_z.append(z)
    else:
        neg_z.append(z)

print("pos slices:", len(pos_z), "neg slices:", len(neg_z))

random.seed(seed)

if neg_per_pos is None:
    keep_z = all_z
else:
    target_neg = min(len(neg_z), neg_per_pos * len(pos_z))
    neg_sample = random.sample(neg_z, k=target_neg) if target_neg > 0 else []
    keep_z = sorted(pos_z + neg_sample)

print("keeping:", len(keep_z), "(pos", len(pos_z), "neg", len(keep_z)-len(pos_z), ")")

# ---- SPLIT INTO TRAIN/VAL ----
random.shuffle(keep_z)
n_train = int(len(keep_z) * train_ratio)
train_z = sorted(keep_z[:n_train])
val_z   = sorted(keep_z[n_train:])

print("train slices:", len(train_z), "val slices:", len(val_z))

# ---- EXPORT ----
def export_split(z_list, img_dir, lab_dir):
    for z in z_list:
        # save image slice
        img_u8 = to_uint8(vol[z])
        img_name = f"tile_{tile_id}_z{z:04d}.png"
        Image.fromarray(img_u8).save(os.path.join(img_dir, img_name))

        # copy label txt (must match basename)
        src_txt = os.path.join(labels_src_dir, f"tile_{tile_id}_z{z:04d}.txt")
        dst_txt = os.path.join(lab_dir, f"tile_{tile_id}_z{z:04d}.txt")
        shutil.copyfile(src_txt, dst_txt)

export_split(train_z, img_train, lab_train)
export_split(val_z, img_val, lab_val)

print("Done exporting to:", out_root)