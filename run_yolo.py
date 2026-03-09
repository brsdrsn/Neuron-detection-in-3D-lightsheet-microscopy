"""
Trial of how the basic YOLO model works on lightsheet data without training.
"""

import numpy as np
import tifffile as tiff
import napari
from ultralytics import YOLO

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)

# ---- LOAD 3D TILE (Z,Y,X) ----
tile_path = r"C:\Users\brsdr\Documents\neurons\tagged tiles\tile 3-3-0\tile.tif"
vol = tiff.imread(tile_path)

print(f"Loaded tile from: {tile_path}")
print(f"Tile shape (Z, Y, X): {vol.shape}, dtype: {vol.dtype}")

if vol.ndim != 3:
    raise ValueError(f"Expected a 3D stack TIFF. Got shape {vol.shape}")

Z, Y, X = vol.shape

# ---- YOLO MODEL ----
model = YOLO("yolov8n.pt")  # pretrained COCO, will likely be nonsense, but runs

# We'll collect:
# - rectangles in napari format: each rectangle is 4 corners in (z,y,x)
# - points (z,y,x) for each detection center
rects_zyx = []
points_zyx = []
conf_list = []

for z in range(Z):
    slice2d = vol[z]               # (Y,X)
    u8 = to_uint8(slice2d)
    img3 = np.stack([u8, u8, u8], axis=-1)  # (Y,X,3)

    res = model.predict(img3, imgsz=640, conf=0.25, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print(f"Slice {z}: no detections")
        continue
    else:
        print(f"Slice {z}: {len(res.boxes)} detections")

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        # rectangle corners: (z, y, x)
        rect = np.array([
            [z, y1, x1],
            [z, y1, x2],
            [z, y2, x2],
            [z, y2, x1],
        ], dtype=np.float32)

        rects_zyx.append(rect)

        # center point (z,y,x)
        yc = 0.5 * (y1 + y2)
        xc = 0.5 * (x1 + x2)
        points_zyx.append([z, yc, xc])

        conf_list.append(float(c))

print(f"Total rectangles collected: {len(rects_zyx)}")
print(f"Total points collected: {len(points_zyx)}")

# ---- VIEW IN NAPARI ----
viewer = napari.Viewer()
viewer.add_image(vol, name="tile (Z,Y,X)", rendering="mip")  # you can switch rendering modes

shapes = viewer.add_shapes(
    rects_zyx,
    shape_type="rectangle",
    edge_width=1.5,
    name="YOLO boxes (per Z)",
)

pts = viewer.add_points(
    np.array(points_zyx, dtype=np.float32),
    size=4,
    name="YOLO centers",
)

# store confidence as properties (optional)
if len(conf_list) > 0:
    shapes.properties = {"conf": np.array(conf_list, dtype=np.float32)}
    shapes.text = {"string": "conf", "size": 8, "color": "white"}

napari.run()