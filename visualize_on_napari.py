"""
Script to visualize the YOLO boxes and points on the napari viewer. This runs the YOLO model with the trained weights and shows the results in the napari viewer.
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

weights = r"C:\Users\brsdr\Documents\neurons\Napari\runs\detect\train8\weights\best.pt"
model = YOLO(weights)

tile_path = r"C:\Users\brsdr\Documents\neurons\tagged tiles\tile 2-2-0\Tile 2-2-0.tif"
vol = tiff.imread(tile_path)  # (Z,Y,X)
Z, Y, X = vol.shape

rects_zyx = []
points_zyx = []
scores = []

for z in range(Z):
    u8 = to_uint8(vol[z])
    img3 = np.stack([u8, u8, u8], axis=-1)

    res = model.predict(img3, imgsz=640, conf=0.1, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        continue

    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, conf):
        rects_zyx.append(np.array([[z, y1, x1],
                                   [z, y1, x2],
                                   [z, y2, x2],
                                   [z, y2, x1]], dtype=np.float32))
        points_zyx.append([z, 0.5*(y1+y2), 0.5*(x1+x2)])
        scores.append(float(c))

viewer = napari.Viewer()
viewer.add_image(vol, name="tile", rendering="mip")

shapes = viewer.add_shapes(rects_zyx, shape_type="rectangle", edge_width=1.5, name="YOLO boxes")
pts = viewer.add_points(np.array(points_zyx, dtype=np.float32), size=3, name="YOLO centers")

if scores:
    shapes.properties = {"conf": np.array(scores, dtype=np.float32)}
    shapes.text = {"string": "conf", "size": 8, "color": "white"}

napari.run()