# Public API — re-export the three Napari widgets registered in napari.yaml
from ._widget import tiler_widget, next_tile, prev_tile

__all__ = ["tiler_widget", "next_tile", "prev_tile"]