"""
This is the older script I used to try stuff on napari with zarr files. 
Still use it as the main script to open the zarr file in the napari viewer.
"""
import napari
import dask.array as da
import matplotlib.pyplot as plt


zarr_path = r"C:\Users\brsdr\Documents\neurons\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\2025_09_25_DiI2_piece_540_595_MFS2_2x2_50ms_fused.zarr\1"
data = da.from_zarr(zarr_path)


def open_in_napari(data):
    """Open a Dask array in Napari viewer."""
    viewer = napari.Viewer()
    viewer.add_image(data, name="neuron_3D")
    napari.run()

def get_dimentionality(data):
    """Prints the dimensionality information of the Dask array."""
    print("Array shape:", data.shape)
    print("Number of dimensions:", data.ndim)
    print("Datatype:", data.dtype)
    print("Chunk size:", data.chunksize)
    return data.ndim

def process_data(data):
    """Processes the Dask array. This makes the array shape same as the native nninteractive array shape(4 dim)."""
    data2 = data[0, 0, :, :, :]
    data_fixed = data2.map_blocks(lambda x: x.astype(x.dtype.newbyteorder("=")))
    return data_fixed

def tile(data2):
    """Tiles the Dask array into smaller chunks for efficient processing."""
    #tiles = data2.rechunk((128, 512, 512))
    #tiles = data2.rechunk((32, 128, 128))   
    tiles = data2.rechunk((512))                       
    return tiles

def visualize_in_python(data2, x):
    plt.imshow(data2[:, :, x].compute(), cmap='gray')
    plt.title(f"X-slice {x}")
    plt.axis('off')
    plt.show()

def mip(data2):
    """Maximum Intensity Projection along the x-axis."""
    mip = data2.max(axis=2).compute()
    plt.imshow(mip, cmap='gray')
    plt.title("X-axis Maximum Intensity Projection")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    processed_data = process_data(data)
    get_dimentionality(processed_data)  # Array shape: (2103, 1441, 124), Chunk size: (256, 256, 124)
    # mip(processed_data)
    tiles = tile(processed_data)
    print("Number of tiles:", tiles.numblocks)
    first_tile = tiles.blocks[0, 0, 0]
    print("tile shape:", first_tile.shape)  # tile shape: (128, 512, 124)
    print("starting napari viewer...")
    open_in_napari(first_tile)

"""
Problem with nnInteractive:
The NNU net that it initiates has a predetermined GPU allocation requirement of 11.39GB. My GPU is 8GB.
The error:
OutOfMemoryError: CUDA out of memory. Tried to allocate 11.39 GiB. 
GPU 0 has a total capacity of 8.00 GiB of which 2.85 GiB is free. 
Of the allocated memory 1.45 GiB is allocated by PyTorch, and 2.56 GiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
"""
