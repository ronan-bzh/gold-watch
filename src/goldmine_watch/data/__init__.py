"""Data ingestion, preprocessing, and patch generation.

Modules:
    ingest: Load mining surface labels from vector files.
    validate: Validate images, labels, and spatial alignment.
    stac: Download Sentinel-2 scenes from STAC catalogs.
    copernicus: Authenticate and download from Copernicus Data Space.
    patches: Generate georeferenced image patches and binary masks.
    cloud_mask: Load SCL band and create cloud masks.
    dataset: PyTorch Dataset for loading saved patches.
    tile_cache: Cache-first tile manager for training and inference.
    mine_clusterer: Group mining polygons by Sentinel-2 tile ID.
    build_training_dataset: Build complete multi-scene training dataset.
    square_postprocess: Convert probability rasters to square bounding boxes.
"""
