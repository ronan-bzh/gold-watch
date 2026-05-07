"""Data ingestion, preprocessing, and patch generation.

Modules:
    ingest: Load mining surface labels from vector files.
    validate: Validate images, labels, and spatial alignment.
    stac: Download Sentinel-2 scenes from STAC catalogs.
    copernicus: Authenticate and download from Copernicus Data Space.
    patches: Generate georeferenced image patches and binary masks.
    cloud_mask: Load SCL band and create cloud masks.
    dataset: PyTorch Dataset for loading saved patches.
"""
