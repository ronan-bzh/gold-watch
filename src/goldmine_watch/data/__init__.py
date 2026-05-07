"""Data ingestion, preprocessing, and patch generation.

Modules:
    ingest: Load mining surface labels from vector files.
    stac: Download Sentinel-2 scenes from STAC catalogs.
    patches: Generate georeferenced image patches and binary masks.
    dataset: PyTorch Dataset for loading saved patches.
"""
