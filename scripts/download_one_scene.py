"""Download a single Sentinel-2 scene for manual inspection.

Usage:
    python scripts/download_one_scene.py
"""

from goldmine_watch.data.stac import download_one_scene

# Saint-Laurent-du-Maroni area, French Guiana (approximate bbox in EPSG:4326)
BBOX = (-54.1, 5.3, -53.9, 5.5)
DATE = "2023-01-01/2023-03-31"
OUTPUT = "data/raw/sentinel2_scene.tif"

if __name__ == "__main__":
    print("Downloading one Sentinel-2 scene...")
    download_one_scene(bbox=BBOX, date=DATE, output_path=OUTPUT)
    print("Done.")
