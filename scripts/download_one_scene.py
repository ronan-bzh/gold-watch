"""Download a single Sentinel-2 scene for manual inspection.

Usage:
    export COPERNICUS_CLIENT_ID="your-client-id"
    export COPERNICUS_CLIENT_SECRET="your-client-secret"
    python scripts/download_one_scene.py
"""

import os

from goldmine_watch.data.stac import download_one_scene

# Saint-Laurent-du-Maroni area, French Guiana (approximate bbox in EPSG:4326)
BBOX = (-54.1, 5.3, -53.9, 5.5)
DATE = "2023-01-01/2023-03-31"
OUTPUT = "data/raw/sentinel2_scene.tif"

if __name__ == "__main__":
    client_id = os.environ.get("COPERNICUS_CLIENT_ID")
    client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit(
            "Error: Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET environment variables"
        )

    print("Downloading one Sentinel-2 scene...")
    download_one_scene(
        bbox=BBOX,
        date=DATE,
        output_path=OUTPUT,
        client_id=client_id,
        client_secret=client_secret,
    )
    print("Done.")
