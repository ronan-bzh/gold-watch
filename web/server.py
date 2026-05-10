"""FastAPI tile server for GoldMine Watch.

Serves cached Sentinel-2 GeoTIFFs as standard XYZ tiles at any zoom level,
using rio-tiler for dynamic resampling.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy
from PIL import Image
from rio_tiler.io import Reader
from rio_tiler.models import ImageData

from goldmine_watch.data.tile_registry import TileRegistry

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

DB_PATH = os.environ.get("TILES_DB", "data/cache/tiles.db")
SCHEMA_PATH = os.environ.get("TILES_SCHEMA", "data/schema.sql")
REFLECTANCE_CLIP = float(os.environ.get("REFLECTANCE_CLIP", "3000"))

app = FastAPI(title="GoldMine Watch Tile Server")
registry = TileRegistry(db_path=DB_PATH, schema_path=SCHEMA_PATH)

# --------------------------------------------------------------------------- #
#  Static web files
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
async def root():
    return FileResponse("web/index.html")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _mercator_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for a mercator tile."""
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    def _lat(y_tile: float) -> float:
        lat_rad = 3.14159265358979323846 - 2.0 * 3.14159265358979323846 * y_tile / n
        return 180.0 / 3.14159265358979323846 * (0.5 * (3.14159265358979323846 - lat_rad))

    north = _lat(y)
    south = _lat(y + 1)
    return (west, south, east, north)


def _natural_color(data: ImageData) -> ImageData:
    """Apply natural color stretch to rio-tiler ImageData.

    Expects B02, B03, B04 ordering (Blue, Green, Red) in the first 3 bands.
    Maps reflectance 0--REFLECTANCE_CLIP to 0--255.
    """
    arr = data.array
    if arr.shape[0] < 3:
        return data

    # Bands are [B02, B03, B04, ...] -> we want RGB = [B04, B03, B02]
    r = arr[2]
    g = arr[1]
    b = arr[0]

    clip = REFLECTANCE_CLIP
    r = (r / clip * 255).clip(0, 255).astype("uint8")
    g = (g / clip * 255).clip(0, 255).astype("uint8")
    b = (b / clip * 255).clip(0, 255).astype("uint8")

    rgb = ImageData(numpy.array([r, g, b]), data.mask)
    return rgb


def _pil_png(data: ImageData) -> bytes:
    """Convert rio-tiler ImageData to PNG bytes."""
    arr = data.as_ndarray()
    if arr.shape[0] == 1:
        mode = "L"
        img_arr = arr[0]
    else:
        mode = "RGB"
        # Transpose from (bands, h, w) to (h, w, bands)
        img_arr = arr.transpose(1, 2, 0)
    img = Image.fromarray(img_arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Tile API
# --------------------------------------------------------------------------- #


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Serve a dynamic XYZ tile from cached Sentinel-2 GeoTIFFs.

    Automatically composites multiple source tiles if viewport spans boundaries.
    Returns 204 if the requested tile does not intersect French Guiana.
    """
    west, south, east, north = _mercator_bbox(z, x, y)

    # French Guiana guard
    fg = registry.get_fg_boundary()
    if (
        east <= fg["west"]
        or west >= fg["east"]
        or north <= fg["south"]
        or south >= fg["north"]
    ):
        return Response(status_code=204)

    # Find source tiles intersecting this mercator tile
    source_tiles = registry.list_for_viewport(west, south, east, north)
    if not source_tiles:
        raise HTTPException(status_code=404, detail="No tiles cover this area")

    # Composite results from all intersecting source tiles
    composite: ImageData | None = None
    for tile in source_tiles:
        tile_path = Path(tile["filepath"])
        if not tile_path.exists():
            continue
        try:
            with Reader(tile_path) as reader:
                img = reader.tile(x, y, z)
        except Exception:
            continue

        if composite is None:
            composite = img
        else:
            # Simple average for overlapping regions
            composite = ImageData(
                (composite.array + img.array) / 2.0,
                composite.mask & img.mask,
            )

    if composite is None:
        raise HTTPException(status_code=404, detail="Unable to render tile")

    # Apply natural color stretch
    composite = _natural_color(composite)

    png_bytes = _pil_png(composite)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/tiles/info")
async def list_tiles():
    """Return JSON list of all registered tiles."""
    return {"tiles": registry.list_tiles()}


@app.get("/tiles/{tile_id}/info")
async def tile_info(tile_id: str):
    """Return metadata for a specific tile ID (latest date)."""
    tile = registry.get_tile(tile_id)
    if tile is None:
        raise HTTPException(status_code=404, detail=f"Tile {tile_id} not found")
    return tile


@app.get("/tiles/{tile_id}/preview.png")
async def tile_preview(tile_id: str):
    """Return a low-res full-tile preview image."""
    tile = registry.get_tile(tile_id)
    if tile is None:
        raise HTTPException(status_code=404, detail=f"Tile {tile_id} not found")

    tile_path = Path(tile["filepath"])
    if not tile_path.exists():
        raise HTTPException(status_code=404, detail=f"Tile file missing: {tile_path}")

    try:
        with Reader(tile_path) as reader:
            img = reader.preview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    img = _natural_color(img)
    png_bytes = _pil_png(img)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/tiles/refresh")
async def refresh_registry():
    """Re-scan data/cache/tiles and update registry."""
    count = registry.refresh_from_disk()
    return {"registered": count}


@app.get("/health")
async def health():
    return {"status": "ok"}
