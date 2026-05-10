# Feature 19: Unified Tile Registry & Dynamic Tile Server

**Goal:** Build a single SQLite-backed tile registry used by training, inference, and web display, plus a FastAPI + rio-tiler server that serves cached Sentinel-2 GeoTIFFs as standard XYZ tiles at any zoom level.

**Prerequisite:** Feature 9 (Unified Tile Cache) and Feature 16 (Web Map) must be complete.

---

## What You Build

### Source Code

#### `data/schema.sql` — Database schema (single source of truth)

The SQL schema lives in the repository as a plain text file. `TileRegistry` reads this file to initialize the database. This makes schema changes version-controlled and reviewable.

```sql
-- GoldMine Watch — Tile Registry Schema
-- This file is the single source of truth for the SQLite database structure.

CREATE TABLE IF NOT EXISTS tiles (
    id          INTEGER PRIMARY KEY,
    tile_id     TEXT NOT NULL,
    date        TEXT NOT NULL,
    filepath    TEXT NOT NULL UNIQUE,
    west        REAL NOT NULL,
    south       REAL NOT NULL,
    east        REAL NOT NULL,
    north       REAL NOT NULL,
    crs         TEXT NOT NULL DEFAULT 'EPSG:4326',
    width       INTEGER,
    height      INTEGER,
    bands       INTEGER,
    size_bytes  INTEGER,
    source      TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(tile_id, date)
);

CREATE INDEX IF NOT EXISTS idx_tiles_tile_id ON tiles(tile_id);
CREATE INDEX IF NOT EXISTS idx_tiles_bounds ON tiles(west, south, east, north);

CREATE TABLE IF NOT EXISTS fg_boundary (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    west        REAL NOT NULL,
    south       REAL NOT NULL,
    east        REAL NOT NULL,
    north       REAL NOT NULL
);

-- Seed French Guiana boundary on first init
INSERT OR IGNORE INTO fg_boundary (id, name, west, south, east, north)
VALUES (1, 'french_guiana', -54.6, 2.1, -51.6, 5.8);
```

---

#### `src/goldmine_watch/data/tile_registry.py` — SQLite registry

```python
class TileRegistry:
    """SQLite-backed catalog of cached Sentinel-2 tiles.
    
    Single source of truth for training, inference, and web display.
    Enforces French Guiana boundary on all inserts.
    
    The database schema is loaded from an external SQL file (data/schema.sql)
    so that schema changes are version-controlled.
    """
    
    def __init__(
        self,
        db_path: str = "data/cache/tiles.db",
        schema_path: str = "data/schema.sql",
    ):
        """Open or create registry database.
        
        If the database does not exist, it is created and initialized from
        *schema_path*. If it already exists, the schema is NOT re-applied
        (migrations are out of scope for MVP).
        """
    
    def _init_db(self) -> None:
        """Execute schema.sql against the SQLite connection."""
    
    def register_tile(
        self,
        tile_id: str,
        date: str,
        filepath: str,
        bounds: tuple[float, float, float, float],  # west, south, east, north
        crs: str = "EPSG:4326",
        width: int | None = None,
        height: int | None = None,
        bands: int | None = None,
        size_bytes: int | None = None,
        source: str = "copernicus",
    ) -> int:
        """Insert a tile into the registry.
        
        Raises ValueError if bounds do not intersect French Guiana.
        """
    
    def get_tile(self, tile_id: str, date: str | None = None) -> dict | None:
        """Return tile record by ID. If date is None, return latest."""
    
    def list_tiles(
        self,
        intersects_bounds: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """List all tiles, optionally filtered by bounding box intersection."""
    
    def list_for_viewport(
        self,
        west: float, south: float, east: float, north: float
    ) -> list[dict]:
        """List tiles whose bounds intersect the given viewport."""
    
    def delete_tile(self, tile_id: str, date: str) -> None:
        """Remove a tile record from the registry."""
    
    def refresh_from_disk(
        self,
        tiles_dir: str = "data/cache/tiles",
    ) -> int:
        """Scan tiles_dir for GeoTIFFs, validate, and auto-register any not in DB.
        
        Returns number of newly registered tiles.
        """
    
    def get_fg_boundary(self) -> dict:
        """Return French Guiana bounding box (west, south, east, north)."""
```

---

#### `src/goldmine_watch/data/tile_cache.py` — Updated

Add auto-registration on successful download:

```python
from goldmine_watch.data.tile_registry import TileRegistry

class TileCache:
    def __init__(self, cache_dir: str = "data/cache/tiles", db_path: str = "data/cache/tiles.db"):
        self.cache_dir = Path(cache_dir)
        self.registry = TileRegistry(db_path)
    
    def get_tile(self, tile_id: str, date_range: str, bbox: tuple, bands: list[str] | None = None) -> Path:
        """Cache-first lookup with auto-registration."""
        # 1. Check registry first (fast SQLite query)
        record = self.registry.get_tile(tile_id)
        if record and Path(record["filepath"]).exists():
            return Path(record["filepath"])
        
        # 2. Check filesystem (backward compatibility)
        cached = self._find_cached(tile_id)
        if cached:
            # Auto-register if found on disk but not in DB
            self.registry.register_tile(...)
            return cached
        
        # 3. Download
        downloaded = self._download(tile_id, date_range, bbox, bands)
        # 4. Auto-register
        self.registry.register_tile(...)
        return downloaded
```

---

#### `web/server.py` — FastAPI tile server

```python
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from rio_tiler.io import Reader
from PIL import Image
import io

from goldmine_watch.data.tile_registry import TileRegistry

# Database path is configurable via environment variable for Docker
DB_PATH = os.environ.get("TILES_DB", "data/cache/tiles.db")
SCHEMA_PATH = os.environ.get("TILES_SCHEMA", "data/schema.sql")

app = FastAPI(title="GoldMine Watch Tile Server")
registry = TileRegistry(db_path=DB_PATH, schema_path=SCHEMA_PATH)

# --- Static web files ---
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def root():
    return FileResponse("web/index.html")

# --- Tile API ---

@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Serve a dynamic XYZ tile from cached Sentinel-2 GeoTIFFs.
    
    Automatically composites multiple source tiles if viewport spans boundaries.
    Returns 204 if the requested tile does not intersect French Guiana.
    """

@app.get("/tiles/info")
async def list_tiles():
    """Return JSON list of all registered tiles."""

@app.get("/tiles/{tile_id}/info")
async def tile_info(tile_id: str):
    """Return metadata for a specific tile ID."""

@app.get("/tiles/{tile_id}/preview.png")
async def tile_preview(tile_id: str):
    """Return a low-res full-tile preview image."""

@app.post("/tiles/refresh")
async def refresh_registry():
    """Re-scan data/cache/tiles and update registry."""

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- French Guiana guard ---
# If mercator tile bbox does not intersect fg_boundary, return 204 No Content
```

**Dynamic tiling logic:**
1. Convert `z/x/y` to mercator bounding box.
2. Query `registry.list_for_viewport(...)` for intersecting source tiles.
3. If none intersect → return 204 (No Content).
4. If source tiles found → for each, open with `rio_tiler.Reader`, call `.tile(x, y, z)`.
5. Composite results (average overlapping pixels).
6. Apply natural color stretch: B04/B03/B02 → R/G/B, reflectance 0–3000 mapped to 0–255.
7. Return as PNG.

**Multi-tile compositing:** Server-side. If the viewport spans 2+ source GeoTIFFs, the server composites them into a single PNG before responding. This keeps the client simple (standard `L.tileLayer`).

**Overview strategy:** No pre-generation. `rio_tiler` handles on-the-fly resampling from full-resolution GeoTIFFs. If performance becomes an issue in Phase 3, add a background task to generate `.tif` overviews.

---

#### `web/app.js` — Updated

Replace GeoRasterLayer + manifest with standard Leaflet tile layer:

```javascript
// Remove: parseGeoraster dependency, tile_manifest.json fetch, GeoRasterLayer

// Sentinel-2 overlay via dynamic XYZ tile server
const copernicusLayer = L.tileLayer('/tiles/{z}/{x}/{y}.png', {
  attribution: 'Sentinel-2 / Copernicus',
  maxZoom: 14,
  opacity: 1.0,
});

// Dezoom is automatic — Leaflet requests lower z tiles, server resamples on the fly
```

Remove:
- `loadCopernicusLayer()` and all GeoRaster-related code
- `boundsIntersect()` helper
- `copernicusLayerGroup`
- `copernicusLoaded` / `copernicusLoading` state

Keep:
- Checkbox toggle for `copernicusLayer`
- Error handling if tile server is offline

---

#### `web/Dockerfile` — Updated

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python package (includes fastapi, uvicorn, rio-tiler)
COPY pyproject.toml ./
RUN pip install -e ".[web]"

# Copy source code, web assets, and SQL schema
COPY src/ ./src/
COPY web/ ./web/
COPY data/schema.sql ./data/schema.sql

# Ensure cache mount points exist for Docker Compose volumes
RUN mkdir -p /app/data/cache/tiles

# The SQLite DB will be created at runtime from schema.sql
ENV TILES_DB=/app/data/cache/tiles.db
ENV TILES_SCHEMA=/app/data/schema.sql

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

#### `web/docker-compose.yml` — Updated

```yaml
version: '3.8'

services:
  goldmine-watch:
    build:
      context: ..
      dockerfile: web/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      # Tiles are read-only (generated outside the container)
      - ../data/cache/tiles:/app/data/cache/tiles:ro
      # SQLite DB is persisted on a named volume (created inside Docker)
      - tiles-db:/app/data/cache/db
      # Schema is baked into the image, but allow override from host
      - ../data/schema.sql:/app/data/schema.sql:ro
    environment:
      - TILES_DB=/app/data/cache/db/tiles.db
      - TILES_SCHEMA=/app/data/schema.sql
    restart: unless-stopped

volumes:
  tiles-db:
    driver: local
```

**Docker architecture notes:**
- SQLite is a file-based database — there is no separate "SQLite service." The `.db` file is created inside the container from `schema.sql` on first startup and persisted via the named Docker volume `tiles-db`.
- The tiles themselves (`.tif` files) are mounted read-only from the host because they are generated by the training/inference pipeline outside the container.
- If you need to reset the database: `docker compose down -v` to destroy the named volume, then `docker compose up` to recreate it from `schema.sql`.

---

### Integration Points

#### Training (`build_training_dataset.py`)

**Current:** Globs filesystem directly.
**New:** Query registry.

```python
from goldmine_watch.data.tile_registry import TileRegistry

reg = TileRegistry(
    db_path=os.environ.get("TILES_DB", "data/cache/tiles.db"),
    schema_path="data/schema.sql",
)
tile = reg.get_tile(tile_id)
if not tile:
    # Fall back to TileCache (downloads + auto-registers)
    tile_path = cache.get_tile(tile_id, date_range, bbox)
else:
    tile_path = Path(tile["filepath"])
```

#### Inference (`batch.py`, `predict_big.py`)

**Current:** Globs filesystem.
**New:** Query registry.

```python
reg = TileRegistry(
    db_path=os.environ.get("TILES_DB", "data/cache/tiles.db"),
    schema_path="data/schema.sql",
)
for tile in reg.list_tiles():  # FG-filtered by default
    predict_big_image(tile["filepath"], model, output_dir)
```

#### Web (`app.js`)

**Current:** Fetches `tile_manifest.json`, loads `.tif` with GeoRasterLayer.
**New:** Standard Leaflet `L.tileLayer('/tiles/{z}/{x}/{y}.png')`.

---

### Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
web = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "rio-tiler>=6.0",
    "pillow>=10.0",
]
dev = [
    # ... existing dev deps ...
    "httpx",  # for testing FastAPI endpoints
]
```

---

## Unit Tests

`tests/unit/test_tile_registry.py`:

```python
class TestTileRegistry:
    def test_registry_creates_database_from_sql_file(self, tmp_path):
        """Should create SQLite DB by executing schema.sql."""
    
    def test_register_tile_inserts_record(self, tmp_path):
        """Should store tile metadata in SQLite."""
    
    def test_register_tile_rejects_out_of_bounds(self, tmp_path):
        """Tile outside French Guiana should raise ValueError."""
    
    def test_get_tile_returns_latest_by_default(self, tmp_path):
        """Should return most recent tile when date not specified."""
    
    def test_list_tiles_filters_by_viewport(self, tmp_path):
        """Should only return tiles intersecting given bounds."""
    
    def test_refresh_from_disk_finds_new_files(self, tmp_path):
        """Should auto-register GeoTIFFs found on disk."""
    
    def test_delete_tile_removes_record(self, tmp_path):
        """Should remove tile from registry."""
    
    def test_fg_boundary_is_seeded_from_sql(self, tmp_path):
        """Registry should contain French Guiana boundary after init."""
```

`tests/unit/test_tile_server.py`:

```python
class TestTileServer:
    def test_health_endpoint(self):
        """/health should return 200 with status ok."""
    
    def test_tiles_info_returns_json(self):
        """/tiles/info should return list of registered tiles."""
    
    def test_tile_endpoint_returns_png(self):
        """/tiles/{z}/{x}/{y}.png should return valid PNG for cached tile."""
    
    def test_tile_outside_fg_returns_204(self):
        """Request for tile outside French Guiana should return 204."""
    
    def test_tile_for_missing_tile_returns_404(self):
        """Request for non-existent tile should return 404."""
    
    def test_refresh_endpoint_updates_registry(self):
        """POST /tiles/refresh should scan disk and update DB."""
```

---

## Functional Tests

`tests/functional/test_feature_19_tile_registry.py`:

```python
class TestFeature19TileRegistryFlow:
    def test_end_to_end_register_and_serve(self, tmp_path):
        """Register a tile, then request it via /tiles/{z}/{x}/{y}.png."""
    
    def test_training_reads_from_registry(self):
        """build_training_dataset should use TileRegistry to find tiles."""
    
    def test_inference_reads_from_registry(self):
        """batch.py should use TileRegistry to iterate tiles."""
    
    def test_registry_auto_registers_downloads(self):
        """TileCache.download() should auto-insert into registry."""
    
    def test_web_map_dezoom_loads_lower_zoom(self):
        """Dezooming in browser should request lower-z tiles successfully."""
```

---

## Demo Script

`scripts/demo_feature19_tile_server.py`:

```bash
# 1. Ensure tiles are cached (or run Feature 9 demo first)
ls data/cache/tiles/*.tif

# 2. Build and start the Docker container (DB is auto-created from schema.sql)
docker compose -f web/docker-compose.yml up --build -d

# 3. Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/tiles/info | python -m json.tool
curl -o /tmp/test_tile.png "http://localhost:8000/tiles/10/300/500.png"
file /tmp/test_tile.png  # Should be PNG

# 4. Refresh registry if new tiles were added to the host cache
curl -X POST http://localhost:8000/tiles/refresh

# 5. Open web map
open http://localhost:8000
# Zoom in/out — dezoom automatically loads lower-resolution tiles

# 6. Stop and reset database if needed
# docker compose -f web/docker-compose.yml down -v  # destroys DB volume
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
cache:
  tiles_dir: "data/cache/tiles"
  tiles_db: "data/cache/tiles.db"
  tiles_schema: "data/schema.sql"
  predictions_dir: "data/cache/predictions"
  max_size_gb: 10

inference:
  # ... existing ...
  
web:
  tile_server:
    host: "0.0.0.0"
    port: 8000
    max_zoom: 14
    # Natural color stretch: reflectance values above this are clipped to white
    reflectance_clip: 3000
```

---

## French Guiana Enforcement

Three layers:

1. **Registry insert** — `register_tile()` checks intersection with `fg_boundary` before writing. Raises `ValueError` if outside.
2. **TileCache download** — `get_tile()` rejects downloads for bboxes not intersecting FG before calling Copernicus API.
3. **Tile server** — `/tiles/{z}/{x}/{y}.png` returns HTTP 204 if the mercator tile does not intersect FG boundary.

---

## Migration Path

| Step | Action |
|------|--------|
| 1 | Add `fastapi`, `uvicorn`, `rio-tiler`, `pillow` to `pyproject.toml [project.optional-dependencies] web` |
| 2 | Create `data/schema.sql` with the schema above |
| 3 | Create `src/goldmine_watch/data/tile_registry.py` that reads `schema.sql` |
| 4 | Update `src/goldmine_watch/data/tile_cache.py` to auto-register on download |
| 5 | Create `web/server.py` (FastAPI + rio-tiler) |
| 6 | Rewrite `web/app.js` to use `L.tileLayer('/tiles/{z}/{x}/{y}.png')` |
| 7 | Update `web/Dockerfile` and `web/docker-compose.yml` |
| 8 | Update `scripts/build_training_dataset.py` to use registry |
| 9 | Update `src/goldmine_watch/inference/batch.py` to use registry |
| 10 | Remove `web/data/tile_manifest.json` and `web/data/cache/` symlink |
| 11 | Add unit tests for registry and server |
| 12 | Seed registry: `docker compose -f web/docker-compose.yml exec goldmine-watch python -c "from goldmine_watch.data.tile_registry import TileRegistry; TileRegistry().refresh_from_disk()"` |

---

## Success Criteria

1. `pytest tests/unit/test_tile_registry.py -v` → **8 passed**
2. `pytest tests/unit/test_tile_server.py -v` → **6 passed**
3. `data/schema.sql` exists and is version-controlled
4. Registry contains all cached tiles after `refresh_from_disk()`
5. `/tiles/{z}/{x}/{y}.png` returns valid PNG for tiles inside French Guiana
6. `/tiles/{z}/{x}/{y}.png` returns HTTP 204 for tiles outside French Guiana
7. Web map loads Sentinel-2 overlay at any zoom level via standard `L.tileLayer`
8. Dezooming in browser loads lower-resolution tiles without full-page reload
9. Training and inference scripts read tile paths from registry (not filesystem globs)
10. `docker compose -f web/docker-compose.yml up` creates and persists the SQLite DB inside Docker
11. No authentication required for any endpoint

---

## What You Learn

- SQLite as a lightweight geospatial catalog
- External SQL schema files for version-controlled database structure
- Docker volume persistence for file-based databases
- rio-tiler for dynamic COG/GeoTIFF tiling
- FastAPI for serving XYZ tile endpoints
- Server-side raster compositing
- Standard Leaflet XYZ integration

---

## What You DON'T Build

- Cloud tile store (S3, GCS)
- Authentication / authorization
- Tile caching layer (Redis, CDN) — rio-tiler internal caching is sufficient for MVP
- Pre-generated overviews / pyramids (rio-tiler handles on-the-fly resampling)
- Mobile app
- Database migration system (schema.sql is applied once on first init)

**Time estimate:** 4–6 hours

---

## Notes

- `rio_tiler.Reader.tile()` is fast enough for MVP because our tiles are small (~7–50 MB). If you add dozens of tiles, consider COG internal overviews or pre-generating `.tif` mosaics per zoom level.
- The French Guiana boundary is stored as a simple bbox in SQLite. If you need sub-tile precision (e.g. rejecting a tile that barely clips FG), replace the bbox with a GeoJSON polygon and use `shapely` intersection checks.
- No authentication means the tile server is open. Add a reverse proxy (nginx, traefik) if you need rate limiting or IP allowlisting in production.
- The server composites multiple source tiles server-side. This simplifies the client but costs ~10–50 ms extra per tile request. Acceptable for MVP.
- Keep `web/data/copernicus_scene.tif` removed (it was deleted in the previous task). The server only serves from `data/cache/tiles/` via the registry.
- To inspect the Docker-managed SQLite DB from the host: `docker compose -f web/docker-compose.yml exec goldmine-watch sqlite3 /app/data/cache/db/tiles.db "SELECT * FROM tiles;"`
