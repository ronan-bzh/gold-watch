# Feature 9: Unified Tile Cache System

**Goal:** Build a cache-first tile manager that downloads Sentinel-2 scenes once and reuses them for both training and inference.

**Prerequisite:** Copernicus Data Space credentials in `.env` file.

---

## What You Build

### Source Code

`src/goldmine_watch/data/tile_cache.py` — New module with:

```python
class TileCache:
    """Cache manager for Sentinel-2 tiles."""
    
    def __init__(self, cache_dir: str = "data/cache/tiles"):
        """Initialize cache directory."""
    
    def get_tile(
        self,
        tile_id: str,
        date_range: str,
        bbox: tuple[float, float, float, float],
        bands: list[str] | None = None,
    ) -> Path:
        """Return cached tile path or download if missing.
        
        Checks local cache first. Downloads via Copernicus STAC API
        only if tile is not present.
        """
    
    def list_cached_tiles(self) -> list[Path]:
        """List all tiles currently in cache."""
    
    def get_cache_size_mb(self) -> float:
        """Return total cache size in megabytes."""
    
    def clear_cache(self) -> None:
        """Remove all cached tiles."""


def get_tile_id_from_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Determine Sentinel-2 tile ID (e.g., T21NZF) from a WGS84 bbox.
    
    Uses UTM zone and grid square calculation.
    """
```

### Unit Tests

`tests/unit/test_tile_cache.py`:

```python
class TestTileCache:
    def test_cache_dir_created(self, tmp_path):
        """Should create cache directory if it doesn't exist."""
    
    def test_tile_not_in_cache_triggers_download(self, tmp_path, requests_mock):
        """Should download tile when not found in cache."""
    
    def test_tile_in_cache_returns_immediately(self, tmp_path):
        """Should return cached tile without downloading."""
    
    def test_list_cached_tiles_returns_paths(self, tmp_path):
        """Should return list of cached tile paths."""
    
    def test_cache_size_computed_correctly(self, tmp_path):
        """Should sum sizes of all cached files."""
    
    def test_clear_cache_removes_files(self, tmp_path):
        """Should delete all cached tiles."""

class TestGetTileIdFromBbox:
    def test_known_bbox_returns_correct_tile(self):
        """Saint-Laurent-du-Maroni bbox should return T21NZF or similar."""
    
    def test_invalid_bbox_raises(self):
        """Bbox outside Sentinel-2 coverage should raise ValueError."""
```

### Functional Tests

`tests/functional/test_feature_9_tile_cache.py`:

```python
class TestFeature9TileCacheFlow:
    def test_first_download_then_cache_reuse(self, tmp_path, requests_mock):
        """First call downloads, second call returns cached version."""
    
    def test_cache_avoids_duplicate_downloads(self, tmp_path, requests_mock):
        """Multiple requests for same tile should only download once."""
    
    def test_different_tiles_use_different_cache_keys(self, tmp_path, requests_mock):
        """Tiles with different IDs should not collide in cache."""
    
    def test_corrupted_cache_file_re_downloads(self, tmp_path, requests_mock):
        """Invalid/corrupted cached file should trigger re-download."""
    
    def test_env_credentials_required(self, tmp_path):
        """Missing COPERNICUS_CLIENT_ID should raise error."""
```

### Demo Script

`scripts/demo_feature9_tile_cache.py`:

```bash
# 1. Ensure credentials are set
export $(cat .env | xargs)

# 2. Run the demo
python scripts/demo_feature9_tile_cache.py \
  --bbox "-54.05,5.45,-54.0,5.49" \
  --date "2023-06-01/2023-12-31"
```

Output:
```
Tile Cache System Demo
======================
Checking cache at: data/cache/tiles/
Tile T21NZF_20231026 not in cache. Downloading...
Downloaded: 554x445 px | 7 bands | 350 MB
Cached to: data/cache/tiles/T21NZF_20231026.tif

Checking cache again...
Tile T21NZF_20231026 found in cache! (350 MB)
Cache reuse avoided 350 MB download.

Cache statistics:
  Tiles cached: 1
  Total size: 350.2 MB
```

---

## Configuration Updates

Update `configs/mvp.yaml`:

```yaml
cache:
  tiles_dir: "data/cache/tiles"
  predictions_dir: "data/cache/predictions"
  # Max cache size in GB before warnings
  max_size_gb: 10
```

---

## Success Criteria

1. `pytest tests/unit/test_tile_cache.py -v` → **6 passed**
2. First download takes ~2-5 minutes. Second request returns instantly from cache.
3. Cache directory structure is clean: `data/cache/tiles/<tile_id>_<date>.tif`
4. Missing `.env` credentials produce clear error message pointing to setup docs
5. Cache size reporting is accurate (within ±5%)

---

## What You Learn

- How to design a cache-first data pipeline
- Sentinel-2 tile naming convention (UTM zone + grid square)
- File-based caching strategies for large geospatial data

---

## What You DON'T Build

- Distributed/cloud cache (S3, GCS)
- Cache eviction policies (LRU, TTL)
- Multi-user cache locking

**Time estimate:** 2–3 hours

---

## Authentication Setup

1. Ensure `.env` file exists with valid credentials:
   ```bash
   export COPERNICUS_CLIENT_ID="your-client-id"
   export COPERNICUS_CLIENT_SECRET="your-client-secret"
   ```
2. Load before any demo or test:
   ```bash
   export $(cat .env | xargs)
   ```

---

## Notes

- The cache uses the filesystem, not a database. Simple and fast.
- Tile filenames include both tile ID and acquisition date to avoid collisions.
- For testing without real downloads, mock the Copernicus API responses.
- Real-data functional tests require valid `.env` credentials and network access.
