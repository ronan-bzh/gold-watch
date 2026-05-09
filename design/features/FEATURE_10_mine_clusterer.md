# Feature 10: Mine Clusterer

**Goal:** Group all 1,189 mining polygons from the geojson file into Sentinel-2 tile clusters so we know which scenes to download.

**Prerequisite:** `data/french_guiana_mines.geojson` must exist.

---

## What You Build

### Source Code

`src/goldmine_watch/data/mine_clusterer.py` — New module with:

```python
def load_mines(geojson_path: str = "data/french_guiana_mines.geojson") -> gpd.GeoDataFrame:
    """Load all mining polygons from geojson."""

def cluster_mines_by_tile(
    mines_gdf: gpd.GeoDataFrame,
    utm_zone: int = 22,
) -> dict[str, gpd.GeoDataFrame]:
    """Group mines by Sentinel-2 tile ID.
    
    Returns a dict mapping tile_id -> GeoDataFrame of mines in that tile.
    """

def get_tile_bbox(tile_id: str) -> tuple[float, float, float, float]:
    """Return WGS84 bbox for a given Sentinel-2 tile ID."""

def get_required_tiles(mines_gdf: gpd.GeoDataFrame) -> list[str]:
    """Return sorted list of unique Sentinel-2 tile IDs needed."""
```

### Unit Tests

`tests/unit/test_mine_clusterer.py`:

```python
class TestLoadMines:
    def test_loads_all_mines(self):
        """Should return 1,189 polygons."""
    
    def test_returns_geodataframe(self):
        """Should return a GeoDataFrame with geometry column."""

class TestClusterMinesByTile:
    def test_clusters_non_empty(self):
        """Should return at least one cluster."""
    
    def test_all_mines_assigned(self):
        """Sum of mines across clusters should equal total."""
    
    def test_known_tile_ids(self):
        """Should produce valid Sentinel-2 tile IDs (T21Nxx, T22Nxx)."""

class TestGetTileBBox:
    def test_returns_four_floats(self):
        """Should return (min_lon, min_lat, max_lon, max_lat)."""
    
    def test_bbox_valid_for_tile(self):
        """Bbox should cover the correct UTM zone grid square."""
```

### Functional Tests

`tests/functional/test_feature_10_mine_clusterer.py`:

```python
class TestFeature10MineClustererFlow:
    def test_full_clustering_pipeline(self):
        """Load -> cluster -> verify all mines assigned."""
    
    def test_cluster_sizes_vary(self):
        """Some tiles should have many mines, others few."""
    
    def test_all_clusters_within_french_guiana(self):
        """All cluster bboxes should be within French Guiana bounds."""
```

### Demo Script

`scripts/demo_feature10_clusterer.py`:

```bash
python scripts/demo_feature10_clusterer.py
```

Output:
```
Mine Clusterer Demo
===================
Loading 1,189 mining polygons...
Clustering by Sentinel-2 tile...

Required tiles: 5
  T21NZE: 234 mines
  T21NZF: 512 mines
  T21NZG: 287 mines
  T22NBL: 98 mines
  T22NBM: 58 mines

Total: 1,189 mines across 5 tiles
Coverage: ~252 km x 253 km
```

---

## Success Criteria

1. `pytest tests/unit/test_mine_clusterer.py -v` → **4 passed**
2. All 1,189 mines are assigned to at least one tile cluster
3. Cluster sizes vary (some tiles have more mines than others)
4. All tile IDs follow Sentinel-2 naming convention
5. No mines are lost or duplicated during clustering

---

## What You Learn

- Sentinel-2 tiling system (UTM zones, 100x100 km grid squares)
- Spatial clustering with GeoPandas
- French Guiana geography and UTM zones

---

## What You DON'T Build

- Automatic tile downloading
- Training patches
- Visualizations

**Time estimate:** 1–2 hours

---

## Notes

- French Guiana spans UTM zones 21 and 22.
- Sentinel-2 tiles are ~100x100 km, so 5-6 tiles cover all mines.
- Some mines near tile edges may overlap two tiles. Assign to the tile containing the centroid.
- Real-data tests use the actual `french_guiana_mines.geojson` file.
