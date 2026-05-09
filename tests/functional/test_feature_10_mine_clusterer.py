"""Functional tests for Feature 10: Mine Clusterer.

These tests exercise the real ``data/french_guiana_mines.geojson`` file.
"""

from __future__ import annotations

import pandas as pd

from goldmine_watch.data.mine_clusterer import (
    cluster_mines_by_tile,
    compute_coverage_km,
    get_required_tiles,
    get_tile_bbox,
    load_mines,
)

# French Guiana approximate bounds (WGS84)
FG_MIN_LON = -54.6
FG_MAX_LON = -51.5
FG_MIN_LAT = 2.0
FG_MAX_LAT = 6.0


class TestFeature10MineClustererFlow:
    """End-to-end functional tests using real mining labels."""

    def test_full_clustering_pipeline(self) -> None:
        """Load -> cluster -> verify all mines assigned."""
        gdf = load_mines("data/french_guiana_mines.geojson")
        assert len(gdf) == 1189

        clusters = cluster_mines_by_tile(gdf)
        total = sum(len(c) for c in clusters.values())
        assert total == 1189, f"Expected 1189 mines, got {total}"

    def test_cluster_sizes_vary(self) -> None:
        """Some tiles should have many mines, others few."""
        gdf = load_mines("data/french_guiana_mines.geojson")
        clusters = cluster_mines_by_tile(gdf)
        sizes = [len(c) for c in clusters.values()]
        assert max(sizes) > min(sizes), "All clusters have the same size"

    def test_all_clusters_within_french_guiana(self) -> None:
        """All tile bboxes should overlap French Guiana bounds."""
        gdf = load_mines("data/french_guiana_mines.geojson")
        tiles = get_required_tiles(gdf)
        for tile_id in tiles:
            min_lon, min_lat, max_lon, max_lat = get_tile_bbox(tile_id)
            # Tile bboxes are ~100 km grid squares; they may extend beyond the
            # country boundary, but must *overlap* French Guiana.
            overlaps = (
                min_lon < FG_MAX_LON
                and max_lon > FG_MIN_LON
                and min_lat < FG_MAX_LAT
                and max_lat > FG_MIN_LAT
            )
            assert overlaps, f"{tile_id} bbox does not overlap French Guiana"

    def test_no_mines_lost_or_duplicated(self) -> None:
        """Every original mine ID should appear in exactly one cluster."""
        gdf = load_mines("data/french_guiana_mines.geojson")
        clusters = cluster_mines_by_tile(gdf)
        combined = pd.concat(clusters.values(), ignore_index=True)
        assert len(combined) == len(gdf)
        # If the original has an 'id' or index we could check uniqueness,
        # but the GeoJSON does not have a guaranteed unique ID column.
        # We rely on the count equality tested above.

    def test_coverage_computed(self) -> None:
        """Coverage helper should return positive width and height."""
        gdf = load_mines("data/french_guiana_mines.geojson")
        tiles = get_required_tiles(gdf)
        width_km, height_km = compute_coverage_km(tiles)
        assert width_km > 0
        assert height_km > 0
        assert width_km > 100  # Should be at least one tile wide
        assert height_km > 100  # Should be at least one tile tall
