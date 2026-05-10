"""SQLite-backed tile registry for GoldMine Watch.

Provides a single source of truth for cached Sentinel-2 tiles,
used by training, inference, and the web tile server.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import rasterio


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

        Args:
            db_path: Path to the SQLite database file.
            schema_path: Path to the SQL schema file.
        """
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._init_db()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Execute schema.sql against the SQLite connection."""
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {self.schema_path}. "
                "Ensure data/schema.sql exists in the repository."
            )
        schema_sql = self.schema_path.read_text()
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)

    def _conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _check_fg_intersection(
        self, west: float, south: float, east: float, north: float
    ) -> bool:
        """Return True if the given bbox intersects the FG boundary."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT west, south, east, north FROM fg_boundary LIMIT 1"
            ).fetchone()
        if row is None:
            # No boundary seeded — allow everything (should not happen)
            return True
        fg_w, fg_s, fg_e, fg_n = row
        return west < fg_e and east > fg_w and south < fg_n and north > fg_s

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def register_tile(
        self,
        tile_id: str,
        date: str,
        filepath: str,
        bounds: tuple[float, float, float, float],
        crs: str = "EPSG:4326",
        width: int | None = None,
        height: int | None = None,
        bands: int | None = None,
        size_bytes: int | None = None,
        source: str = "copernicus",
    ) -> int:
        """Insert a tile into the registry.

        Raises ValueError if bounds do not intersect French Guiana.

        Returns:
            The id of the inserted row.
        """
        west, south, east, north = bounds
        if not self._check_fg_intersection(west, south, east, north):
            raise ValueError(
                f"Tile {tile_id} bounds ({west}, {south}, {east}, {north}) "
                "do not intersect French Guiana."
            )

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO tiles
                    (tile_id, date, filepath, west, south, east, north,
                     crs, width, height, bands, size_bytes, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tile_id, date) DO UPDATE SET
                    filepath=excluded.filepath,
                    west=excluded.west,
                    south=excluded.south,
                    east=excluded.east,
                    north=excluded.north,
                    crs=excluded.crs,
                    width=excluded.width,
                    height=excluded.height,
                    bands=excluded.bands,
                    size_bytes=excluded.size_bytes,
                    source=excluded.source,
                    created_at=datetime('now')
                """,
                (
                    tile_id,
                    date,
                    filepath,
                    west,
                    south,
                    east,
                    north,
                    crs,
                    width,
                    height,
                    bands,
                    size_bytes,
                    source,
                ),
            )
            conn.commit()
            return cur.lastrowid or 0

    def get_tile(self, tile_id: str, date: str | None = None) -> dict[str, Any] | None:
        """Return tile record by ID. If date is None, return latest."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if date is not None:
                row = conn.execute(
                    "SELECT * FROM tiles WHERE tile_id = ? AND date = ?",
                    (tile_id, date),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM tiles WHERE tile_id = ? ORDER BY date DESC LIMIT 1",
                    (tile_id,),
                ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_tiles(
        self,
        intersects_bounds: tuple[float, float, float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """List all tiles, optionally filtered by bounding box intersection."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if intersects_bounds is not None:
                west, south, east, north = intersects_bounds
                rows = conn.execute(
                    """
                    SELECT * FROM tiles
                    WHERE west < ? AND east > ? AND south < ? AND north > ?
                    ORDER BY tile_id, date DESC
                    """,
                    (east, west, north, south),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tiles ORDER BY tile_id, date DESC"
                ).fetchall()
        return [dict(row) for row in rows]

    def list_for_viewport(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> list[dict[str, Any]]:
        """List tiles whose bounds intersect the given viewport."""
        return self.list_tiles(intersects_bounds=(west, south, east, north))

    def delete_tile(self, tile_id: str, date: str) -> None:
        """Remove a tile record from the registry."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM tiles WHERE tile_id = ? AND date = ?",
                (tile_id, date),
            )
            conn.commit()

    def refresh_from_disk(
        self,
        tiles_dir: str = "data/cache/tiles",
    ) -> int:
        """Scan *tiles_dir* for GeoTIFFs, validate, and auto-register any not in DB.

        Returns:
            Number of newly registered tiles.
        """
        tiles_path = Path(tiles_dir)
        if not tiles_path.exists():
            return 0

        registered = 0
        for tif_path in sorted(tiles_path.glob("*.tif")):
            # Derive tile_id and date from filename: <tile_id>_<date>.tif
            stem = tif_path.stem
            if "_" not in stem:
                continue
            tile_id, date = stem.split("_", 1)

            # Skip if already registered
            if self.get_tile(tile_id, date) is not None:
                continue

            # Validate and extract metadata
            try:
                with rasterio.open(tif_path) as src:
                    bounds = src.bounds
                    crs = str(src.crs)
                    width = src.width
                    height = src.height
                    bands = src.count
            except Exception:
                continue

            self.register_tile(
                tile_id=tile_id,
                date=date,
                filepath=str(tif_path),
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                crs=crs,
                width=width,
                height=height,
                bands=bands,
                size_bytes=tif_path.stat().st_size,
                source="copernicus",
            )
            registered += 1

        return registered

    def get_fg_boundary(self) -> dict[str, float | str]:
        """Return French Guiana bounding box."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM fg_boundary WHERE name = 'french_guiana' LIMIT 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("French Guiana boundary not found in registry.")
        return dict(row)
