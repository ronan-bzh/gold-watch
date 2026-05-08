import geopandas as gpd
from shapely.geometry import box
import os

# Paths
input_path = "data/amazon_basin_48px_v3.7-ensemble_0.50_2023-01-01_2023-12-31.geojson"
output_path = "data/french_guiana_mines_2023.geojson"

# Load mine polygons
print("Loading mine polygons...")
gdf_mines = gpd.read_file(input_path)
print(f"Loaded {len(gdf_mines)} polygons")

# Try to get French Guiana boundary from Natural Earth
boundary_urls = [
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
]

fg_boundary = None

for url in boundary_urls:
    try:
        print(f"Trying boundary source: {url}")
        world = gpd.read_file(url)
        # Filter for French Guiana
        # Common columns: SOVEREIGNT, ADMIN, ISO_A3, NAME
        for col in ["ADMIN", "SOVEREIGNT", "NAME", "ISO_A3"]:
            if col in world.columns:
                match = world[world[col] == "French Guiana"]
                if not match.empty:
                    fg_boundary = match.copy()
                    print(f"Found French Guiana boundary via {col}='French Guiana' from {url}")
                    break
                # Also try ISO code GUF
                if col == "ISO_A3":
                    match = world[world[col] == "GUF"]
                    if not match.empty:
                        fg_boundary = match.copy()
                        print(f"Found French Guiana boundary via ISO_A3='GUF' from {url}")
                        break
        if fg_boundary is not None:
            break
    except Exception as e:
        print(f"Failed: {e}")
        continue

if fg_boundary is None:
    print("Using bounding box fallback for French Guiana.")
    bbox = box(-54.6, 2.1, -51.6, 5.8)
    fg_boundary = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

# Ensure CRS match
if fg_boundary.crs is None:
    fg_boundary.set_crs("EPSG:4326", inplace=True)

if gdf_mines.crs is None:
    gdf_mines.set_crs("EPSG:4326", inplace=True)
else:
    fg_boundary = fg_boundary.to_crs(gdf_mines.crs)

# Spatial filter: keep only mines intersecting French Guiana boundary
print("Filtering polygons within French Guiana...")
fg_union = fg_boundary.union_all()
mask = gdf_mines.intersects(fg_union)
gdf_fg = gdf_mines[mask].copy()

print(f"Kept {len(gdf_fg)} polygons in French Guiana")

# Save result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
gdf_fg.to_file(output_path, driver="GeoJSON")
print(f"Saved filtered result to {output_path}")
