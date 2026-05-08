import geopandas as gpd
import os

# Paths
input_path = "data/amazon_basin_48px_v3.7-ensemble_0.50_2023-01-01_2023-12-31.geojson"
output_path = "data/french_guiana_mines_2023_by_centroid.geojson"

# 1. Load mine polygons
print("Loading mine polygons...")
gdf_mines = gpd.read_file(input_path)
print(f"Loaded {len(gdf_mines)} polygons")

# 2. Fetch real French Guiana boundary from GADM
fg_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GUF_0.json"
print("Fetching French Guiana boundary from GADM...")
fg_boundary = gpd.read_file(fg_url)
print(f"Loaded boundary: {fg_boundary.iloc[0]['COUNTRY']} (GID_0: {fg_boundary.iloc[0]['GID_0']})")

# 3. Ensure CRS match
if fg_boundary.crs is None:
    fg_boundary.set_crs("EPSG:4326", inplace=True)
if gdf_mines.crs is None:
    gdf_mines.set_crs("EPSG:4326", inplace=True)
else:
    fg_boundary = fg_boundary.to_crs(gdf_mines.crs)

# 4. Compute centroids and filter by containment within FG boundary
print("Filtering by centroid containment...")
fg_union = fg_boundary.union_all()
# Use within for strict containment
mask = gdf_mines.centroid.within(fg_union)
gdf_fg = gdf_mines[mask].copy()

print(f"Kept {len(gdf_fg)} polygons whose centroid lies inside French Guiana")

# 5. Save result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
gdf_fg.to_file(output_path, driver="GeoJSON")
print(f"Saved filtered result to {output_path}")
