# Feature 0: Copernicus Data Space Download

**Goal:** Download real Sentinel-2 L2A scenes from the official Copernicus Data Space (`dataspace.copernicus.eu`) using its STAC API, with proper authentication, area-of-interest search, and cloud-cover filtering.

**Prerequisite:** A free Copernicus Data Space account and a registered OAuth Client (client_id + client_secret).

---

## What You Build

### Source Code

`src/goldmine_watch/data/copernicus.py` — New module with three functions:

```python
def get_access_token(client_id: str, client_secret: str) -> str:
    """Authenticate with Copernicus Identity Server and return a bearer token.
    
    Uses the OAuth2 client_credentials flow with a registered OAuth Client.
    Token expires after 10 minutes; callers should refresh as needed.
    """

def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    token: str,
    max_cloud_cover: float = 20.0,
    collection: str = "sentinel-2-l2a",
) -> list[dict]:
    """Search the Copernicus STAC catalog for Sentinel-2 scenes.
    
    Returns a list of STAC item metadata dicts sorted by cloud cover (ascending).
    Raises RuntimeError if no scenes match.
    """

def download_scene(
    item: dict,
    token: str,
    output_path: Path,
    bands: list[str] | None = None,
    resolution: int = 10,
) -> Path:
    """Download a single scene's asset bands and stack them into a single GeoTIFF.
    
    Uses rasterio / rioxarray to read COG URLs and write a merged GeoTIFF.
    Returns the path to the saved file.
    """
```

### Tests

`tests/unit/test_copernicus_download.py`:

```python
class TestGetAccessToken:
    def test_token_returns_string(self, requests_mock):
        """Should return a non-empty bearer token on valid credentials."""
        
    def test_invalid_credentials_raises(self, requests_mock):
        """Should raise AuthenticationError on 401."""

class TestSearchScenes:
    def test_search_returns_items(self, requests_mock):
        """Should return list of STAC items for a valid query."""
        
    def test_no_scenes_raises(self, requests_mock):
        """Should raise RuntimeError when catalog returns empty."""
        
    def test_cloud_cover_filter(self, requests_mock):
        """Should exclude scenes with cloud cover above threshold."""

class TestDownloadScene:
    def test_download_writes_geotiff(self, tmp_path, requests_mock):
        """Should create a valid GeoTIFF at output_path."""
        
    def test_band_selection(self, tmp_path, requests_mock):
        """Should only download requested bands."""
```

### Functional Tests

`tests/functional/test_feature_0_download.py`:

```python
class TestFeature0DownloadFlow:
    def test_full_auth_search_download_flow(self, tmp_path, requests_mock):
        """Auth -> search -> download produces a valid GeoTIFF."""

    def test_search_cloud_filter_affects_results(self, requests_mock):
        """STAC search respects max_cloud_cover threshold."""

    def test_download_custom_bands_in_evalscript(self, tmp_path, requests_mock):
        """Custom band selection propagates into the Process API payload."""

    def test_download_custom_resolution_affects_size(self, tmp_path, requests_mock):
        """Resolution parameter changes requested width/height in payload."""

    def test_auth_failure_raises_authentication_error(self, requests_mock):
        """Invalid credentials raise AuthenticationError."""

    def test_empty_search_raises_runtime_error(self, requests_mock):
        """No matching scenes raise RuntimeError."""
```

### Demo Script

`scripts/demo_feature0_download.py`:

```bash
# 1. Set OAuth Client credentials via environment variables
export COPERNICUS_CLIENT_ID="your-client-id"
export COPERNICUS_CLIENT_SECRET="your-client-secret"

# 2. Run the demo
python scripts/demo_feature0_download.py \
  --bbox "-54.1,5.3,-53.9,5.5" \
  --date "2023-01-01/2023-01-31" \
  --output data/raw/sentinel2_scene.tif
```

Output:
```
Authenticating with Copernicus Data Space...
Search: bbox=[-54.1, 5.3, -53.9, 5.5] date=2023-01-01/2023-01-31 max_cloud=20%
Found 3 scenes:
  1. S2A_T22NLL_20230115T...  cloud: 5.2%
  2. S2B_T22NLL_20230110T...  cloud: 12.8%
  3. S2A_T22NLL_20230105T...  cloud: 18.1%
Downloading scene 1 (6 bands, 10m)...
Saved to data/raw/sentinel2_scene.tif
10980x10980 pixels | 6 bands | EPSG:32622 | 5.2% cloud cover
```

---

## Configuration Updates

Update `configs/mvp.yaml` under the `data:` section:

```yaml
data:
  # Copernicus Data Space OAuth Client (prefer env vars over hardcoded values)
  copernicus:
    auth_url: "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    stac_url: "https://catalogue.dataspace.copernicus.eu/stac"
    # Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET as environment variables
  
  # Update primary STAC endpoint
  stac_primary: "https://catalogue.dataspace.copernicus.eu/stac"
  stac_fallback: "https://planetarycomputer.microsoft.com/api/stac/v1"
```

---

## Success Criteria

1. `pytest tests/unit/test_copernicus_download.py -v` → **6 passed**
2. Running the demo script with valid credentials downloads a real Sentinel-2 scene
3. The downloaded GeoTIFF has the correct CRS, 6 bands, and 10m resolution
4. Invalid client credentials, empty search results, or network errors produce **clear error messages**
5. No hardcoded secrets in source code — client_id and client_secret always come from environment variables

---

## What You Learn

- How to authenticate with the Copernicus Identity Server using OAuth2 Clients
- How to query the Copernicus STAC catalog with spatial and temporal filters
- How to download Sentinel-2 COG assets and stack them into a single GeoTIFF
- How to handle token expiration and refresh

---

## What You DON'T Build

- Cloud masking (Feature 2)
- Image validation (Feature 1)
- Patch generation (Feature 3)
- Model training or inference

**Time estimate:** 2–3 hours

---

## Authentication Setup

1. Register at [https://dataspace.copernicus.eu](https://dataspace.copernicus.eu)
2. Go to **User Settings → OAuth Clients → Create New**
3. Give your client a name (e.g. `goldmine-watch`) and save
4. Copy the generated **Client ID** and **Client Secret**
5. Export them in your shell:
   ```bash
   export COPERNICUS_CLIENT_ID="your-client-id"
   export COPERNICUS_CLIENT_SECRET="your-client-secret"
   ```
6. (Optional) Add to your `.env` file (already in `.gitignore`)

---

## API Reference

- **Copernicus Data Space STAC API:** `https://catalogue.dataspace.copernicus.eu/stac`
- **Authentication:** OAuth2 client_credentials at `identity.dataspace.copernicus.eu`
- **Documentation:** [https://documentation.dataspace.copernicus.eu](https://documentation.dataspace.copernicus.eu)

---

## Notes

- Copernicus requires OAuth Client authentication; Planetary Computer does not. This feature makes the pipeline self-sufficient using the official ESA source.
- The Copernicus STAC API returns `sentinel-2-l2a` items with COG assets. Each band is a separate URL.
- Token lifetime is ~10 minutes. For long downloads, implement refresh logic.
- If Copernicus is rate-limiting, the fallback to Planetary Computer (already in `stac.py`) remains available.
