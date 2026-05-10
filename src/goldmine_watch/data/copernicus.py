"""Copernicus Data Space client for downloading Sentinel-2 scenes.

Uses the Sentinel Hub Process API within the Copernicus Data Space ecosystem
to search the STAC catalog and download multi-band GeoTIFFs.

Authentication is via OAuth2 client_credentials flow.
"""

from datetime import datetime, timedelta
from pathlib import Path

import rasterio
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

_AUTH_URL = (
    "https://identity.dataspace.copernicus.eu" "/auth/realms/CDSE/protocol/openid-connect/token"
)
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


class AuthenticationError(Exception):
    """Raised when Copernicus authentication fails."""


def get_access_token(client_id: str, client_secret: str) -> str:
    """Authenticate with Copernicus Identity Server and return a bearer token.

    Uses the OAuth2 client_credentials flow with a registered OAuth Client.
    Token expires after 10 minutes; callers should refresh as needed.

    Args:
        client_id: Copernicus Data Space OAuth Client ID.
        client_secret: Copernicus Data Space OAuth Client Secret.

    Returns:
        Bearer access token string.

    Raises:
        AuthenticationError: If credentials are invalid or authentication fails.
    """
    oauth_client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=oauth_client)
    try:
        token = oauth.fetch_token(
            token_url=_AUTH_URL,
            client_secret=client_secret,
            include_client_id=True,
        )
    except Exception as exc:
        raise AuthenticationError(f"Invalid Copernicus client credentials: {exc}") from exc
    return str(token["access_token"])


def _normalize_datetime_range(date_range: str) -> str:
    """Convert shorthand date ranges to ISO-8601 datetime strings.

    The Copernicus STAC API requires ``T`` separators (e.g.
    ``2024-01-01T00:00:00Z/2024-01-31T23:59:59Z``). This helper accepts the
    common ``YYYY-MM-DD/YYYY-MM-DD`` shorthand and expands it automatically.
    """
    if "/" not in date_range:
        return date_range
    start, end = date_range.split("/", 1)
    start = start.strip()
    end = end.strip()
    # Expand bare dates to full ISO-8601 datetimes
    if len(start) == 10 and start.count("-") == 2:
        start = f"{start}T00:00:00Z"
    if len(end) == 10 and end.count("-") == 2:
        end = f"{end}T23:59:59Z"
    return f"{start}/{end}"


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    token: str,
    max_cloud_cover: float = 20.0,
    collection: str = "sentinel-2-l2a",
) -> list[dict]:
    """Search the Copernicus STAC catalog for Sentinel-2 scenes.

    Args:
        bbox: Bounding box as (min_x, min_y, max_x, max_y) in EPSG:4326.
        date_range: Date or date range string (e.g. "2023-01-01/2023-01-31").
        token: Valid bearer token from :func:`get_access_token`.
        max_cloud_cover: Maximum allowed cloud cover percentage.
        collection: STAC collection ID to search.

    Returns:
        List of STAC item metadata dicts sorted by cloud cover (ascending).

    Raises:
        RuntimeError: If no scenes match the query.
        requests.HTTPError: For HTTP errors from the STAC API.
    """
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": _normalize_datetime_range(date_range),
        "limit": 100,
    }

    response = requests.post(_STAC_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    items = data.get("features", [])

    filtered = [
        item
        for item in items
        if item.get("properties", {}).get("eo:cloud_cover", 100) <= max_cloud_cover
    ]

    if not filtered:
        raise RuntimeError(f"No scenes found for bbox={bbox} date={date_range}")

    filtered.sort(key=lambda x: x.get("properties", {}).get("eo:cloud_cover", 100))
    return filtered


def _build_evalscript(bands: list[str]) -> str:
    """Build a Sentinel Hub evalscript that returns the requested bands.

    Uses the ``DATASET`` input declaration required by the Copernicus Data
    Space Sentinel Hub Process API.
    """
    band_list = ", ".join(f'"{b}"' for b in bands)
    return_values = ", ".join(f"sample.{b}" for b in bands)
    return (
        f"//VERSION=3\n"
        f"function setup() {{\n"
        f"  return {{\n"
        f"    input: [{{\n"
        f"      bands: [{band_list}],\n"
        f'      units: "DN"\n'
        f"    }}],\n"
        f'    output: {{ bands: {len(bands)}, sampleType: "UINT16" }}\n'
        f"  }};\n"
        f"}}\n"
        f"function evaluatePixel(sample) {{\n"
        f"  return [{return_values}];\n"
        f"}}\n"
    )


def _bbox_size_pixels(
    bbox: tuple[float, float, float, float], resolution_m: float
) -> tuple[int, int]:
    """Calculate approximate width/height in pixels for a WGS84 bbox.

    Uses rough approximations:
    - 1° latitude ≈ 111,320 m
    - 1° longitude ≈ 111,320 m * cos(latitude)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_mid = (min_lat + max_lat) / 2
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * __import__("math").cos(__import__("math").radians(lat_mid))
    width_m = (max_lon - min_lon) * meters_per_deg_lon
    height_m = (max_lat - min_lat) * meters_per_deg_lat
    width_px = max(1, int(round(width_m / resolution_m)))
    height_px = max(1, int(round(height_m / resolution_m)))
    return width_px, height_px


def download_scene(
    item: dict,
    token: str,
    output_path: Path,
    bands: list[str] | None = None,
    resolution: int = 10,
    bbox: tuple[float, float, float, float] | None = None,
    time_range: dict[str, str] | None = None,
) -> Path:
    """Download a single scene via Sentinel Hub Process API.

    Uses the Sentinel Hub Process API within the Copernicus Data Space
    ecosystem to request a multi-band GeoTIFF.

    Args:
        item: STAC item dict containing scene metadata.
        token: Valid bearer token for authenticated access.
        output_path: Where to save the output GeoTIFF.
        bands: List of band names to retrieve. Defaults to RGB + NIR + SWIR.
        resolution: Target spatial resolution in meters.
        bbox: Bounding box to download. Defaults to the item's full footprint.
        time_range: Optional explicit ``{"from": "...", "to": "..."}``
            time range for the Process API request. When ``None``, a ±1 day
            window around the item's acquisition datetime is used.

    Returns:
        Path to the saved GeoTIFF.

    Raises:
        ValueError: If the STAC item is missing required fields.
        RuntimeError: If the downloaded file doesn't have the expected bands.
        requests.HTTPError: If the Process API request fails.
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

    # Ensure SCL is always included for cloud masking
    if "SCL" not in bands:
        bands = bands + ["SCL"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if bbox is None:
        bbox = item.get("bbox", [])
    if not bbox:
        raise ValueError("STAC item missing bbox")

    properties = item.get("properties", {})
    datetime_str = properties.get("datetime", "")
    if not datetime_str:
        raise ValueError("STAC item missing datetime")

    if time_range is not None:
        dt_from = time_range["from"]
        dt_to = time_range["to"]
    else:
        # Build a ±1-day time range around the acquisition
        dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        dt_from = (dt - timedelta(days=1)).isoformat().replace("+00:00", "Z")
        dt_to = (dt + timedelta(days=1)).isoformat().replace("+00:00", "Z")

    width_px, height_px = _bbox_size_pixels(bbox, resolution)

    # Sentinel Hub Process API has a 2500px limit per dimension.
    # If the request exceeds this, scale down resolution proportionally
    # so that the largest dimension is exactly 2500px.
    max_dim = max(width_px, height_px)
    if max_dim > 2500:
        scale_factor = 2500 / max_dim
        width_px = int(width_px * scale_factor)
        height_px = int(height_px * scale_factor)
        # Recalculate the effective resolution for logging
        effective_res = resolution / scale_factor
        print(f"  [download_scene] Request exceeds 2500px limit ({max_dim}px). "
              f"Scaling to {width_px}x{height_px} (effective res ~{effective_res:.1f}m)")

    evalscript = _build_evalscript(bands)

    payload = {
        "input": {
            "bounds": {
                "bbox": list(bbox),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": dt_from,
                            "to": dt_to,
                        },
                        "maxCloudCoverage": properties.get("eo:cloud_cover", 100),
                    },
                }
            ],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": evalscript,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(_PROCESS_URL, json=payload, headers=headers, timeout=300)
    if response.status_code == 400:
        # Log the actual error response for debugging
        error_text = response.text[:2000] if response.text else "(empty body)"
        raise RuntimeError(
            f"Copernicus Process API returned 400. "
            f"Request size: {width_px}x{height_px}px ({len(bands)} bands). "
            f"Error: {error_text}"
        )
    response.raise_for_status()

    with open(output_path, "wb") as dst:
        dst.write(response.content)

    # Verify the output and tag the SCL band
    with rasterio.open(output_path, "r+") as src:
        if src.count != len(bands):
            raise RuntimeError(f"Expected {len(bands)} bands, got {src.count}")
        scl_index = bands.index("SCL") + 1
        src.set_band_description(scl_index, "SCL")

    return output_path
