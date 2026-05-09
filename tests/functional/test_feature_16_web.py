"""Functional tests for Feature 16: Web Map.

These tests verify the web map files, their content, and that a local HTTP server
can serve the map and GeoJSON data without errors.
"""

import http.client
import json
import shutil
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
DATA_DIR = PROJECT_ROOT / "data"


class TestFeature16WebMap:
    """End-to-end web map tests."""

    @pytest.fixture(scope="class")
    def web_data_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Copy web files into a temp directory and symlink data."""
        tmp_dir = tmp_path_factory.mktemp("web")
        web_tmp = tmp_dir / "web"
        shutil.copytree(WEB_DIR, web_tmp)

        data_tmp = web_tmp / "data"
        data_tmp.mkdir(exist_ok=True)

        labels_src = DATA_DIR / "french_guiana_mines.geojson"
        if labels_src.exists():
            (data_tmp / "labels.geojson").symlink_to(labels_src)

        detections_src = PROJECT_ROOT / "outputs" / "detections_square.geojson"
        if detections_src.exists():
            (data_tmp / "detections.geojson").symlink_to(detections_src)

        return web_tmp

    @pytest.fixture(scope="class")
    def server(self, web_data_dir: Path) -> int:
        """Start a local HTTP server and return its port."""
        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        web_dir = str(web_data_dir)

        class SilentHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                super().__init__(*args, directory=web_dir, **kwargs)

            def log_message(self, format: str, *args: object) -> None:
                pass

        httpd = TCPServer(("", port), SilentHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        # Wait until the server is accepting connections
        for _ in range(50):
            try:
                with socket.create_connection(("localhost", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)

        yield port

        httpd.shutdown()
        httpd.server_close()

    def test_html_exists(self) -> None:
        """index.html should exist in the web directory."""
        assert (WEB_DIR / "index.html").exists()

    def test_js_exists(self) -> None:
        """app.js should exist in the web directory."""
        assert (WEB_DIR / "app.js").exists()

    def test_css_exists(self) -> None:
        """style.css should exist in the web directory."""
        assert (WEB_DIR / "style.css").exists()

    def test_html_includes_leaflet(self) -> None:
        """HTML should include Leaflet CSS and JS from unpkg."""
        html = (WEB_DIR / "index.html").read_text()
        assert "leaflet@1.9.4" in html
        assert "leaflet.css" in html
        assert "leaflet.js" in html

    def test_html_has_map_div(self) -> None:
        """HTML should contain a map container div."""
        html = (WEB_DIR / "index.html").read_text()
        assert '<div id="map"></div>' in html

    def test_js_references_geojson_files(self) -> None:
        """app.js should reference the expected GeoJSON data paths."""
        js = (WEB_DIR / "app.js").read_text()
        assert "data/detections.geojson" in js
        assert "data/labels.geojson" in js

    def test_js_has_threshold_slider_logic(self) -> None:
        """app.js should implement threshold filtering."""
        js = (WEB_DIR / "app.js").read_text()
        assert "threshold" in js
        assert "renderDetections" in js

    def test_js_has_popup_logic(self) -> None:
        """app.js should bind popups to detection polygons."""
        js = (WEB_DIR / "app.js").read_text()
        assert "bindPopup" in js
        assert "Confidence" in js

    def test_js_has_toggle_logic(self) -> None:
        """app.js should support toggling labels and detections."""
        js = (WEB_DIR / "app.js").read_text()
        assert "updateVisibility" in js
        assert "show-labels" in js
        assert "show-detections" in js

    def test_map_loads_without_errors(self, web_data_dir: Path, server: int) -> None:
        """Open index.html via HTTP, check for 200 OK."""
        conn = http.client.HTTPConnection("localhost", server, timeout=5)
        try:
            conn.request("GET", "/")
            response = conn.getresponse()
            assert response.status == 200
            body = response.read().decode()
            assert "GoldMine Watch" in body
            assert "leaflet" in body.lower()
        finally:
            conn.close()

    def test_labels_geojson_served(self, web_data_dir: Path, server: int) -> None:
        """Labels GeoJSON should be accessible and valid JSON."""
        conn = http.client.HTTPConnection("localhost", server, timeout=5)
        try:
            conn.request("GET", "/data/labels.geojson")
            response = conn.getresponse()
            if response.status == 200:
                body = response.read().decode()
                data = json.loads(body)
                assert data["type"] == "FeatureCollection"
                assert len(data["features"]) > 0
            else:
                pytest.skip("Labels GeoJSON not available on this system")
        finally:
            conn.close()

    def test_detections_geojson_served(self, web_data_dir: Path, server: int) -> None:
        """Detections GeoJSON should be accessible if it exists."""
        conn = http.client.HTTPConnection("localhost", server, timeout=5)
        try:
            conn.request("GET", "/data/detections.geojson")
            response = conn.getresponse()
            if response.status == 200:
                body = response.read().decode()
                data = json.loads(body)
                assert data["type"] == "FeatureCollection"
            else:
                pytest.skip("Detections GeoJSON not yet generated (Feature 14 pending)")
        finally:
            conn.close()

    def test_js_sets_map_view_to_french_guiana(self) -> None:
        """Map should be centered on French Guiana."""
        js = (WEB_DIR / "app.js").read_text()
        assert ".setView([4.0, -53.0]" in js

    def test_css_styles_map_fullscreen(self) -> None:
        """CSS should make the map fill the viewport."""
        css = (WEB_DIR / "style.css").read_text()
        assert "#map" in css
        assert "position: absolute" in css or "position:absolute" in css
