"""Functional tests for Feature 17: Docker Deployment.

These tests verify that the web app can be packaged into a Docker container
and that the container serves the map correctly.
"""

import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
DOCKERFILE = WEB_DIR / "Dockerfile"
COMPOSE_FILE = WEB_DIR / "docker-compose.yml"


def _docker_available() -> bool:
    """Return True if the Docker CLI is available."""
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


DOCKER_AVAILABLE = _docker_available()


@pytest.fixture(scope="session")
def docker_image():
    """Build the Docker image once per session and clean up after."""
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker not available")

    result = subprocess.run(
        ["docker", "build", "-t", "goldmine-watch-test", str(WEB_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"docker build failed:\n{result.stderr}")

    yield "goldmine-watch-test"

    # Cleanup image after all tests
    subprocess.run(
        ["docker", "rmi", "-f", "goldmine-watch-test"],
        capture_output=True,
    )


@pytest.mark.integration
class TestFeature17Docker:
    """Docker deployment functional tests."""

    def test_dockerfile_exists(self):
        """Dockerfile must be present in the web directory."""
        assert DOCKERFILE.exists(), f"Dockerfile not found at {DOCKERFILE}"

    def test_dockerfile_content(self):
        """Dockerfile must use python:3.11-slim and expose port 8000."""
        content = DOCKERFILE.read_text()
        assert "FROM python:3.11-slim" in content
        assert "EXPOSE 8000" in content
        assert 'CMD ["python", "-m", "http.server", "8000"]' in content

    def test_dockerfile_has_healthcheck(self):
        """Dockerfile should include a HEALTHCHECK instruction."""
        content = DOCKERFILE.read_text()
        assert "HEALTHCHECK" in content

    def test_docker_compose_exists(self):
        """docker-compose.yml must be present in the web directory."""
        assert COMPOSE_FILE.exists(), f"docker-compose.yml not found at {COMPOSE_FILE}"

    def test_docker_compose_content(self):
        """docker-compose must define the goldmine-watch service correctly."""
        content = COMPOSE_FILE.read_text()
        assert "goldmine-watch:" in content
        assert "build: ." in content
        assert '"8000:8000"' in content
        assert "./data:/app/data:ro" in content
        assert "unless-stopped" in content

    def test_docker_build_succeeds(self, docker_image):
        """Docker build should complete without errors."""
        # The docker_image fixture handles the build and asserts success.
        assert docker_image == "goldmine-watch-test"

    def test_container_starts(self, docker_image):
        """Docker run should start and listen on port 8000."""
        unique = uuid.uuid4().hex[:8]
        container_name = f"goldmine-watch-test-{unique}"
        # Run container in detached mode
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "8000:8000",
                docker_image,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"docker run failed:\n{result.stderr}"
        container_id = result.stdout.strip()

        try:
            # Wait briefly for the server to start
            time.sleep(2)

            # Check container is running
            ps = subprocess.run(
                ["docker", "ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=True,
            )
            assert ps.stdout.strip() == container_id
        finally:
            # Cleanup
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
            )

    def test_map_accessible_in_container(self, docker_image):
        """Curl http://localhost:8000 should return 200."""
        unique = uuid.uuid4().hex[:8]
        container_name = f"goldmine-watch-test-http-{unique}"
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "8000:8000",
                docker_image,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"docker run failed:\n{result.stderr}"
        container_id = result.stdout.strip()

        try:
            time.sleep(2)
            response = urllib.request.urlopen("http://localhost:8000/", timeout=10)
            assert response.getcode() == 200
            body = response.read().decode("utf-8")
            assert "GoldMine Watch" in body
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
            )

    def test_data_volume_mounted(self, docker_image):
        """GeoJSON files should be accessible inside container."""
        unique = uuid.uuid4().hex[:8]
        container_name = f"goldmine-watch-test-vol-{unique}"
        # Create a temp data directory with a test geojson
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_geojson = tmp_path / "detections.geojson"
            test_geojson.write_text('{"type": "FeatureCollection", "features": []}')

            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-p",
                    "8001:8000",
                    "-v",
                    f"{tmpdir}:/app/data:ro",
                    docker_image,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"docker run failed:\n{result.stderr}"
            container_id = result.stdout.strip()

            try:
                time.sleep(2)
                # Verify the file is accessible via HTTP
                response = urllib.request.urlopen(
                    "http://localhost:8001/data/detections.geojson", timeout=10
                )
                assert response.getcode() == 200
                body = response.read().decode("utf-8")
                assert "FeatureCollection" in body
            finally:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                )
