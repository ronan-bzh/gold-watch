# Feature 17: Docker Deployment

**Goal:** Package the web app and all data into a Docker container for easy deployment.

**Prerequisite:** Feature 16 (Web Map) must be complete.

---

## What You Build

### Source Code

`web/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy web assets
COPY index.html style.css app.js /app/
COPY data/ /app/data/

# Expose port
EXPOSE 8000

# Serve static files
CMD ["python", "-m", "http.server", "8000"]
```

`web/docker-compose.yml`:

```yaml
version: '3.8'

services:
  goldmine-watch:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
    restart: unless-stopped
```

`Makefile` updates:

```makefile
docker-build:
	docker build -t goldmine-watch ./web

docker-run:
	docker run -p 8000:8000 goldmine-watch

docker-compose-up:
	docker-compose -f web/docker-compose.yml up -d
```

### Unit Tests

N/A (Docker testing is integration-level)

### Functional Tests

`tests/functional/test_feature_17_docker.py`:

```python
class TestFeature17Docker:
    def test_docker_build_succeeds(self):
        """docker build should complete without errors."""
    
    def test_container_starts(self):
        """docker run should start and listen on port 8000."""
    
    def test_map_accessible_in_container(self):
        """curl http://localhost:8000 should return 200."""
    
    def test_data_volume_mounted(self):
        """GeoJSON files should be accessible inside container."""
```

### Demo Script

```bash
# Build and run
docker build -t goldmine-watch ./web
docker run -p 8000:8000 -v $(pwd)/outputs:/app/data:ro goldmine-watch

# Or use docker-compose
cd web && docker-compose up -d

# Open browser
open http://localhost:8000
```

Output:
```
Docker Deployment
=================
Building image goldmine-watch...
Successfully built 1a2b3c4d

Starting container...
Container running on http://localhost:8000

Health check: OK (HTTP 200)
Data volume: mounted (2 files)
```

---

## Success Criteria

1. `docker build` completes without errors
2. `docker run -p 8000:8000` serves the map
3. `curl http://localhost:8000` returns HTTP 200
4. GeoJSON data is accessible inside container
5. Container restart policy works
6. Image size <500 MB

---

## What You Learn

- Docker containerization
- Multi-stage builds (if adding Python backend later)
- Volume mounts for data

---

## What You DON'T Build

- Kubernetes deployment
- CI/CD pipeline
- Load balancing

**Time estimate:** 1–2 hours

---

## Notes

- Keep the image small by using python:3.11-slim
- Mount data as read-only volume
- For production, consider nginx instead of python http.server
- Add HEALTHCHECK instruction to Dockerfile
- Real-data tests require all previous features completed.
