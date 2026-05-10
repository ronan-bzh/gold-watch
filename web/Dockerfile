FROM python:3.11-slim

WORKDIR /app

# Install Python package (includes fastapi, uvicorn, rio-tiler)
COPY pyproject.toml ./
RUN pip install -e ".[web]"

# Copy source code, web assets, and SQL schema
COPY src/ ./src/
COPY web/ ./web/
COPY data/schema.sql ./data/schema.sql

# Ensure cache mount points exist for Docker Compose volumes
RUN mkdir -p /app/data/cache/tiles

# The SQLite DB will be created at runtime from schema.sql
ENV TILES_DB=/app/data/cache/db/tiles.db
ENV TILES_SCHEMA=/app/data/schema.sql

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
