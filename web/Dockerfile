FROM python:3.11-slim

WORKDIR /app

# Copy web assets
COPY index.html style.css app.js /app/
COPY data/ /app/data/

# Expose port
EXPOSE 8000

# Health check using Python's built-in urllib
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Serve static files
CMD ["python", "-m", "http.server", "8000"]
