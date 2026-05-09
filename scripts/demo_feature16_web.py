#!/usr/bin/env python3
"""Demo script for Feature 16: Web Map.

Prepares data symlinks and starts a local HTTP server for the web map.
Usage:
    python scripts/demo_feature16_web.py [--config configs/mvp.yaml] [--port 8000]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Prepare data symlinks and start the local HTTP server."""
    parser = argparse.ArgumentParser(description="Start the GoldMine Watch web map")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mvp.yaml"),
        help="Path to config YAML (default: configs/mvp.yaml)",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    web_dir = project_root / "web"
    data_dir = web_dir / "data"

    if not web_dir.exists():
        print(f"Error: web directory not found at {web_dir}")
        sys.exit(1)

    # Load config for paths (fallback to hardcoded defaults if config missing)
    if args.config.exists():
        cfg = load_config(args.config)
        labels_src = project_root / cfg.get("dataset", {}).get(
            "source_mines", "data/french_guiana_mines.geojson"
        )
        outputs_dir = project_root / cfg.get("paths", {}).get("outputs", "outputs")
    else:
        labels_src = project_root / "data" / "french_guiana_mines.geojson"
        outputs_dir = project_root / "outputs"
        print(f"Warning: config not found at {args.config}, using defaults.")

    data_dir.mkdir(exist_ok=True)

    # Symlink labels
    labels_dst = data_dir / "labels.geojson"
    if labels_src.exists() and not labels_dst.exists():
        labels_dst.symlink_to(labels_src)
        print(f"Linked labels: {labels_dst} -> {labels_src}")

    # Symlink detections if available
    detections_src = outputs_dir / "detections_square.geojson"
    detections_dst = data_dir / "detections.geojson"
    if detections_src.exists() and not detections_dst.exists():
        detections_dst.symlink_to(detections_src)
        print(f"Linked detections: {detections_dst} -> {detections_src}")
    elif not detections_src.exists():
        print(
            "Note: detections_square.geojson not found. "
            "Run Feature 14 to generate detections, or the map will show labels only."
        )

    print(f"\nStarting HTTP server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    os.chdir(web_dir)
    subprocess.run([sys.executable, "-m", "http.server", str(args.port)], check=True)


if __name__ == "__main__":
    main()
