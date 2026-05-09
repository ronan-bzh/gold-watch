#!/usr/bin/env python3
r"""Demo script for Feature 11: Multi-Scene Training Dataset.

Usage::

    export $(cat .env | xargs)
    python scripts/demo_feature11_dataset.py \
        --mines data/french_guiana_mines.geojson \
        --output data/splits \
        --background 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts.build_training_dataset import main

if __name__ == "__main__":
    sys.exit(main())
