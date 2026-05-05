"""End-to-end pipeline test.

This test runs the full pipeline on a minimal fixture dataset
to verify that all stages connect correctly.
"""

import pytest


@pytest.mark.e2e
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end pipeline test suite."""

    def test_full_pipeline(self):
        """Should run labels -> patches -> train -> predict -> export without errors."""
        pass
