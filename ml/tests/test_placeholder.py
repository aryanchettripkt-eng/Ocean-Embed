"""Placeholder test for oceanscope_ml to establish CI/CD verification."""

import oceanscope_ml


def test_package_metadata() -> None:
    """Verify package import and version presence."""
    assert oceanscope_ml.__version__ == "0.1.0"
