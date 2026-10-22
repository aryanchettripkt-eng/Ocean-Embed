"""Export or verify OpenAPI specification for OceanScope backend."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.main import app


def get_openapi_spec() -> dict[str, Any]:
    """Generate OpenAPI schema dictionary."""
    return app.openapi()


def main() -> None:
    """CLI entry point for export_openapi."""
    parser = argparse.ArgumentParser(description="Export or verify OpenAPI specification.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that openapi.json is up to date without overwriting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openapi.json"),
        help="Path to output openapi.json file.",
    )
    args = parser.parse_args()

    spec = get_openapi_spec()
    spec_json = json.dumps(spec, indent=2, sort_keys=True) + "\n"

    target_path: Path = args.output
    if args.check:
        if not target_path.exists():
            print(f"Error: {target_path} does not exist. Run without --check to generate.")
            sys.exit(1)
        current_content = target_path.read_text(encoding="utf-8")
        if current_content != spec_json:
            print(f"Error: {target_path} is out of date. Run without --check to update.")
            sys.exit(1)
        print(f"{target_path} is up to date.")
    else:
        target_path.write_text(spec_json, encoding="utf-8")
        print(f"Exported OpenAPI spec to {target_path}")


if __name__ == "__main__":
    main()
