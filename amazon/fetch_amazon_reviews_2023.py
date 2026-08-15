#!/usr/bin/env python3

"""Download Amazon Reviews 2023 files directly from McAuley Lab."""

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
DEFAULT_TARGET_DIR = Path(__file__).resolve().parent.parent / "raw" / "amazon_reviews_2023"


def download(url: str, destination: Path, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Skip existing file: {destination}")
        return

    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        print(f"Downloading: {url}")
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    print(f"Saved: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="All_Beauty", help="Comma-separated categories")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    categories = [name.strip() for name in args.categories.split(",") if name.strip()]
    if not categories:
        raise ValueError("At least one category is required")
    invalid_categories = [
        name for name in categories if not name.replace("_", "").isalnum()
    ]
    if invalid_categories:
        raise ValueError(f"Invalid category names: {', '.join(invalid_categories)}")

    for category in categories:
        review_filename = f"{category}.jsonl.gz"
        download(
            f"{BASE_URL}/review_categories/{review_filename}",
            args.target_dir / review_filename,
            args.force,
        )
        if args.include_metadata:
            metadata_filename = f"meta_{category}.jsonl.gz"
            download(
                f"{BASE_URL}/meta_categories/{metadata_filename}",
                args.target_dir / metadata_filename,
                args.force,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
