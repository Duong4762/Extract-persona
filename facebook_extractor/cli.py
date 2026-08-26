"""Command-line interface for the Facebook persona pipeline."""

import argparse
from typing import Iterable

from .config import Config
from .pipeline import (
    compact_profiles,
    extract_personas,
    generate_persona_stats,
    ingest_posts,
    prepare_histories,
    select_users,
)

DESCRIPTION = "Build schema-constrained personas from Facebook profiles and content."


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=("all", "ingest", "prepare", "compact", "extract", "stats"),
        nargs="?",
        default="all",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config()
    if config.post_shards < 1:
        raise ValueError("post-shards must be at least 1")
    if config.min_history_days < 0:
        raise ValueError("min-history-days must be at least 0")

    print("Work directory:", config.work_dir.resolve())
    config.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "ingest"}:
        ingest_posts(config)
    if args.stage in {"all", "prepare"}:
        select_users(config)
        prepare_histories(config)
    if args.stage in {"all", "compact"}:
        compact_profiles(config)
    if args.stage in {"all", "extract"}:
        extract_personas(config)
    if args.stage in {"all", "stats"}:
        generate_persona_stats(config)
