"""Command-line interface for the Facebook persona pipeline."""

import argparse
from typing import Iterable

from .config import FRESH_BATCH_NAME_START, NUM_FRESH_BATCHES, Config
from .pipeline import (
    analyze_user_thresholds,
    benchmark_llm_concurrency,
    compact_profiles,
    extract_personas,
    generate_persona_stats,
    find_user_history,
    ingest_posts,
    migrate_personas,
    prepare_histories,
    select_users,
)

DESCRIPTION = "Build schema-constrained personas from Facebook profiles and content."


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "stage",
        choices=(
            "all", "ingest", "prepare", "compact", "extract", "stats", "find",
            "bench-llm", "migrate", "analyze-users",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("user_id", nargs="?", help="Facebook user ID for the find stage")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config()
    if args.stage == "find":
        if not args.user_id:
            raise SystemExit("The find stage requires a user_id")
        path = find_user_history(config, args.user_id)
        if path is None:
            raise SystemExit(f"User {args.user_id} was not found in history shards")
        print(f"User {args.user_id} history shard: {path.resolve()}")
        return
    if args.user_id:
        raise SystemExit("user_id is only valid with the find stage")
    fresh_batch_max = FRESH_BATCH_NAME_START + NUM_FRESH_BATCHES - 1
    if not 1 <= config.fresh_batch <= fresh_batch_max:
        raise ValueError(
            f"FRESH_BATCH must be between 1 and {fresh_batch_max}"
        )
    if config.post_shards < 1:
        raise ValueError("post-shards must be at least 1")
    if config.min_history_days < 0:
        raise ValueError("min-history-days must be at least 0")
    if config.llm_workers < 1:
        raise ValueError("llm-workers must be at least 1")
    if config.preprocess_workers < 1:
        raise ValueError("preprocess-workers must be at least 1")
    if config.max_contents < 1:
        raise ValueError("max-contents must be at least 1")
    if not 0 <= config.timeline_content_ratio <= 1:
        raise ValueError("timeline-content-ratio must be between 0 and 1")
    if min(
        config.max_content_score_at,
        config.max_character_score_at,
        config.max_history_score_at,
    ) <= 0:
        raise ValueError("user score caps must be greater than 0")

    print("Work directory:", config.work_dir.resolve(), f"(FRESH_BATCH={config.fresh_batch})")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "ingest"}:
        print(
            f"Ingest reads {tuple(d.parent.name for d in config.content_dirs)} and writes "
            f"all {NUM_FRESH_BATCHES} output batches "
            f"({config.fresh_batch_dir(FRESH_BATCH_NAME_START).name}.."
            f"{config.fresh_batch_dir(FRESH_BATCH_NAME_START + NUM_FRESH_BATCHES - 1).name}); "
            "FRESH_BATCH only selects which one prepare/compact/extract/stats use below."
        )
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
    if args.stage == "bench-llm":
        benchmark_llm_concurrency(config)
    if args.stage == "migrate":
        migrate_personas(config)
    if args.stage == "analyze-users":
        analyze_user_thresholds(config)
