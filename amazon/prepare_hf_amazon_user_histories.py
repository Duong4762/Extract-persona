#!/usr/bin/env python3
"""Prepare HF-exported Amazon histories for persona inference/evaluation.

The lightweight Hugging Face exporter writes one row per user with all retrieved
reviews in `reviews`. This script converts that raw shape into the richer
contract used by the Amazon persona inference and rating-holdout workflows:

- low-signal review filtering
- chronological per-user construction/validation split
- `temporal_split` metadata
- per-category review/rating/text summary stats for both splits
- aggregate filter summary JSON
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


FULFILLMENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(as expected|works as expected|just as described)\b",
        r"\b(fast shipping|quick shipping|arrived (on time|quickly|early))\b",
        r"\b(great product|good product|nice product|excellent product)\b",
        r"\b(would buy again|will buy again|recommend this product)\b",
        r"\b(five stars|four stars|three stars|two stars|one star)\b",
        r"^\s*(good|great|excellent|perfect|nice|ok|okay|love it|liked it)[.!]?\s*$",
    )
]


def iter_jsonl_or_gz(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_timestamp(value: Any) -> int | None:
    if value is None:
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp < 0:
        return None
    if timestamp < 10_000_000_000:
        timestamp *= 1000
    return timestamp


def timestamp_to_date(value: Any) -> str | None:
    timestamp = normalize_timestamp(value)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date().isoformat()


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def valid_rating(value: Any) -> bool:
    rating = float_or_none(value)
    return rating is not None and 1 <= rating <= 5


def review_text(review: dict[str, Any]) -> str:
    return compact_text(review.get("text") or review.get("review_text"))


def review_title(review: dict[str, Any]) -> str:
    return compact_text(review.get("title") or review.get("review_title"))


def product_title(review: dict[str, Any]) -> str:
    return compact_text(review.get("product_title"))


def product_main_category(review: dict[str, Any]) -> str:
    return compact_text(review.get("product_main_category"))


def product_category_path(review: dict[str, Any]) -> list[str]:
    value = review.get("product_category_path")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = [value]
    if not isinstance(value, list):
        return []
    path = []
    seen = set()
    for item in value:
        text = compact_text(item)
        if text and text not in seen:
            seen.add(text)
            path.append(text)
        if len(path) >= 6:
            break
    return path


def has_product_info(review: dict[str, Any]) -> bool:
    return bool(
        product_title(review)
        or product_main_category(review)
        or product_category_path(review)
    )


def has_textual_review(review: dict[str, Any]) -> bool:
    return bool(review_title(review) or review_text(review))


def fulfillment_or_template_review_match(review: dict[str, Any]) -> str | None:
    text = " ".join(part for part in (review_title(review), review_text(review)) if part)
    if not text:
        return None
    if len(text) > 220:
        return None
    for pattern in FULFILLMENT_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def review_key(review: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(review.get("category") or ""),
        str(review.get("parent_asin") or ""),
        str(review.get("asin") or ""),
        str(normalize_timestamp(review.get("timestamp")) or ""),
        review_title(review).lower(),
        review_text(review).lower(),
        product_title(review).lower(),
    )


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    row = dict(review)
    timestamp = normalize_timestamp(row.get("timestamp"))
    row["timestamp"] = timestamp
    row["date"] = row.get("date") or timestamp_to_date(timestamp)
    row["title"] = review_title(row)
    row["text"] = review_text(row)
    if product_category_path(row):
        row["product_category_path"] = product_category_path(row)
    if product_title(row):
        row["product_title"] = product_title(row)
    if product_main_category(row):
        row["product_main_category"] = product_main_category(row)
    return row


def filter_reviews(
    reviews: list[dict[str, Any]],
    *,
    min_review_text_chars: int,
    filter_fulfillment_reviews: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    seen = set()
    removed_by_reason: dict[str, int] = defaultdict(int)
    removed_by_category: dict[str, int] = defaultdict(int)

    for raw_review in reviews:
        if not isinstance(raw_review, dict):
            removed_by_reason["not_object"] += 1
            continue
        review = normalize_review(raw_review)
        category = str(review.get("category") or "Unknown")
        timestamp = normalize_timestamp(review.get("timestamp"))
        text = review_text(review)

        reason = None
        if timestamp is None:
            reason = "missing_or_invalid_timestamp"
        elif not valid_rating(review.get("rating")):
            reason = "missing_or_invalid_rating"
        elif (
            not has_product_info(review)
            and not review_title(review)
            and len(text) < min_review_text_chars
        ):
            reason = "insufficient_text_evidence"
        elif review_key(review) in seen:
            reason = "duplicate_review"
        elif filter_fulfillment_reviews and (
            pattern := fulfillment_or_template_review_match(review)
        ):
            reason = f"fulfillment_or_template:{pattern}"

        if reason:
            removed_by_reason[reason] += 1
            removed_by_category[category] += 1
            continue

        seen.add(review_key(review))
        kept.append(review)

    kept.sort(key=lambda row: (normalize_timestamp(row.get("timestamp")) or 0, str(row.get("asin") or "")))
    return kept, {
        "input_reviews": len(reviews),
        "kept_reviews": len(kept),
        "removed_reviews": len(reviews) - len(kept),
        "removed_by_reason": dict(sorted(removed_by_reason.items())),
        "removed_by_category": dict(sorted(removed_by_category.items())),
        "filter_fulfillment_reviews": filter_fulfillment_reviews,
        "min_review_text_chars": min_review_text_chars,
    }


def temporal_train_validation_split(
    reviews: list[dict[str, Any]],
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be in (0, 1)")
    split_index = int(len(reviews) * train_fraction)
    split_index = max(1, min(split_index, len(reviews) - 1))
    construction = reviews[:split_index]
    validation = reviews[split_index:]

    def count_text(rows: list[dict[str, Any]]) -> int:
        return sum(1 for row in rows if review_text(row))

    def count_ratings(rows: list[dict[str, Any]]) -> int:
        return sum(1 for row in rows if valid_rating(row.get("rating")))

    return construction, validation, {
        "method": "per_user_temporal",
        "unit": "review_or_rating_row",
        "train_fraction": train_fraction,
        "construction_row_count": len(construction),
        "validation_row_count": len(validation),
        "full_row_count": len(reviews),
        "construction_review_count": len(construction),
        "validation_review_count": len(validation),
        "full_review_count": len(reviews),
        "construction_text_review_count": count_text(construction),
        "validation_text_review_count": count_text(validation),
        "construction_rating_count": count_ratings(construction),
        "validation_rating_count": count_ratings(validation),
        "construction_first_timestamp": construction[0].get("timestamp") if construction else None,
        "construction_last_timestamp": construction[-1].get("timestamp") if construction else None,
        "validation_first_timestamp": validation[0].get("timestamp") if validation else None,
        "validation_last_timestamp": validation[-1].get("timestamp") if validation else None,
    }


def category_review_stats(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for review in reviews:
        category = str(review.get("category") or "Unknown")
        item = stats.setdefault(
            category,
            {
                "review_count": 0,
                "text_review_count": 0,
                "text_chars": 0,
                "verified_count": 0,
                "helpful_vote_sum": 0,
                "rating_count": 0,
                "rating_sum": 0.0,
                "rating_counts": {},
                "rating_only_count": 0,
                "rating_only_rating_counts": {},
                "product_title_counts": {},
                "product_main_category_counts": {},
                "product_category_counts": {},
                "rating_only_product_title_counts": {},
                "rating_only_product_main_category_counts": {},
                "rating_only_product_category_counts": {},
            },
        )
        item["review_count"] += 1
        text = review_text(review)
        is_rating_only = not has_textual_review(review)
        if text:
            item["text_review_count"] += 1
            item["text_chars"] += len(text)
        if is_rating_only:
            item["rating_only_count"] += 1
        title = product_title(review)
        if title:
            item["product_title_counts"][title] = item["product_title_counts"].get(title, 0) + 1
            if is_rating_only:
                item["rating_only_product_title_counts"][title] = item["rating_only_product_title_counts"].get(title, 0) + 1
        main_category = product_main_category(review)
        if main_category:
            item["product_main_category_counts"][main_category] = item["product_main_category_counts"].get(main_category, 0) + 1
            if is_rating_only:
                item["rating_only_product_main_category_counts"][main_category] = item["rating_only_product_main_category_counts"].get(main_category, 0) + 1
        for product_category in product_category_path(review):
            item["product_category_counts"][product_category] = item["product_category_counts"].get(product_category, 0) + 1
            if is_rating_only:
                item["rating_only_product_category_counts"][product_category] = item["rating_only_product_category_counts"].get(product_category, 0) + 1
        if review.get("verified_purchase") is True:
            item["verified_count"] += 1
        try:
            item["helpful_vote_sum"] += int(review.get("helpful_vote") or 0)
        except (TypeError, ValueError):
            pass
        rating = float_or_none(review.get("rating"))
        if rating is not None:
            key = str(int(rating)) if rating.is_integer() else str(rating)
            item["rating_count"] += 1
            item["rating_sum"] += rating
            item["rating_counts"][key] = item["rating_counts"].get(key, 0) + 1
            if is_rating_only:
                item["rating_only_rating_counts"][key] = item["rating_only_rating_counts"].get(key, 0) + 1

    def top_counts(counts: dict[str, int], limit: int = 25) -> dict[str, int]:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])

    for item in stats.values():
        count = item["review_count"]
        rating_count = item["rating_count"]
        item["verified_share"] = item["verified_count"] / count if count else 0.0
        item["mean_rating"] = item["rating_sum"] / rating_count if rating_count else None
        item["rating_counts"] = dict(sorted(item["rating_counts"].items()))
        item["rating_only_rating_counts"] = dict(sorted(item["rating_only_rating_counts"].items()))
        for key in (
            "product_title_counts",
            "product_main_category_counts",
            "product_category_counts",
            "rating_only_product_title_counts",
            "rating_only_product_main_category_counts",
            "rating_only_product_category_counts",
        ):
            item[key] = top_counts(item[key])
    return dict(sorted(stats.items()))


def merge_category_stats(target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]) -> None:
    for category, stats in source.items():
        out = target.setdefault(
            category,
            {
                "review_count": 0,
                "text_review_count": 0,
                "text_chars": 0,
                "verified_count": 0,
                "helpful_vote_sum": 0,
                "rating_count": 0,
                "rating_sum": 0.0,
                "rating_counts": {},
                "rating_only_count": 0,
                "rating_only_rating_counts": {},
                "product_title_counts": {},
                "product_main_category_counts": {},
                "product_category_counts": {},
                "rating_only_product_title_counts": {},
                "rating_only_product_main_category_counts": {},
                "rating_only_product_category_counts": {},
            },
        )
        for key in ("review_count", "text_review_count", "text_chars", "verified_count", "helpful_vote_sum", "rating_count", "rating_only_count"):
            out[key] += stats.get(key, 0) or 0
        out["rating_sum"] += stats.get("rating_sum", 0.0) or 0.0
        for rating, count in (stats.get("rating_counts") or {}).items():
            out["rating_counts"][rating] = out["rating_counts"].get(rating, 0) + count
        for rating, count in (stats.get("rating_only_rating_counts") or {}).items():
            out["rating_only_rating_counts"][rating] = out["rating_only_rating_counts"].get(rating, 0) + count
        for key in (
            "product_title_counts",
            "product_main_category_counts",
            "product_category_counts",
            "rating_only_product_title_counts",
            "rating_only_product_main_category_counts",
            "rating_only_product_category_counts",
        ):
            for value, count in (stats.get(key) or {}).items():
                out[key][value] = out[key].get(value, 0) + count


def finalize_aggregate_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized = {}

    def top_counts(counts: dict[str, int], limit: int = 25) -> dict[str, int]:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])

    for category, item in sorted(stats.items()):
        count = item["review_count"]
        rating_count = item["rating_count"]
        out = dict(item)
        out["verified_share"] = item["verified_count"] / count if count else 0.0
        out["mean_rating"] = item["rating_sum"] / rating_count if rating_count else None
        out["rating_counts"] = dict(sorted(item["rating_counts"].items()))
        out["rating_only_rating_counts"] = dict(sorted(item["rating_only_rating_counts"].items()))
        for key in (
            "product_title_counts",
            "product_main_category_counts",
            "product_category_counts",
            "rating_only_product_title_counts",
            "rating_only_product_main_category_counts",
            "rating_only_product_category_counts",
        ):
            out[key] = top_counts(item[key])
        finalized[category] = out
    return finalized


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-total-reviews", type=int, default=2)
    parser.add_argument("--min-construction-reviews", type=int, default=1)
    parser.add_argument("--min-validation-reviews", type=int, default=1)
    parser.add_argument("--min-review-text-chars", type=int, default=1)
    parser.add_argument("--keep-fulfillment-reviews", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_output or args.output.with_suffix(args.output.suffix + ".filter_summary.json")

    rows_read = 0
    rows_written = 0
    skipped_users: dict[str, str] = {}
    aggregate_filter = {
        "input_reviews": 0,
        "kept_reviews": 0,
        "removed_reviews": 0,
        "removed_by_reason": defaultdict(int),
        "removed_by_category": defaultdict(int),
    }
    aggregate_category_stats: dict[str, dict[str, Any]] = {}
    aggregate_validation_category_stats: dict[str, dict[str, Any]] = {}

    with args.output.open("w", encoding="utf-8") as out:
        for user_row in iter_jsonl_or_gz(args.input):
            rows_read += 1
            user_id = str(user_row.get("user_id") or "")
            raw_reviews = user_row.get("reviews") or []
            if not user_id or not isinstance(raw_reviews, list):
                skipped_users[user_id or f"row_{rows_read}"] = "missing_user_id_or_reviews"
                continue
            reviews, filter_summary = filter_reviews(
                raw_reviews,
                min_review_text_chars=args.min_review_text_chars,
                filter_fulfillment_reviews=not args.keep_fulfillment_reviews,
            )
            for key in ("input_reviews", "kept_reviews", "removed_reviews"):
                aggregate_filter[key] += filter_summary[key]
            for reason, count in filter_summary["removed_by_reason"].items():
                aggregate_filter["removed_by_reason"][reason] += count
            for category, count in filter_summary["removed_by_category"].items():
                aggregate_filter["removed_by_category"][category] += count

            if len(reviews) < args.min_total_reviews:
                skipped_users[user_id] = "too_few_reviews_after_filter"
                continue
            construction, validation, split = temporal_train_validation_split(
                reviews,
                args.train_fraction,
            )
            if len(construction) < args.min_construction_reviews:
                skipped_users[user_id] = "too_few_construction_reviews"
                continue
            if len(validation) < args.min_validation_reviews:
                skipped_users[user_id] = "too_few_validation_reviews"
                continue

            construction_stats = category_review_stats(construction)
            validation_stats = category_review_stats(validation)
            merge_category_stats(aggregate_category_stats, construction_stats)
            merge_category_stats(aggregate_validation_category_stats, validation_stats)
            prepared = {
                **{key: value for key, value in user_row.items() if key != "reviews"},
                "source": user_row.get("source", "amazon_reviews_2023"),
                "user_id": user_id,
                "review_count": len(construction),
                "retrieved_review_count": len(raw_reviews),
                "validation_review_count": len(validation),
                "categories": sorted({str(review.get("category") or "Unknown") for review in construction}),
                "validation_categories": sorted({str(review.get("category") or "Unknown") for review in validation}),
                "temporal_split": split,
                "category_review_stats": construction_stats,
                "validation_category_review_stats": validation_stats,
                "first_timestamp": construction[0].get("timestamp") if construction else None,
                "last_timestamp": construction[-1].get("timestamp") if construction else None,
                "validation_first_timestamp": validation[0].get("timestamp") if validation else None,
                "validation_last_timestamp": validation[-1].get("timestamp") if validation else None,
                "review_filter_summary": filter_summary,
                "reviews": construction,
                "validation_reviews": validation,
            }
            out.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            rows_written += 1

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "skipped_users": skipped_users,
        "filter": {
            **{key: aggregate_filter[key] for key in ("input_reviews", "kept_reviews", "removed_reviews")},
            "removed_by_reason": dict(sorted(aggregate_filter["removed_by_reason"].items())),
            "removed_by_category": dict(sorted(aggregate_filter["removed_by_category"].items())),
            "filter_fulfillment_reviews": not args.keep_fulfillment_reviews,
            "min_review_text_chars": args.min_review_text_chars,
        },
        "temporal_split": {
            "method": "per_user_temporal",
            "train_fraction": args.train_fraction,
        },
        "category_review_stats": finalize_aggregate_stats(aggregate_category_stats),
        "validation_category_review_stats": finalize_aggregate_stats(
            aggregate_validation_category_stats
        ),
    }
    write_json(summary_path, summary)
    print(
        f"Prepared {rows_written:,}/{rows_read:,} user histories: {args.output}; "
        f"summary: {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
