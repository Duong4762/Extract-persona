#!/usr/bin/env python3
"""Production Amazon-reviewer persona extraction — sharded, resumable, 1 GPU.

One array task = one user_bucket (hex 00..ff) = one GPU. For its bucket it:
  1. loads the selection index (data/amazon/selected_users_100k.parquet),
  2. downloads that bucket's raw reviews from the gated HF dataset,
  3. filters low-signal reviews and makes a per-user temporal construction /
     validation split,
  4. enriches retained rows with compact product metadata and category summary
     stats,
  5. assembles each selected user's construction reviews into profile_text,
  6. runs the Amazon persona prompt over all category dimension-chunks, and
  7. appends one JSON object per user to data/amazon/extraction_v1/shard_<bkt>.jsonl.

Persona = one user. Resumable: skips user_id already written, so a preempted /
re-queued task continues where it left off. Output schema matches the wiki
extractor (fields:[{field_id,value,confidence,evidence,description,assignment_type}]).

A100 80GB note: the 35B MoE is ~70 GB in bf16 and will not leave room for the KV
cache on a single 80 GB card, so this script defaults to --quantization fp8
(weight-only FP8 via Marlin on Ampere → ~35 GB weights, plenty of KV headroom).

Example (single card):
  python run_extraction_amazon.py --shard-id 0 --quantization fp8 \
      --out-dir data/amazon/extraction_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("HF_HUB_CACHE", f"{CACHE}/hub")
os.environ.setdefault("HF_XET_CACHE", f"{CACHE}/xet")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd  # noqa: E402

REPO_ROOT = Path(os.environ.get("MATRIX_REPO_ROOT", ".")).resolve()
DATA_DIR = REPO_ROOT / "persona/human_extraction/data"
SELECTION = DATA_DIR / "amazon/selected_users_100k.parquet"
DIMENSIONS_JSON = REPO_ROOT / "persona/schema/dimensions.json"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
OPENROUTER_MODEL_ID = "qwen/qwen3.6-35b-a3b"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ASSIGNMENT_TYPES = {
    "direct",
    "structured_claim",
    "summary_inference",
    "unsupported",
}
NULLISH_VALUES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "unsupported",
    "not applicable",
}

DATASET_REPO = "MatrAIx2026/MatrAIx2026"
UBUK = ("amazon/modal_artifacts/"
        "amazon_reviews_2018_2023_user_buckets_min30_verified70_text2000")
METADATA_PREFIX = (
    "amazon/modal_artifacts/"
    "amazon_reviews_2023_metadata_by_parent_asin_bucket_v2"
)

REVIEW_TMPL = ("[{date}] {category} | {parent_asin} | rating={rating:.0f}/5 | "
               "verified={verified}\nTitle: {title}\n{text}")
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


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_matraix")
    if tok:
        return tok
    bashrc = Path(os.path.expanduser("~/.bashrc"))
    if bashrc.exists():
        for line in bashrc.read_text().splitlines():
            m = re.search(r"HF_TOKEN_matraix=['\"]?([^'\"\s]+)", line)
            if m:
                return m.group(1)
    return None


def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 15].rstrip() + " ... [truncated]"
    return text


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def has_textual_review(review: dict[str, Any]) -> bool:
    return bool(review_title(review) or review_text(review))


def has_product_info(review: dict[str, Any]) -> bool:
    return bool(product_title(review) or product_main_category(review) or product_category_path(review))


def fulfillment_or_template_review_match(review: dict[str, Any]) -> str | None:
    text = " ".join(part for part in (review_title(review), review_text(review)) if part)
    if not text or len(text) > 220:
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


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def normalize_review_record(review: dict[str, Any]) -> dict[str, Any]:
    row = {key: jsonable(value) for key, value in dict(review).items()}
    timestamp = normalize_timestamp(row.get("timestamp"))
    row["timestamp"] = timestamp
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
        review = normalize_review_record(raw_review)
        category = str(review.get("category") or "Unknown")
        text = review_text(review)
        reason = None
        if normalize_timestamp(review.get("timestamp")) is None:
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
    return construction, validation, {
        "method": "per_user_temporal",
        "unit": "review_or_rating_row",
        "train_fraction": train_fraction,
        "construction_row_count": len(construction),
        "validation_row_count": len(validation),
        "construction_text_review_count": sum(1 for row in construction if review_text(row)),
        "validation_text_review_count": sum(1 for row in validation if review_text(row)),
        "construction_rating_count": sum(1 for row in construction if valid_rating(row.get("rating"))),
        "validation_rating_count": sum(1 for row in validation if valid_rating(row.get("rating"))),
        "construction_last_timestamp": construction[-1].get("timestamp") if construction else None,
        "validation_first_timestamp": validation[0].get("timestamp") if validation else None,
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


def format_counts(counts: Any, *, limit: int = 6) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    items = sorted(counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[:limit]
    return ", ".join(f"{key}={value}" for key, value in items)


def render_summary_stats(row: dict[str, Any], *, max_chars: int = 4000) -> str:
    stats = row.get("category_review_stats")
    if not isinstance(stats, dict) or not stats:
        return ""
    split = row.get("temporal_split") if isinstance(row.get("temporal_split"), dict) else {}
    lines = ["=== Summary Stats ==="]
    lines.append(
        "construction_rows: "
        f"{split.get('construction_row_count', row.get('review_count', 'unknown'))}; "
        "validation_rows: "
        f"{split.get('validation_row_count', row.get('validation_review_count', 'unknown'))}; "
        "construction_text_reviews: "
        f"{split.get('construction_text_review_count', 'unknown')}; "
        "construction_ratings: "
        f"{split.get('construction_rating_count', 'unknown')}"
    )
    sorted_stats = sorted(
        stats.items(),
        key=lambda pair: (-(pair[1].get("review_count", 0) or 0), str(pair[0])),
    )
    for category, item in sorted_stats[:12]:
        mean_rating = item.get("mean_rating")
        mean_rating_text = f"{mean_rating:.2f}" if isinstance(mean_rating, int | float) else "unknown"
        parts = [
            f"category={category}",
            f"rows={item.get('review_count', 0)}",
            f"text_reviews={item.get('text_review_count', 0)}",
            f"rating_only={item.get('rating_only_count', 0)}",
            f"mean_rating={mean_rating_text}",
        ]
        for label, key, limit in (
            ("ratings", "rating_counts", 5),
            ("product_categories", "product_category_counts", 4),
            ("rating_only_products", "rating_only_product_title_counts", 3),
        ):
            rendered = format_counts(item.get(key), limit=limit)
            if rendered:
                parts.append(f"{label}={rendered}")
        lines.append("; ".join(parts))
    return compact_text("\n".join(lines), max_chars)


def render_review(review: dict[str, Any], *, index: int, max_review_text_chars: int) -> str:
    lines = [
        f"[review {index}]",
        f"date: {review.get('date') or review.get('timestamp') or 'unknown'}",
        f"category: {review.get('category') or 'Unknown'}",
        f"parent_asin: {review.get('parent_asin') or review.get('asin') or ''}",
        f"rating: {review.get('rating', 'unknown')}",
        f"review_title: {review_title(review) or '(untitled)'}",
    ]
    if product_title(review):
        lines.append(f"product_title: {compact_text(product_title(review), 220)}")
    if product_main_category(review):
        lines.append(f"product_main_category: {compact_text(product_main_category(review), 120)}")
    if product_category_path(review):
        lines.append("product_category_path: " + " > ".join(compact_text(item, 80) for item in product_category_path(review)))
    lines.extend(
        [
            f"verified: {review.get('verified_purchase', 'unknown')}",
            f"helpful_vote: {review.get('helpful_vote', 0)}",
            f"text: {compact_text(review_text(review), max_review_text_chars)}",
        ]
    )
    return "\n".join(lines)


def assemble_profile(row: dict[str, Any], max_chars: int, max_review_text_chars: int) -> str:
    reviews = row.get("reviews") or []
    categories = {review.get("category") or "Unknown" for review in reviews}
    header = (
        f"Amazon reviewer profile - construction split: {len(reviews)} reviews "
        f"across {len(categories)} categories."
    )
    parts = [header]
    summary = render_summary_stats(row)
    if summary:
        parts.append(summary)
    for index, review in enumerate(reviews, start=1):
        parts.append(render_review(review, index=index, max_review_text_chars=max_review_text_chars))
    return "\n\n".join(parts)[:max_chars]


def build_amazon_prompt(profile_text: str, dimensions: list[dict]) -> str:
    """Amazon-reviewer persona-extraction prompt (see extract_personas_amazon.ipynb)."""
    lines = [
        "You are mapping observable Amazon review evidence to schema-constrained "
        "persona fields for one reviewer. Fill attributes that are well supported "
        "by the review history, and leave unsupported or identity-like claims null.",
        "",
        "Important: emitting one field object is bookkeeping, not permission to "
        "fill the attribute. For every dimension, start from value=null and "
        'assignment_type="unsupported". Change value only when the evidence '
        "passes the rules below.",
        "",
        "Return ONLY JSON with this shape (no markdown, no commentary):",
        '{"fields": [{"field_id": "<one id from DIMENSIONS below>", '
        '"value": "<one allowed value, copied verbatim, or null>", '
        '"confidence": 0.0, '
        '"evidence": "<one short exact quote copied from REVIEWER HISTORY, or empty string>", '
        '"description": "<1-2 concrete sentences, or empty string>", '
        '"assignment_type": "direct|structured_claim|summary_inference|unsupported"}]}',
        "",
        "Allowed support:",
        "- direct: use when the reviewer explicitly states the fact about "
        "themselves in review text.",
        "- structured_claim: use for repeated owned/use-context statements or "
        "concrete non-sensitive purchase/review facts supported by at least 2 "
        "distinct reviews, products, or category clusters.",
        "- summary_inference: use for non-sensitive interests, shopping behavior, "
        "preferences, review style, communication style, or expertise when a "
        "repeated pattern is visible across the review history.",
        "- Overall writing style may support communication/cognitive-style "
        "dimensions only when the pattern is visible across at least 5 reviews.",
        "- unsupported: use when evidence is absent, one-off, ambiguous, generic, "
        "gift-related, or mainly about someone other than the reviewer.",
        "",
        "Hard limits:",
        "- For age, gender, health, disability, ethnicity, religion, politics, "
        "income, family/household status, occupation, location, employment, and "
        "parenthood: assign a non-null value only from an explicit self-statement. "
        "Do not use product category alone.",
        "- Do not attribute traits of gift recipients or other product users to "
        "the reviewer. A gift may support shopping behavior, not the reviewer's "
        "own identity, household, or hobbies.",
        "- Generic praise like \"great product\" or product titles alone is not "
        "diagnostic evidence for persona attributes.",
        "- Do not infer personality inventories, values, worldview, MBTI, Big "
        "Five, HEXACO, clinical attributes, or mental-state attributes from "
        "ordinary shopping reviews unless the reviewer explicitly states the "
        "trait or belief.",
        "",
        "Output rules:",
        "- Emit exactly one object per dimension listed below.",
        "- Do not output any field_id that is not listed in DIMENSIONS.",
        "- Do not duplicate field_id. Each listed field_id appears exactly once.",
        "- Do not omit assignment_type. Every object must include one of the four "
        "assignment_type strings above.",
        "- value MUST be exactly one of that dimension's allowed values (copied "
        "verbatim), OR null.",
        '- Never use "Unsupported", "unsupported", "Not applicable", "N/A", '
        '"unknown", or "" as value unless that exact string appears in that '
        "field's allowed values.",
        "- Judge the history as a whole; prefer attributes backed by MULTIPLE "
        "reviews over a single purchase (one-off items may be gifts for others).",
        "- If the reviews do not support a dimension, set value to null, "
        'confidence to 0.0, evidence to "", assignment_type to "unsupported", '
        'and description to "".',
        "- Every non-null value MUST include a short evidence quote copied "
        "verbatim from one of the reviews.",
        "- Evidence must be an exact quote from REVIEWER HISTORY, not your reasoning, "
        "a paraphrase, or a summary. If you cannot copy an exact quote, return "
        "unsupported.",
        "- If you cannot copy an exact quote, return unsupported.",
        "- Do not append support counts, explanations, or labels to evidence. "
        "Evidence must be only text that appears in REVIEWER HISTORY.",
        "- description: 1-2 concrete sentences describing THIS shopper for this "
        "attribute using details from their reviews (categories, products, "
        "statements). Describe the person; do not justify the label.",
        "- Sensitive / high-risk fields require explicit self-statements: age, "
        "gender, income, marital status, children count, religion, politics, "
        "ethnicity, health, disability, mental health, neurotype, MBTI, Big Five, "
        "personality traits, attachment style, and relationship style.",
        "- Do not infer these fields from product category, product size, possible "
        "gift purchases, cooking tools, romance books, writing style, tone, "
        "vocabulary, price level, or household items.",
        "- Return valid JSON only, with no markdown.",
        "- Most dimensions can be unsupported. Do not make the persona complete.",
        "",
        "DIMENSIONS (field_id — label — description — allowed values):",
    ]
    for d in dimensions:
        allowed = " | ".join(str(v) for v in d.get("values", [])) or "(free value)"
        desc = str(d.get("description", "")).strip()
        lines.append(f"- {d['id']} — {d.get('label', d['id'])} — {desc} — [{allowed}]")
    lines += ["", "REVIEWER HISTORY:", profile_text]
    return "\n".join(lines)


def parse_fields(text: str) -> list[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    fields = obj.get("fields")
    return fields if isinstance(fields, list) else []


def _unsupported(dim: dict) -> dict:
    return {
        "field_id": str(dim["id"]),
        "value": None,
        "confidence": 0.0,
        "evidence": "",
        "description": "",
        "assignment_type": "unsupported",
    }


def _confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _normalized_key(value: str) -> str:
    return " ".join(value.replace("-", "–").split()).casefold()


def _coerce_value(value, dim: dict) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in NULLISH_VALUES:
        return None
    allowed = [str(item) for item in dim.get("values", [])]
    if not allowed:
        return text
    if text in allowed:
        return text
    allowed_by_key = {_normalized_key(item): item for item in allowed}
    return allowed_by_key.get(_normalized_key(text))


def _quote_is_in_profile(evidence: str, profile_text: str) -> bool:
    if not evidence:
        return False
    if evidence in profile_text:
        return True
    return " ".join(evidence.split()) in " ".join(profile_text.split())


def sanitize_fields(
    fields: list[dict],
    dimensions: list[dict],
    profile_text: str = "",
) -> list[dict]:
    """Clamp Amazon model output to one schema-conformant field per dimension."""
    dim_by_id = {str(dim["id"]): dim for dim in dimensions}
    best_by_id: dict[str, dict] = {}

    for raw in fields:
        if not isinstance(raw, dict):
            continue
        field_id = str(raw.get("field_id") or "").strip()
        dim = dim_by_id.get(field_id)
        if dim is None:
            continue

        assignment_type = str(raw.get("assignment_type") or "").strip()
        value = _coerce_value(raw.get("value"), dim)
        confidence = _confidence(raw.get("confidence"))
        evidence = str(raw.get("evidence") or "").strip()
        description = str(raw.get("description") or "").strip()
        supported = (
            value is not None
            and assignment_type in ASSIGNMENT_TYPES
            and assignment_type != "unsupported"
            and _quote_is_in_profile(evidence, profile_text)
        )

        if supported:
            clean = {
                "field_id": field_id,
                "value": value,
                "confidence": confidence,
                "evidence": evidence,
                "description": description,
                "assignment_type": assignment_type,
            }
        else:
            clean = _unsupported(dim)

        prior = best_by_id.get(field_id)
        if prior is None:
            best_by_id[field_id] = clean
            continue
        prior_supported = prior.get("value") is not None
        clean_supported = clean.get("value") is not None
        if clean_supported and not prior_supported:
            best_by_id[field_id] = clean
        elif clean_supported == prior_supported and _confidence(
            clean.get("confidence")
        ) > _confidence(prior.get("confidence")):
            best_by_id[field_id] = clean

    return [best_by_id.get(str(dim["id"])) or _unsupported(dim) for dim in dimensions]


def cat_chunks(by_category: dict, per_chunk: int):
    out = []
    for cat_dims in by_category.values():
        for i in range(0, len(cat_dims), per_chunk):
            out.append(cat_dims[i : i + per_chunk])
    return out


def parent_asin_bucket(parent_asin: str) -> str:
    return hashlib.sha1(parent_asin.encode("utf-8")).hexdigest()[:2]


def parse_category_path(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
    values: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, list):
                values.extend(str(part) for part in item if part)
            elif item:
                values.append(str(item))
    elif parsed:
        values.append(str(parsed))
    deduped = []
    seen = set()
    for item in values:
        text = compact_text(item)
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
        if len(deduped) >= 6:
            break
    return deduped


def compact_product_info(row: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    title = compact_text(row.get("title"))
    if title:
        info["product_title"] = title
    main_category = compact_text(row.get("main_category"))
    if main_category:
        info["product_main_category"] = main_category
    category_path = parse_category_path(row.get("categories_json"))
    if category_path:
        info["product_category_path"] = category_path
    return info


def parent_asins_by_category(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    requested: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field in ("reviews", "validation_reviews"):
            for review in row.get(field) or []:
                parent_asin = review.get("parent_asin")
                category = review.get("category")
                if parent_asin and category:
                    requested[str(category)].add(str(parent_asin))
    return requested


def list_relevant_metadata_shards(
    repo_files: list[str],
    metadata_prefix: str,
    requested: dict[str, set[str]],
) -> dict[tuple[str, str], list[str]]:
    prefix = metadata_prefix.rstrip("/") + "/"
    wanted_keys = {
        (parent_asin_bucket(parent_asin), category)
        for category, parent_asins in requested.items()
        for parent_asin in parent_asins
    }
    shards: dict[tuple[str, str], list[str]] = defaultdict(list)
    for filename in repo_files:
        if not filename.startswith(prefix) or not filename.endswith(".parquet"):
            continue
        parts = filename[len(prefix) :].split("/")
        if len(parts) != 3:
            continue
        bucket_part, category_part, _name = parts
        if not bucket_part.startswith("bucket=") or not category_part.startswith("source_category="):
            continue
        bucket = bucket_part.split("=", 1)[1]
        category = category_part.split("=", 1)[1]
        key = (bucket, category)
        if key in wanted_keys:
            shards[key].append(filename)
    return {key: sorted(value) for key, value in shards.items()}


def attach_product_metadata(
    rows: list[dict[str, Any]],
    *,
    repo_files: list[str],
    repo_id: str,
    metadata_prefix: str,
    token: str | bool | None,
    download_delay_seconds: float,
) -> dict[str, int]:
    from huggingface_hub import hf_hub_download

    requested = parent_asins_by_category(rows)
    requested_pairs = sum(len(parent_asins) for parent_asins in requested.values())
    if not requested_pairs:
        return {
            "requested_category_parent_asin_pairs": 0,
            "matched_category_parent_asin_pairs": 0,
            "matched_review_rows": 0,
        }

    metadata_shards = list_relevant_metadata_shards(repo_files, metadata_prefix, requested)
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    columns = ["parent_asin", "source_category", "main_category", "title", "categories_json"]
    for (bucket, category), filenames in sorted(metadata_shards.items()):
        wanted_parent_asins = requested.get(category, set())
        for filename in filenames:
            if download_delay_seconds > 0:
                time.sleep(download_delay_seconds)
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                token=token,
            )
            for row in pd.read_parquet(local_path, columns=columns).to_dict(orient="records"):
                parent_asin = str(row.get("parent_asin") or "")
                if parent_asin in wanted_parent_asins:
                    metadata[(parent_asin, category)] = compact_product_info(row)

    matched_reviews = 0
    for row in rows:
        for field in ("reviews", "validation_reviews"):
            for review in row.get(field) or []:
                parent_asin = str(review.get("parent_asin") or "")
                category = str(review.get("category") or "")
                info = metadata.get((parent_asin, category))
                if info:
                    review.update(info)
                    matched_reviews += 1
        row["category_review_stats"] = category_review_stats(row.get("reviews") or [])
        row["validation_category_review_stats"] = category_review_stats(
            row.get("validation_reviews") or []
        )

    return {
        "requested_category_parent_asin_pairs": requested_pairs,
        "matched_category_parent_asin_pairs": len(metadata),
        "matched_review_rows": matched_reviews,
    }


def load_bucket_reviews(bucket: str, token: str | None) -> tuple[pd.DataFrame, list[str]]:
    """All reviews in one user_bucket (across every category file)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    repo_files = list(api.list_repo_files(DATASET_REPO, repo_type="dataset"))
    files = [f for f in repo_files if f.startswith(f"{UBUK}/bucket={bucket}/") and f.endswith(".parquet")]
    dfs = [pd.read_parquet(hf_hub_download(DATASET_REPO, f, repo_type="dataset",
                                           token=token)) for f in files]
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), repo_files


def openrouter_chat(
    conversations: list[list[dict]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = 6,
) -> list[str]:
    """Run prompts through OpenRouter's OpenAI-compatible chat endpoint."""
    texts: list[str] = []
    for conv in conversations:
        payload = {
            "model": model,
            "messages": conv,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "MatrAIx Amazon persona extraction",
            },
            method="POST",
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                texts.append(data["choices"][0]["message"]["content"])
                break
            except urllib.error.HTTPError as err:
                err_body = err.read().decode("utf-8", errors="replace")
                retryable = err.code in {408, 429, 500, 502, 503, 504}
                if retryable and attempt < retries - 1:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(
                    f"OpenRouter API error {err.code}: {err_body[:1000]}"
                ) from err
            except (urllib.error.URLError, KeyError, IndexError, TypeError) as err:
                if attempt < retries - 1:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(f"OpenRouter request failed: {err}") from err
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-id", type=int, required=True,
                    help="0..255 -> user_bucket hex 00..ff")
    ap.add_argument("--out-dir", default=str(DATA_DIR / "amazon/extraction_v1"))
    ap.add_argument("--batch-profiles", type=int, default=32,
                    help="profiles per vLLM submit / checkpoint granularity")
    ap.add_argument("--max-dims-per-chunk", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-profile-chars", type=int, default=48000)
    ap.add_argument("--max-review-text-chars", type=int, default=2000)
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--min-total-reviews", type=int, default=2)
    ap.add_argument("--min-construction-reviews", type=int, default=1)
    ap.add_argument("--min-validation-reviews", type=int, default=1)
    ap.add_argument("--min-review-text-chars", type=int, default=20)
    ap.add_argument("--keep-fulfillment-reviews", action="store_true")
    ap.add_argument("--no-product-info", dest="include_product_info", action="store_false")
    ap.set_defaults(include_product_info=True)
    ap.add_argument("--metadata-artifact-prefix", default=METADATA_PREFIX)
    ap.add_argument("--metadata-download-delay-seconds", type=float, default=0.05)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="GPUs per task (2 => bf16 fits across 2x A100 80GB, no quant)")
    ap.add_argument("--quantization", default="fp8",
                    help="fp8 (fits single A100 80GB) | none (bf16, needs 2x A100)")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap users this shard")
    ap.add_argument("--backend", choices=("vllm", "openrouter"), default="vllm")
    ap.add_argument("--openrouter-model", default=OPENROUTER_MODEL_ID)
    ap.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--openrouter-base-url", default=OPENROUTER_CHAT_URL)
    args = ap.parse_args()

    bucket = f"{args.shard_id:02x}"
    token = hf_token()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{bucket}.jsonl"

    # --- schema / chunks ---
    schema_doc = json.load(open(DIMENSIONS_JSON))
    by_category: dict[str, list] = {}
    for d in schema_doc["dimensions"]:
        by_category.setdefault(d.get("category", "Uncategorized"), []).append(d)
    chunk_list = cat_chunks(by_category, args.max_dims_per_chunk)

    # --- selection for this bucket ---
    sel = pd.read_parquet(SELECTION)
    sel_b = sel[sel.user_bucket == bucket]
    want = set(sel_b.user_id)
    review_count = dict(zip(sel_b.user_id, sel_b.review_count))
    if args.limit:
        want = set(list(want)[: args.limit])

    # --- resume: skip already-written user_id ---
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["user_id"])
                except Exception:
                    pass
    todo_ids = [u for u in want if u not in done]

    print(f"[shard {args.shard_id} bucket={bucket}] selected={len(sel_b):,} "
          f"want={len(want):,} done={len(done):,} todo={len(todo_ids):,} "
          f"chunks/user={len(chunk_list)}", flush=True)
    if not todo_ids:
        print("[shard] nothing to do — complete.", flush=True)
        return

    # --- load this bucket's reviews, prepare construction split, and assemble profiles ---
    t0 = time.time()
    rev, repo_files = load_bucket_reviews(bucket, token)
    rev = rev[rev.user_id.isin(set(todo_ids))]
    prepared_rows: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for uid, group in rev.groupby("user_id", sort=False):
        raw_reviews = group.to_dict(orient="records")
        reviews, filter_summary = filter_reviews(
            raw_reviews,
            min_review_text_chars=args.min_review_text_chars,
            filter_fulfillment_reviews=not args.keep_fulfillment_reviews,
        )
        if len(reviews) < args.min_total_reviews:
            skipped[str(uid)] = "too_few_reviews_after_filter"
            continue
        construction, validation, split_summary = temporal_train_validation_split(
            reviews, args.train_fraction
        )
        if len(construction) < args.min_construction_reviews:
            skipped[str(uid)] = "too_few_construction_reviews"
            continue
        if len(validation) < args.min_validation_reviews:
            skipped[str(uid)] = "too_few_validation_reviews"
            continue
        prepared_rows[str(uid)] = {
            "user_id": str(uid),
            "review_count": len(reviews),
            "validation_review_count": len(validation),
            "temporal_split": split_summary,
            "review_filter_summary": filter_summary,
            "reviews": construction,
            "validation_reviews": validation,
        }

    if args.include_product_info and prepared_rows:
        product_summary = attach_product_metadata(
            list(prepared_rows.values()),
            repo_files=repo_files,
            repo_id=DATASET_REPO,
            metadata_prefix=args.metadata_artifact_prefix,
            token=token,
            download_delay_seconds=args.metadata_download_delay_seconds,
        )
        print(f"[shard] product metadata summary: {product_summary}", flush=True)
    else:
        for row in prepared_rows.values():
            row["category_review_stats"] = category_review_stats(row.get("reviews") or [])
            row["validation_category_review_stats"] = category_review_stats(
                row.get("validation_reviews") or []
            )

    profiles = {
        uid: assemble_profile(row, args.max_profile_chars, args.max_review_text_chars)
        for uid, row in prepared_rows.items()
    }
    todo = [u for u in todo_ids if u in profiles]
    print(f"[shard] loaded {len(rev):,} reviews, assembled {len(profiles):,} "
          f"profiles, skipped={len(skipped):,} in {time.time()-t0:.0f}s", flush=True)
    if skipped:
        print(f"[shard] skipped reasons: {dict(sorted((reason, list(skipped.values()).count(reason)) for reason in set(skipped.values())))}", flush=True)

    # --- load model/client once ---
    t0 = time.time()
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        llm_kwargs = dict(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir=f"{CACHE}/hub",
        )
        if args.quantization and args.quantization.lower() != "none":
            llm_kwargs["quantization"] = args.quantization
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
        print(f"[shard] model loaded in {time.time()-t0:.0f}s "
              f"(tp={args.tensor_parallel}, quant={args.quantization})", flush=True)

        def chat(convs: list[list[dict]]) -> list[str]:
            try:
                outs = llm.chat(convs, sampling,
                                chat_template_kwargs={"enable_thinking": False},
                                use_tqdm=False)
            except TypeError:
                outs = llm.chat(convs, sampling, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
    else:
        api_key = os.environ.get(args.openrouter_api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{args.openrouter_api_key_env} is required for --backend openrouter"
            )
        print(f"[shard] using OpenRouter model={args.openrouter_model}", flush=True)

        def chat(convs: list[list[dict]]) -> list[str]:
            return openrouter_chat(
                convs,
                model=args.openrouter_model,
                api_key=api_key,
                base_url=args.openrouter_base_url,
                max_tokens=args.max_tokens,
            )

    # --- stream in batches; checkpoint after each ---
    n_done = 0
    t_gen = time.time()
    with open(out_path, "a") as out_fh:
        for bstart in range(0, len(todo), args.batch_profiles):
            batch = todo[bstart : bstart + args.batch_profiles]
            convs, idx = [], []
            for uid in batch:
                prof = profiles[uid]
                for chunk in chunk_list:
                    convs.append([{"role": "user", "content": build_amazon_prompt(prof, chunk)}])
                    idx.append((uid, chunk))
            outs = chat(convs)
            merged: dict[str, list] = {uid: [] for uid in batch}
            for (uid, chunk), text in zip(idx, outs):
                merged[uid].extend(
                    sanitize_fields(parse_fields(text), chunk, profiles[uid])
                )
            for uid in batch:
                out_fh.write(json.dumps(
                    {"user_id": uid, "user_bucket": bucket,
                     "review_count": int(review_count.get(uid, 0)),
                     "prepared_review_count": int(prepared_rows[uid].get("review_count", 0)),
                     "validation_review_count": int(prepared_rows[uid].get("validation_review_count", 0)),
                     "temporal_split": prepared_rows[uid].get("temporal_split"),
                     "review_filter_summary": prepared_rows[uid].get("review_filter_summary"),
                     "fields": merged[uid]}, ensure_ascii=False) + "\n")
            out_fh.flush()
            os.fsync(out_fh.fileno())
            n_done += len(batch)
            rate = n_done / max(1e-9, time.time() - t_gen)
            eta = (len(todo) - n_done) / max(1e-9, rate)
            print(f"[shard {args.shard_id}] {n_done}/{len(todo)} "
                  f"({100*n_done/len(todo):.1f}%)  {rate:.2f} user/s  "
                  f"ETA {eta/3600:.1f}h", flush=True)

    print(f"[shard {args.shard_id}] DONE {n_done} users in "
          f"{(time.time()-t_gen)/3600:.2f}h -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
