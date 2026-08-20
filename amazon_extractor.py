"""Build schema-constrained personas from local Amazon review data.

The pipeline is split into five resumable stages: ingest, prepare, compact,
extract and stats. Run ``python amazon_extractor.py --help`` for usage.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from tqdm.auto import tqdm
from persona_coverage_chart import render_category_coverage_chart

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value: Path) -> Path:
    """Resolve relative paths from the project directory, not the current shell."""
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Config:
    review_dir: Path
    metadata_dir: Path
    work_dir: Path
    schema_path: Path
    start_year: int = 2018
    end_year: int = 2023
    max_rows_per_file: int = 0
    top_k: int = 500
    min_reviews: int = 30
    min_text_chars: int = 2_000
    min_verified_share: float = 0.70
    train_fraction: float = 0.8
    min_review_text_chars: int = 20
    max_profile_chars: int = 48_000
    max_review_text_chars: int = 2_000
    max_dims_per_chunk: int = 20
    max_llm_users: int = 0
    review_shards: int = 256
    model: str = os.environ.get("LLM_MODEL", "Qwen3-14B")
    llm_endpoint: str = os.environ.get(
        "LLM_ENDPOINT", 
        "http://203.113.152.4:7777/llm/v1/chat/completions",
    )
    llm_authorization: str = os.environ.get("LLM_AUTHORIZATION", "")
    llm_timeout_seconds: int = 300

    @property
    def review_shards_dir(self) -> Path:
        return self.work_dir / "review_shards"

    @property
    def selected_users_path(self) -> Path:
        return self.work_dir / "selected_users.jsonl"

    @property
    def histories_path(self) -> Path:
        return self.work_dir / "user_histories.prepared.jsonl"

    @property
    def compact_profiles_path(self) -> Path:
        return self.work_dir / "compact_profiles.jsonl"

    @property
    def personas_path(self) -> Path:
        return self.work_dir / "personas_1290.jsonl"

    @property
    def persona_stats_path(self) -> Path:
        return self.work_dir / "persona_stats.json"

    @property
    def persona_coverage_chart_path(self) -> Path:
        return self.work_dir / "persona_category_coverage.png"

    @property
    def prompt_log_dir(self) -> Path:
        return self.work_dir / "prompt_log"

# ===== STANDALONE PRODUCTION HELPERS =====
# Các hàm dưới đây được đặt trực tiếp trong notebook; không import code từ project.
ASSIGNMENT_TYPES = {"direct", "structured_claim", "summary_inference", "unsupported"}
NULLISH_VALUES = {"", "null", "none", "n/a", "na", "unknown", "unsupported", "not applicable"}
FULFILLMENT_PATTERNS = [
    re.compile(pattern, re.I) for pattern in (
        r"\b(as expected|works as expected|just as described)\b",
        r"\b(fast shipping|quick shipping|arrived (on time|quickly|early))\b",
        r"\b(great product|good product|nice product|excellent product)\b",
        r"\b(would buy again|will buy again|recommend this product)\b",
        r"\b(five stars|four stars|three stars|two stars|one star)\b",
        r"^\s*(good|great|excellent|perfect|nice|ok|okay|love it|liked it)[.!]?\s*$",
    )
]

def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars - 15].rstrip() + " ... [truncated]"
    return text

def normalize_timestamp(value: Any) -> int | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp < 0:
        return None
    return timestamp * 1000 if timestamp < 10_000_000_000 else timestamp

def valid_rating(value: Any) -> bool:
    try:
        return 1 <= float(value) <= 5
    except (TypeError, ValueError):
        return False

def review_text(review: dict[str, Any]) -> str:
    return compact_text(review.get("text") or review.get("review_text"))

def review_title(review: dict[str, Any]) -> str:
    return compact_text(review.get("title") or review.get("review_title"))

def product_title(review: dict[str, Any]) -> str:
    return compact_text(review.get("product_title"))

def product_main_category(review: dict[str, Any]) -> str:
    return compact_text(review.get("product_main_category"))

def product_category_path(review: dict[str, Any]) -> list[str]:
    value = review.get("product_category_path") or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for item in value:
        if isinstance(item, list):
            candidates = item
        elif item:
            candidates = [item]
        else:
            candidates = []
        for candidate in candidates:
            text = compact_text(candidate)
            if text and text not in seen:
                seen.add(text); result.append(text)
            if len(result) >= 6:
                return result
    return result

def normalize_review_record(review: dict[str, Any]) -> dict[str, Any]:
    row = dict(review)
    row["timestamp"] = normalize_timestamp(row.get("timestamp", row.get("sort_timestamp")))
    if valid_rating(row.get("rating")):
        row["rating"] = float(row["rating"])
    row["text"] = review_text(row)
    row["title"] = review_title(row)
    row["product_category_path"] = product_category_path(row)
    return row

def fulfillment_match(review: dict[str, Any]) -> str | None:
    text = " ".join(part for part in (review_title(review), review_text(review)) if part)
    if not text or len(text) > 220:
        return None
    for pattern in FULFILLMENT_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None

def review_key(review: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(review.get("category") or ""), str(review.get("parent_asin") or ""),
        str(review.get("asin") or ""), str(normalize_timestamp(review.get("timestamp")) or ""),
        review_title(review).lower(), review_text(review).lower(), product_title(review).lower(),
    )

def filter_reviews(reviews: list[dict[str, Any]], *, min_review_text_chars: int, filter_fulfillment_reviews: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept, seen = [], set()
    removed_by_reason: dict[str, int] = defaultdict(int)
    removed_by_category: dict[str, int] = defaultdict(int)
    for raw_review in reviews:
        review = normalize_review_record(raw_review)
        category = str(review.get("category") or "Unknown")
        reason = None
        if review["timestamp"] is None:
            reason = "missing_or_invalid_timestamp"
        elif not valid_rating(review.get("rating")):
            reason = "missing_or_invalid_rating"
        elif not (product_title(review) or product_main_category(review) or product_category_path(review)) and not review_title(review) and len(review_text(review)) < min_review_text_chars:
            reason = "insufficient_text_evidence"
        elif review_key(review) in seen:
            reason = "duplicate_review"
        elif filter_fulfillment_reviews and fulfillment_match(review):
            reason = f"fulfillment_or_template:{fulfillment_match(review)}"
        if reason:
            removed_by_reason[reason] += 1
            removed_by_category[category] += 1
            continue
        seen.add(review_key(review))
        kept.append(review)
    kept.sort(key=lambda row: (row.get("timestamp") or 0, str(row.get("asin") or "")))
    return kept, {
        "input_reviews": len(reviews), "kept_reviews": len(kept),
        "removed_reviews": len(reviews) - len(kept),
        "removed_by_reason": dict(sorted(removed_by_reason.items())),
        "removed_by_category": dict(sorted(removed_by_category.items())),
    }

def temporal_train_validation_split(reviews: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < train_fraction < 1:
        raise ValueError("TRAIN_FRACTION must be in (0, 1)")
    split_index = max(1, min(int(len(reviews) * train_fraction), len(reviews) - 1))
    construction, validation = reviews[:split_index], reviews[split_index:]
    return construction, validation, {
        "method": "per_user_temporal", "unit": "review_or_rating_row",
        "train_fraction": train_fraction,
        "construction_row_count": len(construction), "validation_row_count": len(validation),
        "construction_text_review_count": sum(bool(review_text(row)) for row in construction),
        "validation_text_review_count": sum(bool(review_text(row)) for row in validation),
        "construction_rating_count": sum(valid_rating(row.get("rating")) for row in construction),
        "validation_rating_count": sum(valid_rating(row.get("rating")) for row in validation),
        "construction_last_timestamp": construction[-1].get("timestamp") if construction else None,
        "validation_first_timestamp": validation[0].get("timestamp") if validation else None,
    }

def category_review_stats(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for review in reviews:
        category = str(review.get("category") or "Unknown")
        item = stats.setdefault(category, {
            "review_count": 0, "text_review_count": 0, "rating_only_count": 0,
            "text_chars": 0, "rating_sum": 0.0, "rating_count": 0,
            "rating_counts": defaultdict(int), "product_category_counts": defaultdict(int),
            "rating_only_product_title_counts": defaultdict(int),
        })
        text = review_text(review)
        item["review_count"] += 1
        item["text_review_count"] += int(bool(text))
        is_rating_only = not (review_title(review) or text)
        item["rating_only_count"] += int(is_rating_only)
        item["text_chars"] += len(text)
        if valid_rating(review.get("rating")):
            rating = float(review["rating"])
            item["rating_sum"] += rating; item["rating_count"] += 1
            item["rating_counts"][str(int(rating) if rating.is_integer() else rating)] += 1
        for product_category in product_category_path(review):
            item["product_category_counts"][product_category] += 1
        if is_rating_only and product_title(review):
            item["rating_only_product_title_counts"][product_title(review)] += 1
    for item in stats.values():
        item["mean_rating"] = round(item["rating_sum"] / item["rating_count"], 3) if item["rating_count"] else None
        item.pop("rating_sum")
        for key in ("rating_counts", "product_category_counts", "rating_only_product_title_counts"):
            item[key] = dict(item[key])
    return stats

def format_counts(counts: Any, limit: int) -> str:
    if not isinstance(counts, dict):
        return ""
    items = sorted(counts.items(), key=lambda pair: (-int(pair[1] or 0), str(pair[0])))[:limit]
    return ", ".join(f"{key}={value}" for key, value in items)

def render_summary_stats(row: dict[str, Any], max_chars: int = 4_000) -> str:
    stats = row.get("category_review_stats") or {}
    split = row.get("temporal_split") or {}
    lines = ["=== Summary Stats ===", (
        f"construction_rows: {split.get('construction_row_count', 0)}; "
        f"validation_rows: {split.get('validation_row_count', 0)}; "
        f"construction_text_reviews: {split.get('construction_text_review_count', 0)}; "
        f"construction_ratings: {split.get('construction_rating_count', 0)}"
    )]
    for category, item in sorted(stats.items(), key=lambda pair: (-pair[1].get("review_count", 0), pair[0]))[:12]:
        parts = [
            f"category={category}", f"rows={item.get('review_count', 0)}",
            f"text_reviews={item.get('text_review_count', 0)}",
            f"rating_only={item.get('rating_only_count', 0)}",
            f"mean_rating={item.get('mean_rating'):.2f}" if isinstance(item.get('mean_rating'), (int, float)) else "mean_rating=unknown",
        ]
        for label, key, limit in (("ratings", "rating_counts", 5), ("product_categories", "product_category_counts", 4), ("rating_only_products", "rating_only_product_title_counts", 3)):
            rendered = format_counts(item.get(key), limit)
            if rendered:
                parts.append(f"{label}={rendered}")
        lines.append("; ".join(parts))
    return compact_text("\n".join(lines), max_chars)

def render_review(review: dict[str, Any], index: int, max_review_text_chars: int) -> str:
    lines = [
        f"[review {index}]", f"date: {review.get('date') or review.get('timestamp') or 'unknown'}", f"category: {review.get('category') or 'Unknown'}",
        f"parent_asin: {review.get('parent_asin') or review.get('asin') or ''}",
        f"rating: {review.get('rating', 'unknown')}", f"review_title: {review_title(review) or '(untitled)'}",
    ]
    if product_title(review): lines.append(f"product_title: {compact_text(product_title(review), 220)}")
    if product_main_category(review): lines.append(f"product_main_category: {compact_text(product_main_category(review), 120)}")
    if product_category_path(review): lines.append("product_category_path: " + " > ".join(compact_text(item, 80) for item in product_category_path(review)))
    lines.extend([
        f"verified: {review.get('verified_purchase', 'unknown')}",
        f"helpful_vote: {review.get('helpful_vote', 0)}",
        f"text: {compact_text(review_text(review), max_review_text_chars)}",
    ])
    return "\n".join(lines)

def assemble_profile(row: dict[str, Any], max_chars: int, max_review_text_chars: int) -> str:
    reviews = row.get("reviews") or []
    categories = {review.get("category") or "Unknown" for review in reviews}
    parts = [f"Amazon reviewer profile - construction split: {len(reviews)} reviews across {len(categories)} categories."]
    summary = render_summary_stats(row)
    if summary: parts.append(summary)
    parts.extend(render_review(review, index, max_review_text_chars) for index, review in enumerate(reviews, 1))
    return "\n\n".join(parts)[:max_chars]

def build_amazon_prompt(profile_text: str, dimensions: list[dict[str, Any]]) -> str:
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
        '"confidence": <float between 0.0 and 1.0>, '
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
        "- For supported attributes, estimate confidence as a float between 0.5 and 1.0 based on the strength and frequency of evidence.",
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

def parse_fields(text: str) -> list[dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start: return []
    try: payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError: return []
    fields = payload.get("fields") if isinstance(payload, dict) else None
    return fields if isinstance(fields, list) else []

def unsupported_field(dimension: dict[str, Any]) -> dict[str, Any]:
    return {"field_id": str(dimension["id"]), "value": None, "confidence": 0.0, "evidence": "", "description": "", "assignment_type": "unsupported"}

def normalized_key(value: str) -> str:
    return " ".join(str(value).replace("-", "–").split()).casefold()

def coerce_value(value: Any, dimension: dict[str, Any]) -> str | None:
    if value is None or str(value).strip().casefold() in NULLISH_VALUES: return None
    text = str(value).strip(); allowed = [str(item) for item in dimension.get("values", [])]
    if not allowed: return text
    if text in allowed: return text
    return {normalized_key(item): item for item in allowed}.get(normalized_key(text))

def confidence_value(value: Any) -> float:
    try: return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError): return 0.0

def quote_is_in_profile(evidence: str, profile_text: str) -> bool:
    return bool(evidence) and (evidence in profile_text or " ".join(evidence.split()) in " ".join(profile_text.split()))

def sanitize_fields(fields: list[dict[str, Any]], dimensions: list[dict[str, Any]], profile_text: str) -> list[dict[str, Any]]:
    dimensions_by_id = {str(dimension["id"]): dimension for dimension in dimensions}
    best: dict[str, dict[str, Any]] = {}
    for raw in fields:
        if not isinstance(raw, dict): continue
        field_id = str(raw.get("field_id") or "").strip(); dimension = dimensions_by_id.get(field_id)
        if dimension is None: continue
        assignment_type = str(raw.get("assignment_type") or "").strip()
        value = coerce_value(raw.get("value"), dimension)
        confidence = confidence_value(raw.get("confidence"))
        evidence = str(raw.get("evidence") or "").strip()
        supported = value is not None and assignment_type in ASSIGNMENT_TYPES and assignment_type != "unsupported" and quote_is_in_profile(evidence, profile_text)
        clean = {"field_id": field_id, "value": value, "confidence": confidence, "evidence": evidence, "description": str(raw.get("description") or "").strip(), "assignment_type": assignment_type} if supported else unsupported_field(dimension)
        prior = best.get(field_id)
        if prior is None or (clean["value"] is not None and prior["value"] is None) or (bool(clean["value"]) == bool(prior["value"]) and clean["confidence"] > prior["confidence"]):
            best[field_id] = clean
    return [best.get(str(dimension["id"])) or unsupported_field(dimension) for dimension in dimensions]

def cat_chunks(by_category: dict[str, list[dict[str, Any]]], per_chunk: int) -> list[list[dict[str, Any]]]:
    chunks = []
    for dimensions in by_category.values():
        chunks.extend(dimensions[start:start + per_chunk] for start in range(0, len(dimensions), per_chunk))
    return chunks

def iter_local_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def local_jsonl_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted({*root.rglob("*.jsonl"), *root.rglob("*.jsonl.gz")})


def category_from_path(path: Path, metadata: bool = False) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    if metadata and name.startswith("meta_"):
        name = name[5:]
    # Cho phép các tên chuẩn như Books, Books_part_000, Books_sample_50000.
    for marker in ("_part_", "_sample_"):
        if marker in name:
            name = name.split(marker, 1)[0]
    return name


def timestamp_ms(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number * 1000 if number < 10_000_000_000 else number


def year_of(timestamp: int) -> int:
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).year


def stable_review_id(row: dict[str, Any], category: str) -> str:
    if row.get("review_id"):
        return str(row["review_id"])
    identity = "|".join(str(row.get(key) or "") for key in (
        "user_id", "parent_asin", "asin", "timestamp", "rating", "text"
    ))
    return hashlib.sha1(f"{category}|{identity}".encode()).hexdigest()



def shard_path(config: Config, shard_index: int) -> Path:
    return config.review_shards_dir / f"reviews-{shard_index:04d}.jsonl"


def user_shard(user_id: str, shard_count: int) -> int:
    digest = hashlib.sha1(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % shard_count


def iter_review_shards(config: Config) -> Iterator[Path]:
    for index in range(config.review_shards):
        path = shard_path(config, index)
        if path.is_file():
            yield path


def ingest_reviews(config: Config) -> None:
    files = local_jsonl_files(config.review_dir)
    if not files:
        raise FileNotFoundError(f"No .jsonl/.jsonl.gz files found in {config.review_dir}")
    config.review_shards_dir.mkdir(parents=True, exist_ok=True)
    handles = [shard_path(config, index).open("w", encoding="utf-8") for index in range(config.review_shards)]
    total_kept = 0
    try:
        for path in files:
            fallback_category = category_from_path(path)
            scanned = kept = 0
            for row in tqdm(iter_local_jsonl(path), desc=f"ingest:{path.name}"):
                scanned += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                user_id = str(row.get("user_id") or "").strip()
                timestamp = timestamp_ms(row.get("timestamp", row.get("sort_timestamp")))
                if not user_id or timestamp is None or not config.start_year <= year_of(timestamp) <= config.end_year:
                    continue
                category = str(row.get("category") or fallback_category or "Unknown")
                review_id = stable_review_id(row, category)
                record = {"review_id": review_id, "user_id": user_id, "category": category,
                    "parent_asin": str(row.get("parent_asin") or ""), "asin": str(row.get("asin") or ""),
                    "timestamp": timestamp, "rating": row.get("rating"), "title": str(row.get("title") or ""),
                    "text": str(row.get("text") or ""), "verified_purchase": row.get("verified_purchase") is True,
                    "helpful_vote": int(row.get("helpful_vote", row.get("helpful_votes", 0)) or 0)}
                handle = handles[user_shard(user_id, config.review_shards)]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1; total_kept += 1
            print(f"{path.name}: scanned={scanned:,}, kept={kept:,}")
    finally:
        for handle in handles:
            handle.close()
    print(f"Stored reviews={total_kept:,} in {config.review_shards} JSONL shards")


def percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        percentile = ((position + 1 + end) / 2) / len(order)
        for index in order[position:end]:
            result[index] = percentile
        position = end
    return result


def select_users(config: Config) -> None:
    if not any(iter_review_shards(config)):
        raise FileNotFoundError(f"No review shards in {config.review_shards_dir}. Run ingest first.")
    aggregate: dict[str, dict[str, Any]] = {}
    for path in tqdm(list(iter_review_shards(config)), desc="select users"):
        seen_review_ids: set[str] = set()
        for row in iter_local_jsonl(path):
            review_id = str(row.get("review_id") or "")
            if review_id in seen_review_ids:
                continue
            seen_review_ids.add(review_id)
            user_id = row["user_id"]
            item = aggregate.setdefault(user_id, {"count": 0, "categories": set(), "text_reviews": 0,
                "text_chars": 0, "min_ts": row["timestamp"], "max_ts": row["timestamp"], "verified": 0})
            text = str(row.get("text") or "")
            item["count"] += 1; item["categories"].add(row.get("category") or "Unknown")
            item["text_reviews"] += int(bool(text.strip())); item["text_chars"] += len(text)
            item["min_ts"] = min(item["min_ts"], row["timestamp"]); item["max_ts"] = max(item["max_ts"], row["timestamp"])
            item["verified"] += int(row.get("verified_purchase") is True)
    eligible = []
    for user_id, item in aggregate.items():
        verified_share = item["verified"] / item["count"]
        if item["count"] >= config.min_reviews and item["text_chars"] >= config.min_text_chars and verified_share >= config.min_verified_share:
            eligible.append((user_id, item["count"], len(item["categories"]), item["text_reviews"], item["text_chars"],
                             (item["max_ts"] - item["min_ts"]) / 86_400_000, verified_share))
    metric_columns = (4, 3, 2, 5, 1, 6)
    weights = (0.35, 0.20, 0.20, 0.15, 0.05, 0.05)
    ranks = [percentile_ranks([float(row[column] or 0) for row in eligible]) for column in metric_columns]
    scored = [(sum(weight * vector[index] for weight, vector in zip(weights, ranks)), row)
              for index, row in enumerate(eligible)]
    scored.sort(key=lambda item: (-item[0], -item[1][4], -item[1][2], -item[1][5], item[1][0]))
    with config.selected_users_path.open("w", encoding="utf-8") as output:
        for rank, (score, row) in enumerate(scored[:config.top_k], 1):
            keys = ("user_id", "review_count", "category_count", "text_reviews", "text_chars", "history_days", "verified_share")
            record = {key: value for key, value in zip(keys, row)}
            record.update({"rank": rank, "score": score})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Eligible users={len(eligible):,}; selected={min(config.top_k, len(scored)):,}")


def load_metadata(config: Config, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for path in local_jsonl_files(config.metadata_dir):
        fallback_category = category_from_path(path, metadata=True)
        for meta in tqdm(iter_local_jsonl(path), desc=f"metadata:{path.name}"):
            category = str(meta.get("category") or fallback_category or "Unknown")
            parent_asin = str(meta.get("parent_asin") or "")
            if (category, parent_asin) not in wanted:
                continue
            metadata[(category, parent_asin)] = {"product_title": str(meta.get("title") or "")[:500],
                "product_main_category": str(meta.get("main_category") or "")[:200],
                "product_category_path": meta.get("categories") or []}
    print(f"Metadata matches loaded: {len(metadata):,}")
    return metadata


def prepare_histories(config: Config) -> None:
    require_file(config.selected_users_path, "Run the prepare selection after ingest")
    with config.selected_users_path.open(encoding="utf-8") as source:
        selected_records = [json.loads(line) for line in source if line.strip()]
    selected = {str(row["user_id"]): row for row in selected_records}
    wanted: set[tuple[str, str]] = set()
    shard_files = list(iter_review_shards(config))
    for path in tqdm(shard_files, desc="find selected products"):
        for row in iter_local_jsonl(path):
            if str(row["user_id"]) in selected and row.get("parent_asin"):
                wanted.add((str(row.get("category") or "Unknown"), str(row["parent_asin"])))
    metadata = load_metadata(config, wanted)
    with config.histories_path.open("w", encoding="utf-8") as output:
        for path in tqdm(shard_files, desc="prepare histories"):
            reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for review in iter_local_jsonl(path):
                user_id = str(review["user_id"])
                if user_id in selected:
                    product = metadata.get((str(review.get("category") or "Unknown"), str(review.get("parent_asin") or "")), {})
                    review.update(product)
                    reviews_by_user[user_id].append(review)
            for user_id, raw_reviews in reviews_by_user.items():
                reviews, filter_summary = filter_reviews(raw_reviews, min_review_text_chars=config.min_review_text_chars,
                                                           filter_fulfillment_reviews=True)
                if len(reviews) < 2:
                    continue
                construction, validation, split = temporal_train_validation_split(reviews, config.train_fraction)
                record = {"source": "amazon_reviews_2023", "user_id": user_id, "rank": selected[user_id]["rank"],
                    "review_count": len(reviews), "validation_review_count": len(validation), "temporal_split": split,
                    "review_filter_summary": filter_summary, "reviews": construction, "validation_reviews": validation,
                    "category_review_stats": category_review_stats(construction),
                    "validation_category_review_stats": category_review_stats(validation)}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Prepared histories:", config.histories_path)


def compact_profiles(config: Config) -> None:
    require_file(config.histories_path, "Run the prepare stage first")
    count = 0
    with config.histories_path.open(encoding="utf-8") as source, config.compact_profiles_path.open("w", encoding="utf-8") as output:
        for line in tqdm(source, desc="compact profiles"):
            if not line.strip():
                continue
            user = json.loads(line)
            profile = assemble_profile(user, config.max_profile_chars, config.max_review_text_chars)
            record = {key: user[key] for key in ("user_id", "source", "review_count", "validation_review_count",
                                                   "temporal_split", "review_filter_summary")}
            record.update({"compact_profile_chars": len(profile), "max_profile_chars": config.max_profile_chars,
                           "max_review_text_chars": config.max_review_text_chars, "profile_text": profile})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Compact profiles={count:,}; output={config.compact_profiles_path}")


def load_schema(config: Config) -> list[dict[str, Any]]:
    require_file(config.schema_path, "Provide --schema-path")
    document = json.loads(config.schema_path.read_text(encoding="utf-8"))
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"Invalid schema: {config.schema_path}")
    return dimensions


def call_llm(prompt: str, config: Config) -> str:
    headers = {"Content-Type": "application/json"}
    if config.llm_authorization:
        headers["Authorization"] = config.llm_authorization
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 30,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        config.llm_endpoint,
        headers=headers,
        json=payload,
        timeout=config.llm_timeout_seconds,
    )
    response.raise_for_status()
    result = response.json()
    try:
        return str(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Unexpected LLM response structure: {result}") from error


def reset_prompt_log(config: Config) -> None:
    """Keep prompt logs for the currently processed user only."""
    config.prompt_log_dir.mkdir(parents=True, exist_ok=True)
    for path in config.prompt_log_dir.glob("*.txt"):
        if path.is_file():
            path.unlink()


def save_prompt_log(config: Config, chunk_index: int, prompt: str) -> Path:
    path = config.prompt_log_dir / f"prompt-{chunk_index:04d}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def extract_personas(config: Config) -> None:
    require_file(config.compact_profiles_path, "Run the compact stage first")
    schema = load_schema(config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dimension in schema:
        grouped[dimension.get("category", "Uncategorized")].append(dimension)
    chunks = cat_chunks(grouped, config.max_dims_per_chunk)
    done = set()
    if config.personas_path.exists():
        with config.personas_path.open(encoding="utf-8") as existing:
            done = {str(json.loads(line)["user_id"]) for line in existing if line.strip()}
    processed = 0
    with config.compact_profiles_path.open(encoding="utf-8") as source, config.personas_path.open("a", encoding="utf-8") as output:
        for line in tqdm(source, desc="extract personas"):
            if not line.strip():
                continue
            record = json.loads(line)
            user_id = str(record["user_id"])
            if user_id in done:
                continue
            if config.max_llm_users and processed >= config.max_llm_users:
                break
            reset_prompt_log(config)
            fields = []
            for chunk_index, dimensions in enumerate(chunks, start=1):
                started_at = time.perf_counter()
                prompt = build_amazon_prompt(record["profile_text"], dimensions)
                prompt_path = save_prompt_log(config, chunk_index, prompt)
                response = call_llm(prompt, config)
                chunk_fields = sanitize_fields(parse_fields(response), dimensions, record["profile_text"])
                fields.extend(chunk_fields)
                categories = sorted({str(dimension.get("category") or "Uncategorized") for dimension in dimensions})
                supported_count = sum(field["value"] is not None for field in chunk_fields)
                elapsed_seconds = time.perf_counter() - started_at
                tqdm.write(
                    f"user={user_id} chunk={chunk_index}/{len(chunks)} "
                    f"category={','.join(categories)} dimensions={len(dimensions)} "
                    f"supported={supported_count} elapsed={elapsed_seconds:.2f}s "
                    f"prompt_log={prompt_path.name}"
                )
            if len(fields) != len(schema):
                raise RuntimeError(f"Expected {len(schema)} fields, got {len(fields)} for {user_id}")
            result = {key: record[key] for key in ("user_id", "source", "review_count", "validation_review_count",
                                                    "temporal_split", "review_filter_summary", "compact_profile_chars")}
            result["fields"] = fields
            output.write(json.dumps(result, ensure_ascii=False) + "\n"); output.flush(); os.fsync(output.fileno())
            done.add(user_id); processed += 1
    print(f"New personas={processed}; output={config.personas_path}")


def generate_persona_stats(config: Config) -> None:
    """Summarize supported persona dimensions overall and by schema category."""
    require_file(config.personas_path, "Run the extract stage first")
    schema = load_schema(config)
    field_categories = {
        str(dimension["id"]): str(dimension.get("category") or "Uncategorized")
        for dimension in schema
    }
    schema_dimensions_by_category: dict[str, int] = defaultdict(int)
    for category in field_categories.values():
        schema_dimensions_by_category[category] += 1
    seen_users: set[str] = set()
    supported_dimensions_total = 0
    supported_by_category: dict[str, int] = defaultdict(int)

    with config.personas_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                persona = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {config.personas_path}:{line_number}") from error
            user_id = str(persona.get("user_id") or "").strip()
            if not user_id or user_id in seen_users:
                continue
            seen_users.add(user_id)
            for field in persona.get("fields") or []:
                if not isinstance(field, dict) or field.get("value") is None:
                    continue
                supported_dimensions_total += 1
                field_id = str(field.get("field_id") or "")
                category = field_categories.get(field_id, "Unknown field category")
                supported_by_category[category] += 1

    persona_count = len(seen_users)
    category_stats = [
        {"category": category, "supported_dimension_count": count}
        for category, count in sorted(
            supported_by_category.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    stats = {
        "persona_count": persona_count,
        "schema_dimension_count": len(schema),
        "supported_dimension_count": supported_dimensions_total,
        "average_supported_dimensions_per_persona": round(
            supported_dimensions_total / persona_count, 4
        ) if persona_count else 0.0,
        "categories": category_stats,
    }
    config.persona_stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chart_rows = []
    for category, schema_dimension_count in schema_dimensions_by_category.items():
        supported_count = supported_by_category.get(category, 0)
        denominator = persona_count * schema_dimension_count
        chart_rows.append({
            "category": category,
            "coverage_percentage": 100.0 * supported_count / denominator if denominator else 0.0,
        })
    chart_rows.sort(key=lambda row: (-row["coverage_percentage"], row["category"]))
    render_category_coverage_chart(
        chart_rows,
        config.persona_coverage_chart_path,
        f"Category Coverage Analysis (Amazon Personas: {persona_count})",
    )
    print(f"Extracted personas: {persona_count:,}")
    print(f"Average supported dimensions/persona: {stats['average_supported_dimensions_per_persona']:.4f}")
    for item in category_stats:
        print(f"{item['category']}: {item['supported_dimension_count']:,}")
    print("Persona stats:", config.persona_stats_path)
    print("Persona coverage chart:", config.persona_coverage_chart_path)


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}. {hint}.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "ingest", "prepare", "compact", "extract", "stats"), nargs="?", default="all")
    parser.add_argument("--review-dir", type=Path, default=Path("data/amazon/reviews"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/amazon/metadata"))
    parser.add_argument("--work-dir", type=Path, default=Path("amazon_persona_fresh"))
    parser.add_argument("--schema-path", type=Path, default=Path("schema/dimension.json"))
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--top-k", type=int, default=100_000)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-llm-users", type=int, default=0)
    parser.add_argument("--review-shards", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(review_dir=project_path(args.review_dir), metadata_dir=project_path(args.metadata_dir),
                    work_dir=project_path(args.work_dir), schema_path=project_path(args.schema_path),
                    start_year=args.start_year, end_year=args.end_year,
                    top_k=args.top_k, max_rows_per_file=args.max_rows_per_file,
                    max_llm_users=args.max_llm_users, review_shards=args.review_shards)
    if config.start_year > config.end_year:
        raise ValueError("start-year must not be greater than end-year")
    if config.review_shards < 1:
        raise ValueError("review-shards must be at least 1")
    print("Work directory:", config.work_dir.resolve())
    config.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "ingest"}:
        ingest_reviews(config)
    if args.stage in {"all", "prepare"}:
        select_users(config)
        prepare_histories(config)
    if args.stage in {"all", "compact"}:
        compact_profiles(config)
    if args.stage in {"all", "extract"}:
        extract_personas(config)
    if args.stage in {"all", "stats"}:
        generate_persona_stats(config)


if __name__ == "__main__":
    main()
