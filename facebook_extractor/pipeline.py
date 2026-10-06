"""Build schema-constrained personas from Facebook profiles and content.

The pipeline is split into five resumable stages: ingest, prepare, compact,
extract and stats. Run ``python facebook_extractor.py --help`` for usage.
"""

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm
from llm_client import LLMCancelledError, LLMSettings, LLMUnauthorizedError, complete_prompt
from persona_coverage_chart import render_category_coverage_chart
from .config import FRESH_BATCH_NAME_START, NUM_FRESH_BATCHES, Config
from .content_selection import select_contents
from .filters import is_advertising
from .records import (
    assemble_profile,
    compact_text,
    filter_posts,
    format_vietnam_datetime,
    parse_comment_date,
)

csv.field_size_limit(100_000_000)

ASSIGNMENT_TYPES = {"direct", "structured_claim", "summary_inference", "unsupported"}
NULLISH_VALUES = {"", "null", "none", "n/a", "na", "unknown", "unsupported", "not applicable"}

# Populated via ProcessPoolExecutor(initializer=...) so each preprocessing worker
# process gets its own copy once, instead of re-pickling it on every submitted task.
_worker_config: Config | None = None
_worker_selected: dict[str, dict[str, Any]] = {}
_worker_profiles: dict[str, dict[str, Any]] = {}


def _init_config_worker(config: Config) -> None:
    global _worker_config
    _worker_config = config


def _init_prepare_worker(
    config: Config,
    selected: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> None:
    global _worker_config, _worker_selected, _worker_profiles
    _worker_config = config
    _worker_selected = selected
    _worker_profiles = profiles


def capped_score(value: float, maximum: float) -> float:
    return min(max(value, 0.0) / maximum, 1.0)


def user_selection_score(
    post_count: float, text_chars: float, history_days: float, config: Config
) -> float:
    return (
        0.4375 * capped_score(text_chars, config.max_character_score_at)
        + 0.1875 * capped_score(history_days, config.max_history_score_at)
        + 0.375 * capped_score(post_count, config.max_content_score_at)
    )


def advertising_score_factor(advertising_ratio: float) -> float:
    ratio = max(float(advertising_ratio), 0.0)
    if ratio == 0:
        return 1.0
    if ratio < 0.05:
        return 0.90
    if ratio < 0.15:
        return 0.75
    if ratio < 0.30:
        return 0.55
    if ratio < 0.50:
        return 0.30
    return 0.0


def build_post_prompt(profile_text: str, dimensions: list[dict[str, Any]]) -> str:
    """Build a schema-constrained prompt for a Vietnamese Facebook user."""
    lines = [
        "You are mapping a declared Vietnamese Facebook profile and that user's posts/comments "
        "to schema-constrained persona fields. Fill attributes that are well supported "
        "by the profile or content history, and leave unsupported claims null.",
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
        '"evidence": "<one short exact quote copied from USER DATA, or empty string>", '
        '"description": "<1-2 concrete sentences, or empty string>", '
        '"assignment_type": "direct|structured_claim|summary_inference|unsupported"}]}',
        "",
        "Allowed support:",
        "- direct: use for an explicit [declared profile] value, or when the user explicitly states the fact about themselves in content text.",
        "- structured_claim: use for repeated concrete non-sensitive claims supported by at least 2 distinct posts or threads.",
        "- summary_inference: use for non-sensitive interests, participation behavior, communication style, or expertise when a repeated pattern is visible across the posting history.",
        "- Overall writing style may support communication/cognitive-style "
        "dimensions only when the pattern is visible across at least 5 posts.",
        "- unsupported: use when evidence is absent, one-off, ambiguous, generic, "
        "or mainly about someone other than the Facebook user.",
        "",
        "Hard limits:",
        "- Treat all advertising, promotional, sales, quotation, product-listing, "
        "lead-generation, and invitations to buy goods or services as unusable noise.",
        "- Do NOT extract, infer, summarize, or support ANY persona field from such "
        "commercial content, even when it is repeated or explicitly written by the user.",
        "- Commercial content must not be used as evidence for occupation, employment, "
        "income, expertise, interests, preferences, lifestyle, location, personality, "
        "communication style, or any other dimension.",
        "- Never quote advertising or sales content in evidence or descriptions. Ignore "
        "calls to contact/inbox, prices, phone numbers, promotions, ordering, shipping, "
        "stock availability, shop links, and invitations to purchase goods or services.",
        "- For age, gender, health, disability, ethnicity, religion, politics, "
        "income, family/household status, occupation, location, employment, and "
        "parenthood: assign a non-null value only from an explicit self-statement. "
        "A matching [declared profile] field counts as explicit. Do not infer them from topic, quoted news, or other participants.",
        "- Do not attribute claims from quoted material or another participant to the member.",
        "- Do not infer personality inventories, values, worldview, MBTI, Big "
        "Five, HEXACO, clinical attributes, or mental-state attributes from "
        "ordinary Facebook content unless the user explicitly states the "
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
        "posts over one isolated comment.",
        "- For supported attributes, estimate confidence as a float between 0.5 and 1.0 based on the strength and frequency of evidence.",
        "- If the posts do not support a dimension, set value to null, "
        'confidence to 0.0, evidence to "", assignment_type to "unsupported", '
        'and description to "".',
        "- Every non-null value MUST include a short evidence quote copied "
        "verbatim from the declared profile or one of the content items.",
        "- Evidence must be an exact quote from USER DATA, not your reasoning, "
        "a paraphrase, or a summary. If you cannot copy an exact quote, return "
        "unsupported.",
        "- If you cannot copy an exact quote, return unsupported.",
        "- Do not append support counts, explanations, or labels to evidence. "
        "Evidence must be only text that appears in USER DATA.",
        "- description: 1-2 concrete Vietnamese sentences describing THIS Facebook user using details from their profile and content. Describe the person; do not justify the label.",
        "- Every non-empty description MUST be written in Vietnamese. Do not "
        "write the description in English or any other language.",
        "- Sensitive / high-risk fields require explicit self-statements: age, "
        "gender, income, marital status, children count, religion, politics, "
        "ethnicity, health, disability, mental health, neurotype, MBTI, Big Five, "
        "personality traits, attachment style, and relationship style.",
        "- Do not infer these fields from thread topic, quoted content, writing style, tone, or vocabulary.",
        "- Return valid JSON only, with no markdown.",
        "- Most dimensions can be unsupported. Do not make the persona complete.",
        "",
        "DIMENSIONS (field_id | label | description | allowed values):",
    ]
    
    for d in dimensions:
        allowed = " | ".join(str(v) for v in d.get("values", [])) or "(free value)"
        desc = str(d.get("description", "")).strip()
        lines.append(f"- {d['id']} | {d.get('label', d['id'])} | {desc} | [{allowed}]")
        
    lines += ["", "USER DATA:", profile_text]
    
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
    return " ".join(str(value).replace("–", "-").replace("—", "-").split()).casefold()

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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_json_array(path: Path, chunk_size: int = 1_048_576) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    with path.open("r", encoding="utf-8-sig") as handle:
        while True:
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] in "[,\n\r"):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer) or len(buffer) - position < chunk_size // 4:
                chunk = handle.read(chunk_size)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] in "[,\n\r"):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer) and eof:
                    return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"Invalid JSON array in {path}")
                chunk = handle.read(chunk_size)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
                continue
            position = end
            if isinstance(value, dict):
                yield value


def local_csv_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def local_json_files(root: Path) -> list[Path]:
    """Return supported Facebook exports in deterministic order."""
    if not root.exists():
        return []
    return sorted((*root.rglob("*.jsonl"), *root.rglob("*.json")))


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_local_jsonl(path)
    else:
        yield from iter_json_array(path)


def category_from_path(path: Path, metadata: bool = False) -> str:
    if not metadata:
        return path.stem
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


def stable_post_id(row: dict[str, Any], category: str) -> str:
    explicit = compact_text(row.get("post_id"))
    if explicit:
        return explicit
    identity = "|".join(str(row.get(key) or "") for key in (
        "user_id", "timestamp", "thread_id", "text_hash", "text"
    ))
    return hashlib.sha1(f"{category}|{identity}".encode()).hexdigest()



def shard_path(config: Config, shard_index: int) -> Path:
    return config.post_shards_dir / f"posts-{shard_index:04d}.jsonl"


def shard_path_in(work_dir: Path, shard_index: int) -> Path:
    return work_dir / "post_shards" / f"posts-{shard_index:04d}.jsonl"


def user_shard(user_id: str, shard_count: int) -> int:
    digest = hashlib.sha1(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % shard_count


def iter_post_shards(config: Config) -> Iterator[Path]:
    for index in range(config.post_shards):
        path = shard_path(config, index)
        if path.is_file():
            yield path


def history_shard_path(config: Config, shard_index: int) -> Path:
    return config.history_shards_dir / f"histories-{shard_index:04d}.jsonl"


def iter_history_shards(config: Config) -> Iterator[Path]:
    for index in range(config.post_shards):
        path = history_shard_path(config, index)
        if path.is_file():
            yield path


def find_user_history(config: Config, user_id: str) -> Path | None:
    """Return the history shard containing ``user_id``, if it exists."""
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be empty")

    expected_index = user_shard(normalized_user_id, config.post_shards)
    expected_path = history_shard_path(config, expected_index)
    candidates = [expected_path]
    candidates.extend(
        path for path in iter_history_shards(config) if path != expected_path
    )
    for path in candidates:
        if not path.is_file():
            continue
        for record in iter_local_jsonl(path):
            if str(record.get("user_id") or "").strip() == normalized_user_id:
                return path
    return None


def compact_shard_path(config: Config, shard_index: int) -> Path:
    return config.compact_shards_dir / f"compact-{shard_index:04d}.jsonl"


def iter_compact_shards(config: Config) -> Iterator[Path]:
    for index in range(config.post_shards):
        path = compact_shard_path(config, index)
        if path.is_file():
            yield path


def iter_compact_profiles(config: Config) -> Iterator[dict[str, Any]]:
    for path in iter_compact_shards(config):
        yield from iter_local_jsonl(path)


def _reshard_batch(args: tuple[int, Path, Path]) -> int:
    batch_number, raw_path, work_dir = args
    config = _worker_config
    shard_handles = [
        shard_path_in(work_dir, index).open("w", encoding="utf-8")
        for index in range(config.post_shards)
    ]
    try:
        for record in iter_local_jsonl(raw_path):
            shard_index = user_shard(record["user_id"], config.post_shards)
            shard_handles[shard_index].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in shard_handles:
            handle.close()
    raw_path.unlink()
    return batch_number


def ingest_posts(config: Config) -> None:
    """Read every source batch and fan users out across NUM_FRESH_BATCHES output batches.

    Each user is assigned to exactly one output batch (stable hash of user_id), so this
    is a single streaming pass over the source data rather than one pass per output batch.
    """
    user_files = [path for source_dir in config.user_dirs for path in local_json_files(source_dir)]
    content_files = [path for source_dir in config.content_dirs for path in local_json_files(source_dir)]
    if not user_files:
        raise FileNotFoundError(f"No .jsonl/.json files found in {list(config.user_dirs)}")
    if not content_files:
        raise FileNotFoundError(f"No .jsonl/.json files found in {list(config.content_dirs)}")

    batch_numbers = range(FRESH_BATCH_NAME_START, FRESH_BATCH_NAME_START + NUM_FRESH_BATCHES)
    batch_dirs = {number: config.fresh_batch_dir(number) for number in batch_numbers}
    for work_dir in batch_dirs.values():
        (work_dir / "post_shards").mkdir(parents=True, exist_ok=True)

    excluded_profile_fields = {"phone", "number_follow", "number_friend", "averageReact", "hobby"}
    target_user_batch: dict[str, int] = {}
    profile_handles = {
        number: (work_dir / "user_profiles.jsonl").open("w", encoding="utf-8")
        for number, work_dir in batch_dirs.items()
    }
    try:
        for path in user_files:
            scanned = kept = 0
            for row in tqdm(iter_json_records(path), desc=f"users:{path.name}"):
                scanned += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                user_id = compact_text(row.get("id"))
                if not user_id or user_id in target_user_batch:
                    continue
                batch_number = user_shard(user_id, NUM_FRESH_BATCHES) + FRESH_BATCH_NAME_START
                target_user_batch[user_id] = batch_number
                profile = {key: value for key, value in row.items() if key not in excluded_profile_fields}
                profile["id"] = user_id
                profile_handles[batch_number].write(json.dumps(profile, ensure_ascii=False) + "\n")
                kept += 1
            print(f"{path.name}: profiles scanned={scanned:,}, kept={kept:,}")
    finally:
        for handle in profile_handles.values():
            handle.close()

    # Pass 1: stream all source content once, routing each row to its output batch's
    # raw file. Only NUM_FRESH_BATCHES handles are open at a time (not
    # NUM_FRESH_BATCHES * post_shards), so this stays well under OS file-handle limits.
    raw_paths = {number: batch_dirs[number] / "post_shards" / "_raw.jsonl" for number in batch_numbers}
    raw_handles = {number: path.open("w", encoding="utf-8") for number, path in raw_paths.items()}
    total_kept = 0
    try:
        for path in content_files:
            scanned = kept = 0
            for row in tqdm(iter_json_records(path), desc=f"content:{path.name}"):
                scanned += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                user_id = compact_text(row.get("author_id"))
                timestamp = parse_comment_date(row.get("published_time"))
                batch_number = target_user_batch.get(user_id)
                if batch_number is None or timestamp is None:
                    continue
                category = compact_text(row.get("article_type") or "facebook_content")
                text = str(row.get("content") or "")
                record = {
                    "user_id": user_id,
                    "category": category,
                    "source": "facebook",
                    "text": text,
                    "timestamp": format_vietnam_datetime(timestamp),
                    "like_count": int(row.get("like_count") or 0),
                    "share_count": int(row.get("share_count") or 0),
                    "comment_count": int(row.get("comment_count") or 0),
                    "reply_count": int(row.get("reply_count") or 0),
                }
                raw_handles[batch_number].write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1; total_kept += 1
            print(f"{path.name}: scanned={scanned:,}, kept={kept:,}")
    finally:
        for handle in raw_handles.values():
            handle.close()

    # Pass 2: reshard each output batch's raw file into config.post_shards JSONL
    # shards. Each batch's raw file is independent, so batches reshard in parallel.
    reshard_args = [
        (number, raw_paths[number], batch_dirs[number]) for number in batch_numbers
    ]
    with ProcessPoolExecutor(
        max_workers=config.preprocess_workers,
        initializer=_init_config_worker,
        initargs=(config,),
    ) as executor:
        for _ in tqdm(
            executor.map(_reshard_batch, reshard_args),
            total=len(reshard_args), desc="reshard output batches",
        ):
            pass

    batch_user_counts: dict[int, int] = defaultdict(int)
    for batch_number in target_user_batch.values():
        batch_user_counts[batch_number] += 1
    print(f"Target profiles={len(target_user_batch):,}; joined content={total_kept:,} "
          f"across {NUM_FRESH_BATCHES} output batches ({config.post_shards} shards each)")
    for number in batch_numbers:
        print(f"  {batch_dirs[number].name}: users={batch_user_counts.get(number, 0):,}")


def _select_users_shard(path: Path) -> list[tuple]:
    """Aggregate and filter one post shard's users.

    Shards are keyed by ``user_shard(user_id, ...)``, so every user's posts land
    in exactly one shard and shards can be scored independently in parallel.
    """
    config = _worker_config
    aggregate: dict[str, dict[str, Any]] = {}
    seen_posts: set[tuple[str, ...]] = set()
    for row in iter_local_jsonl(path):
        duplicate_key = (
            str(row.get("user_id") or ""), str(row.get("category") or ""),
            str(row.get("timestamp") or ""), str(row.get("text") or ""),
        )
        if duplicate_key in seen_posts:
            continue
        seen_posts.add(duplicate_key)
        user_id = row["user_id"]
        timestamp_ms = parse_comment_date(row.get("timestamp"))
        if timestamp_ms is None:
            continue
        item = aggregate.setdefault(user_id, {
            "count": 0,
            "text_chars": 0,
            "total_before_ad_filter": 0,
            "advertising_filtered_count": 0,
            "min_ts": None,
            "max_ts": None,
        })
        text = str(row.get("text") or "")
        item["total_before_ad_filter"] += 1
        if is_advertising(text):
            item["advertising_filtered_count"] += 1
            continue
        item["count"] += 1
        item["text_chars"] += len(text)
        item["min_ts"] = timestamp_ms if item["min_ts"] is None else min(item["min_ts"], timestamp_ms)
        item["max_ts"] = timestamp_ms if item["max_ts"] is None else max(item["max_ts"], timestamp_ms)

    eligible = []
    for user_id, item in aggregate.items():
        if item["min_ts"] is None or item["max_ts"] is None:
            continue
        history_days = (item["max_ts"] - item["min_ts"]) / 86_400_000
        advertising_ratio = (
            item["advertising_filtered_count"] / item["total_before_ad_filter"]
            if item["total_before_ad_filter"] else 0.0
        )
        score_factor = advertising_score_factor(advertising_ratio)
        if (item["count"] >= config.min_posts
                and item["text_chars"] >= config.min_text_chars
                and history_days >= config.min_history_days
                and score_factor > 0):
            eligible.append((
                user_id,
                item["count"],
                item["text_chars"],
                history_days,
                item["advertising_filtered_count"],
                item["total_before_ad_filter"],
                advertising_ratio,
                score_factor,
            ))
    return eligible


def select_users(config: Config) -> None:
    shard_files = list(iter_post_shards(config))
    if not shard_files:
        raise FileNotFoundError(f"No post shards in {config.post_shards_dir}. Run ingest first.")
    eligible: list[tuple] = []
    with ProcessPoolExecutor(
        max_workers=config.preprocess_workers,
        initializer=_init_config_worker,
        initargs=(config,),
    ) as executor:
        for shard_eligible in tqdm(
            executor.map(_select_users_shard, shard_files),
            total=len(shard_files), desc="select users",
        ):
            eligible.extend(shard_eligible)
    scored = [(
        user_selection_score(row[1], row[2], row[3], config) * row[7],
        row,
    ) for row in eligible]
    scored.sort(key=lambda item: (-item[0], -item[1][2], -item[1][3], -item[1][1], item[1][0]))
    with config.selected_users_path.open("w", encoding="utf-8") as output:
        for rank, (score, row) in enumerate(scored[:config.top_k], 1):
            keys = (
                "user_id",
                "post_count",
                "text_chars",
                "history_days",
                "advertising_filtered_count",
                "total_posts_before_ad_filter",
                "advertising_ratio",
                "advertising_score_factor",
            )
            record = {key: value for key, value in zip(keys, row)}
            record.update({"rank": rank, "score": score})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Eligible users={len(eligible):,}; selected={min(config.top_k, len(scored)):,}")


def _prepare_shard(path: Path) -> int:
    config = _worker_config
    selected = _worker_selected
    profiles = _worker_profiles
    shard_index = int(path.stem.rsplit("-", 1)[-1])
    output_path = history_shard_path(config, shard_index)
    posts_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in iter_local_jsonl(path):
        user_id = str(post["user_id"])
        if user_id in selected:
            posts_by_user[user_id].append(post)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for user_id, raw_posts in posts_by_user.items():
            posts = filter_posts(
                raw_posts, min_post_text_chars=config.min_post_text_chars
            )
            posts = select_contents(
                posts,
                max_contents=config.max_contents,
                timeline_ratio=config.timeline_content_ratio,
            )
            if len(posts) < 2:
                continue
            for post in posts:
                post.pop("timestamp_ms", None)
                post.pop("source_index", None)
            record = {"source": "facebook", "user_id": user_id,
                "rank": selected[user_id]["rank"], "profile": profiles.get(user_id, {}),
                "post_count": len(posts), "posts": posts}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_histories(config: Config) -> None:
    require_file(config.selected_users_path, "Run the prepare selection after ingest")
    require_file(config.user_profiles_path, "Run ingest first")
    with config.selected_users_path.open(encoding="utf-8") as source:
        selected_records = [json.loads(line) for line in source if line.strip()]
    selected = {str(row["user_id"]): row for row in selected_records}
    profiles = {
        str(row["id"]): row for row in iter_local_jsonl(config.user_profiles_path)
        if str(row.get("id") or "") in selected
    }
    shard_files = list(iter_post_shards(config))
    config.history_shards_dir.mkdir(parents=True, exist_ok=True)
    total_histories = 0
    with ProcessPoolExecutor(
        max_workers=config.preprocess_workers,
        initializer=_init_prepare_worker,
        initargs=(config, selected, profiles),
    ) as executor:
        for shard_count in tqdm(
            executor.map(_prepare_shard, shard_files),
            total=len(shard_files), desc="prepare histories",
        ):
            total_histories += shard_count
    print(f"Prepared histories={total_histories:,} in {config.history_shards_dir}")


def _compact_shard(path: Path) -> int:
    config = _worker_config
    shard_index = int(path.stem.rsplit("-", 1)[-1])
    output_path = compact_shard_path(config, shard_index)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for user in iter_local_jsonl(path):
            profile = assemble_profile(user, config.max_profile_chars)
            record = {key: user[key] for key in ("user_id", "source", "post_count")}
            record.update({"compact_profile_chars": len(profile), "max_profile_chars": config.max_profile_chars,
                           "profile_text": profile})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def compact_profiles(config: Config) -> None:
    history_shards = list(iter_history_shards(config))
    if not history_shards:
        raise FileNotFoundError(f"No history shards in {config.history_shards_dir}. Run prepare first.")
    config.compact_shards_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with ProcessPoolExecutor(
        max_workers=config.preprocess_workers,
        initializer=_init_config_worker,
        initargs=(config,),
    ) as executor:
        for shard_count in tqdm(
            executor.map(_compact_shard, history_shards),
            total=len(history_shards), desc="compact history shards",
        ):
            count += shard_count
    print(f"Compact profiles={count:,} in {config.compact_shards_dir}")


def load_schema(config: Config) -> list[dict[str, Any]]:
    require_file(config.schema_path, "Provide --schema-path")
    document = json.loads(config.schema_path.read_text(encoding="utf-8"))
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"Invalid schema: {config.schema_path}")
    return dimensions


def call_llm(prompt: str, config: Config) -> str:
    return complete_prompt(prompt, LLMSettings(
        provider=config.llm_provider,
        local_endpoint=config.llm_endpoint,
        local_model=config.model,
        local_authorization=config.llm_authorization,
        openrouter_api_key=config.openrouter_api_key,
        openrouter_model=config.openrouter_model,
        timeout_seconds=config.llm_timeout_seconds,
    ))


def _timed_llm_call(prompt: str, config: Config) -> tuple[float, str | None]:
    started_at = time.perf_counter()
    try:
        call_llm(prompt, config)
        return time.perf_counter() - started_at, None
    except (LLMUnauthorizedError, LLMCancelledError):
        raise
    except Exception as error:
        return time.perf_counter() - started_at, str(error)


def benchmark_llm_concurrency(config: Config) -> None:
    """Probe how many concurrent requests the LLM endpoint sustains.

    Replays the real extraction prompt (schema chunk + an actual compact profile when
    one is available) at increasing concurrency levels, so ``llm_workers`` can be sized
    from measured latency/error behavior instead of guesswork.
    """
    if config.llm_bench_prompt_index:
        prompt_path = config.prompt_log_dir / f"prompt-{config.llm_bench_prompt_index:04d}.txt"
        require_file(prompt_path, "Run extract first so prompt_log has files, or pick a valid index")
        prompt = prompt_path.read_text(encoding="utf-8")
        print(f"Benchmarking {config.llm_endpoint} with {prompt_path.name} ({len(prompt):,} chars)")
    else:
        schema = load_schema(config)
        chunks = cat_chunks(_group_by_category(schema), config.max_dims_per_chunk)
        if not chunks:
            raise ValueError(f"Schema at {config.schema_path} produced no dimension chunks")

        profile_text = next(
            (record["profile_text"] for record in iter_compact_profiles(config)), None
        )
        if profile_text is None:
            profile_text = "Sample profile text. " * (config.max_profile_chars // 21)
            print("No compact profiles found; benchmarking with a synthetic profile instead.")

        prompt = build_post_prompt(profile_text, chunks[0])
        print(f"Benchmarking {config.llm_endpoint} with a {len(prompt):,}-char prompt")

    results: list[dict[str, Any]] = []
    for concurrency in config.llm_bench_concurrency_levels:
        total_requests = concurrency * config.llm_bench_requests_per_worker
        latencies: list[float] = []
        errors = 0
        wall_started_at = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [
                executor.submit(_timed_llm_call, prompt, config)
                for _ in range(total_requests)
            ]
            for future in as_completed(futures):
                elapsed_seconds, error = future.result()
                if error is not None:
                    errors += 1
                    tqdm.write(f"concurrency={concurrency} error={error}")
                else:
                    latencies.append(elapsed_seconds)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        wall_elapsed_seconds = time.perf_counter() - wall_started_at
        ok = len(latencies)
        row = {
            "concurrency": concurrency,
            "requests": total_requests,
            "ok": ok,
            "errors": errors,
            "avg_latency_seconds": round(sum(latencies) / ok, 2) if ok else None,
            "max_latency_seconds": round(max(latencies), 2) if ok else None,
            "throughput_requests_per_second": round(ok / wall_elapsed_seconds, 3)
            if wall_elapsed_seconds else 0.0,
        }
        results.append(row)
        print(
            f"concurrency={concurrency:>3} ok={ok}/{total_requests} "
            f"avg={row['avg_latency_seconds']}s max={row['max_latency_seconds']}s "
            f"throughput={row['throughput_requests_per_second']}req/s"
        )
        if errors * 2 > total_requests:
            print(f"Stopping: concurrency={concurrency} failed more than half its requests")
            break

    config.llm_bench_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    baseline_latency = results[0]["avg_latency_seconds"] if results else None
    recommended = results[0]["concurrency"] if results else 1
    for row in results:
        latency_ok = (
            baseline_latency is None
            or row["avg_latency_seconds"] is None
            or row["avg_latency_seconds"] <= baseline_latency * 2.5
        )
        if row["errors"] == 0 and latency_ok:
            recommended = row["concurrency"]
    print(f"Recommended llm_workers: {recommended}")
    print("Benchmark results:", config.llm_bench_path)


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


def _group_by_category(dimensions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dimension in dimensions:
        grouped[dimension.get("category", "Uncategorized")].append(dimension)
    return grouped


def _run_dimension_chunks(
    user_id: str,
    profile_text: str,
    chunks: list[list[dict[str, Any]]],
    config: Config,
) -> list[dict[str, Any]]:
    """Call the LLM for every dimension chunk of one user; return flat sanitized fields."""
    reset_prompt_log(config)
    chunk_results: dict[int, list[dict[str, Any]]] = {}
    futures = {}
    executor = ThreadPoolExecutor(max_workers=config.llm_workers)
    for chunk_index, dimensions in enumerate(chunks, start=1):
        prompt = build_post_prompt(profile_text, dimensions)
        prompt_path = save_prompt_log(config, chunk_index, prompt)
        future = executor.submit(call_llm, prompt, config)
        futures[future] = (chunk_index, dimensions, prompt_path, time.perf_counter())
    try:
        for future in as_completed(futures):
            chunk_index, dimensions, prompt_path, started_at = futures[future]
            try:
                response = future.result()
                chunk_fields = sanitize_fields(
                    parse_fields(response), dimensions, profile_text
                )
            except (LLMUnauthorizedError, LLMCancelledError):
                for pending in futures:
                    pending.cancel()
                raise
            except Exception as error:
                chunk_fields = sanitize_fields([], dimensions, profile_text)
                tqdm.write(
                    f"user={user_id} chunk={chunk_index}/{len(chunks)} "
                    f"LLM retries exhausted; marking {len(dimensions)} dimensions "
                    f"unsupported; error={error}"
                )
            chunk_results[chunk_index] = chunk_fields
            categories = sorted({
                str(dimension.get("category") or "Uncategorized")
                for dimension in dimensions
            })
            supported_count = sum(
                field["value"] is not None for field in chunk_fields
            )
            elapsed_seconds = time.perf_counter() - started_at
            tqdm.write(
                f"user={user_id} chunk={chunk_index}/{len(chunks)} "
                f"category={','.join(categories)} dimensions={len(dimensions)} "
                f"supported={supported_count} elapsed={elapsed_seconds:.2f}s "
                f"prompt_log={prompt_path.name}"
            )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [
        field
        for chunk_index in range(1, len(chunks) + 1)
        for field in chunk_results[chunk_index]
    ]


def extract_personas(config: Config) -> None:
    compact_shards = list(iter_compact_shards(config))
    if not compact_shards:
        raise FileNotFoundError(f"No compact shards in {config.compact_shards_dir}. Run compact first.")
    schema = load_schema(config)
    chunks = cat_chunks(_group_by_category(schema), config.max_dims_per_chunk)
    done = set()
    if config.personas_path.exists():
        with config.personas_path.open(encoding="utf-8") as existing:
            done = {str(json.loads(line)["user_id"]) for line in existing if line.strip()}
    processed = 0
    with config.personas_path.open("a", encoding="utf-8") as output:
        for record in tqdm(iter_compact_profiles(config), desc="extract personas"):
            user_id = str(record["user_id"])
            if user_id in done:
                continue
            if config.max_llm_users and processed >= config.max_llm_users:
                break
            fields = _run_dimension_chunks(user_id, record["profile_text"], chunks, config)
            if len(fields) != len(schema):
                raise RuntimeError(f"Expected {len(schema)} fields, got {len(fields)} for {user_id}")
            result = {key: record[key] for key in (
                "user_id", "source", "post_count", "compact_profile_chars"
            )}
            result["fields"] = fields
            output.write(json.dumps(result, ensure_ascii=False) + "\n"); output.flush(); os.fsync(output.fileno())
            done.add(user_id); processed += 1
    print(f"New personas={processed}; output={config.personas_path}")


def _legacy_schema_dimensions(path: Path) -> list[dict[str, Any]]:
    require_file(path, "Check Config.legacy_schema_paths")
    document = json.loads(path.read_text(encoding="utf-8"))
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"Invalid schema: {path}")
    return dimensions


def _normalized_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _build_legacy_migration(
    final_schema: list[dict[str, Any]], legacy_dimensions: list[dict[str, Any]]
) -> dict[str, tuple[str, list[str], list[str]]]:
    """Map final_id -> (legacy_id, legacy_values, final_values) for every dimension that
    still has a safe source: the same id, or -- when the id itself was renamed -- a
    uniquely matching label against the legacy schema's own English ``label`` (matched
    against ``final``'s English ``dimension`` name, since ``final``'s own ``label`` is
    Vietnamese). Either way, only when the allowed-value count is unchanged, so an old
    value can be translated positionally.
    """
    legacy_by_id = {str(d["id"]): d for d in legacy_dimensions}
    legacy_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dimension in legacy_dimensions:
        legacy_by_label[_normalized_label(dimension.get("label"))].append(dimension)

    migration: dict[str, tuple[str, list[str], list[str]]] = {}
    for dimension in final_schema:
        final_id = str(dimension["id"])
        legacy = legacy_by_id.get(final_id)
        if legacy is None:
            candidates = legacy_by_label.get(_normalized_label(dimension.get("dimension")), [])
            legacy = candidates[0] if len(candidates) == 1 else None
        if legacy is None:
            continue
        legacy_values = [str(v) for v in legacy.get("values", [])]
        final_values = [str(v) for v in dimension.get("values", [])]
        if legacy_values and len(legacy_values) == len(final_values):
            migration[final_id] = (str(legacy["id"]), legacy_values, final_values)
    return migration


def _detect_legacy_format(field_ids: set[str], distinctive_ids_by_format: list[set[str]]) -> int | None:
    """Return the legacy schema index whose distinctive ids this persona's fields match
    best, or None when it matches none (already on the final schema, or unrecognized)."""
    best_index, best_hits = None, 0
    for index, distinctive_ids in enumerate(distinctive_ids_by_format):
        hits = len(field_ids & distinctive_ids)
        if hits > best_hits:
            best_index, best_hits = index, hits
    return best_index


def migrate_personas(config: Config) -> None:
    """Carry personas extracted under an older schema onto the current (final) schema.

    A dimension whose id and allowed-value count are unchanged is translated in place,
    reusing the old evidence/confidence/description and only mapping ``value`` onto its
    new-schema wording. Every other dimension has no safe old counterpart (new dimension,
    or its allowed values were restructured) and is re-extracted via the LLM, exactly like
    the extract stage but scoped to only that missing subset.
    """
    require_file(config.personas_path, "Run the extract stage first")
    compact_shards = list(iter_compact_shards(config))
    if not compact_shards:
        raise FileNotFoundError(f"No compact shards in {config.compact_shards_dir}. Run compact first.")

    final_schema = load_schema(config)
    final_ids = {str(d["id"]) for d in final_schema}
    legacy_dimensions_by_format = [
        _legacy_schema_dimensions(path) for path in config.legacy_schema_paths
    ]
    legacy_ids_by_format = [{str(d["id"]) for d in dims} for dims in legacy_dimensions_by_format]
    # Ids this legacy format uses that neither another legacy format nor the final schema
    # itself also uses -- i.e. a reliable fingerprint that a persona's fields predate the
    # final schema. Excluding final's own ids matters because the final schema reuses many
    # of dimensions.json's original ids verbatim, so an already-final persona would
    # otherwise look like it matches that legacy format too.
    distinctive_ids_by_format = [
        ids - final_ids - set().union(*(other for j, other in enumerate(legacy_ids_by_format) if j != i), set())
        for i, ids in enumerate(legacy_ids_by_format)
    ]
    migration_by_format = [
        _build_legacy_migration(final_schema, dims) for dims in legacy_dimensions_by_format
    ]
    missing_chunks_by_format = [
        cat_chunks(
            _group_by_category([d for d in final_schema if str(d["id"]) not in migration]),
            config.max_dims_per_chunk,
        )
        for migration in migration_by_format
    ]
    for index, migration in enumerate(migration_by_format):
        migratable = len(migration)
        print(
            f"Legacy format #{index} ({config.legacy_schema_paths[index].name}): "
            f"{migratable}/{len(final_schema)} dimensions migratable, "
            f"{len(final_schema) - migratable} need fresh LLM extraction"
        )

    old_personas: dict[str, dict[str, Any]] = {}
    with config.personas_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                persona = json.loads(line)
                old_personas[str(persona["user_id"])] = persona

    done = set()
    if config.migrated_personas_path.exists():
        with config.migrated_personas_path.open(encoding="utf-8") as existing:
            done = {str(json.loads(line)["user_id"]) for line in existing if line.strip()}

    migrated = 0
    skipped_no_old_persona = 0
    with config.migrated_personas_path.open("a", encoding="utf-8") as output:
        for record in tqdm(iter_compact_profiles(config), desc="migrate personas"):
            user_id = str(record["user_id"])
            if user_id in done:
                continue
            old_persona = old_personas.get(user_id)
            if old_persona is None:
                skipped_no_old_persona += 1
                continue
            old_fields_by_id = {
                str(field.get("field_id")): field
                for field in old_persona.get("fields", [])
                if isinstance(field, dict)
            }
            field_ids = set(old_fields_by_id)
            format_index = _detect_legacy_format(field_ids, distinctive_ids_by_format)
            if format_index is None:
                # Not a recognized legacy format: pass through any field whose id already
                # matches the final schema verbatim (e.g. already-migrated personas).
                migration = {
                    str(d["id"]): (str(d["id"]), [], [])
                    for d in final_schema if str(d["id"]) in field_ids
                }
                missing_chunks = cat_chunks(
                    _group_by_category([d for d in final_schema if str(d["id"]) not in migration]),
                    config.max_dims_per_chunk,
                )
            else:
                migration = migration_by_format[format_index]
                missing_chunks = missing_chunks_by_format[format_index]

            migrated_fields: dict[str, dict[str, Any]] = {}
            for final_id, (legacy_id, legacy_values, final_values) in migration.items():
                old_field = old_fields_by_id.get(legacy_id)
                if old_field is None:
                    continue
                old_value = old_field.get("value")
                if old_value is None or not legacy_values:
                    new_value = old_value
                else:
                    try:
                        new_value = final_values[legacy_values.index(str(old_value))]
                    except ValueError:
                        new_value = None
                migrated_fields[final_id] = {
                    "field_id": final_id,
                    "value": new_value,
                    "confidence": old_field.get("confidence", 0.0),
                    "evidence": old_field.get("evidence", ""),
                    "description": old_field.get("description", ""),
                    "assignment_type": old_field.get("assignment_type", "unsupported"),
                }

            if missing_chunks:
                fresh_fields = _run_dimension_chunks(
                    user_id, record["profile_text"], missing_chunks, config
                )
                for field in fresh_fields:
                    migrated_fields[str(field["field_id"])] = field

            fields = [
                migrated_fields.get(str(d["id"])) or unsupported_field(d)
                for d in final_schema
            ]
            result = {key: record[key] for key in (
                "user_id", "source", "post_count", "compact_profile_chars"
            )}
            result["fields"] = fields
            output.write(json.dumps(result, ensure_ascii=False) + "\n"); output.flush(); os.fsync(output.fileno())
            done.add(user_id); migrated += 1
    print(
        f"Migrated personas={migrated}; skipped (no old persona)={skipped_no_old_persona}; "
        f"output={config.migrated_personas_path}"
    )


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
    supported_dimensions_by_user: dict[str, int] = {}
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
            supported_dimensions_by_user[user_id] = 0
            fields = persona.get("fields") or []
            for field in fields:
                if not isinstance(field, dict) or field.get("value") is None:
                    continue
                supported_dimensions_total += 1
                supported_dimensions_by_user[user_id] += 1
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
        "top_100_users_by_supported_dimension_count": [
            {"user_id": user_id, "supported_dimension_count": count}
            for user_id, count in sorted(
                supported_dimensions_by_user.items(),
                key=lambda item: (-item[1], item[0]),
            )[:100]
        ],
        "bottom_100_users_by_supported_dimension_count": [
            {"user_id": user_id, "supported_dimension_count": count}
            for user_id, count in sorted(
                supported_dimensions_by_user.items(),
                key=lambda item: (item[1], item[0]),
            )[:100]
        ],
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
        f"Category Coverage Analysis (Vietnamese Personas: {persona_count})",
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


