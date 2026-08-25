"""Build schema-constrained personas from Facebook profiles and content.

The pipeline is split into five resumable stages: ingest, prepare, compact,
extract and stats. Run ``python facebook_extractor.py --help`` for usage.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm
from llm_client import LLMCancelledError, LLMSettings, LLMUnauthorizedError, complete_prompt
from persona_coverage_chart import render_category_coverage_chart

PROJECT_ROOT = Path(__file__).resolve().parent
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
csv.field_size_limit(100_000_000)


def project_path(value: Path) -> Path:
    """Resolve relative paths from the project directory, not the current shell."""
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Config:
    content_dir: Path
    user_dir: Path
    work_dir: Path
    schema_path: Path
    max_rows_per_file: int = 0
    top_k: int = 50000
    min_posts: int = 5
    min_text_chars: int = 1000
    min_post_text_chars: int = 20
    min_history_days: float = 30
    max_profile_chars: int = 35_000
    max_post_text_chars: int = 2_000
    max_dims_per_chunk: int = 50
    max_llm_users: int = 0
    post_shards: int = 50
    llm_provider: str = os.environ.get("LLM_PROVIDER", "local")
    model: str = os.environ.get("LLM_MODEL", "Qwen3-14B")
    llm_endpoint: str = os.environ.get(
        "LLM_ENDPOINT",
        "http://203.113.152.4:7777/llm/v1/chat/completions",
    )
    llm_authorization: str = os.environ.get("LLM_AUTHORIZATION", "")
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    llm_timeout_seconds: int = 300

    @property
    def post_shards_dir(self) -> Path:
        return self.work_dir / "post_shards"

    @property
    def selected_users_path(self) -> Path:
        return self.work_dir / "selected_users.jsonl"

    @property
    def user_profiles_path(self) -> Path:
        return self.work_dir / "user_profiles.jsonl"

    @property
    def history_shards_dir(self) -> Path:
        return self.work_dir / "history_shards"

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

ASSIGNMENT_TYPES = {"direct", "structured_claim", "summary_inference", "unsupported"}
NULLISH_VALUES = {"", "null", "none", "n/a", "na", "unknown", "unsupported", "not applicable"}
IB_PATTERN = re.compile(r"(?i)(?<!\w)ib(?!\w)")


def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars - 15].rstrip() + " ... [truncated]"
    return text

def parse_comment_date(value: Any) -> int | None:
    """Parse ISO-8601 and Facebook ``YYYY/MM/DD HH:MM:SS`` timestamps."""
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", text):
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=VIETNAM_TIMEZONE)
    return int(parsed.timestamp() * 1000)


def format_vietnam_datetime(timestamp_ms: int) -> str:
    """Format epoch milliseconds as ISO-8601 in Vietnam time (UTC+07:00)."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=VIETNAM_TIMEZONE).isoformat(timespec="seconds")

def post_text(post: dict[str, Any]) -> str:
    return compact_text(post.get("text"))


def normalize_post_record(post: dict[str, Any]) -> dict[str, Any]:
    row = dict(post)
    timestamp_ms = row.get("timestamp_ms")
    if not isinstance(timestamp_ms, int):
        timestamp_ms = parse_comment_date(row.get("timestamp") or row.get("comment_date"))
    row["timestamp_ms"] = timestamp_ms
    row["timestamp"] = format_vietnam_datetime(timestamp_ms) if timestamp_ms is not None else ""
    row["text"] = post_text(row)
    return row

def post_key(post: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(post.get("post_id") or ""), str(post.get("text_hash") or ""),
        str(post.get("thread_id") or ""), str(post.get("timestamp_ms") or ""),
        post_text(post).lower(),
    )

def filter_posts(posts: list[dict[str, Any]], *, min_post_text_chars: int) -> list[dict[str, Any]]:
    kept, seen = [], set()
    for raw_post in posts:
        post = normalize_post_record(raw_post)
        if post.get("timestamp_ms") is None:
            continue
        elif IB_PATTERN.search(post_text(post)):
            continue
        elif len(post_text(post)) < min_post_text_chars:
            continue
        elif post_key(post) in seen:
            continue
        seen.add(post_key(post))
        kept.append(post)
    kept.sort(key=lambda row: (int(row.get("timestamp_ms") or 0), int(row.get("source_index") or 0)))
    return kept

def render_post(post: dict[str, Any], index: int, max_post_text_chars: int) -> str:
    lines = [
        f"[content {index}]", f"timestamp: {post.get('timestamp') or 'unknown'}",
        f"type: {post.get('category') or 'unknown'}",
    ]
    lines.append(f"text: {compact_text(post_text(post), max_post_text_chars)}")
    return "\n".join(lines)

def assemble_profile(row: dict[str, Any], max_chars: int, max_post_text_chars: int) -> str:
    posts = row.get("posts") or []
    profile = row.get("profile") or {}
    parts = ["Vietnamese Facebook user profile."]
    profile_lines = []
    for key in ("name", "about", "gender", "dateOfBirth", "birthday", "age_range",
                "hometown", "relationship_status", "religion", "education", "work",
                "isVietnamese"):
        value = profile.get(key)
        if value not in (None, "", []):
            profile_lines.append(f"{key}: {compact_text(value, 3000)}")
    if profile_lines:
        parts.append("[declared profile]\n" + "\n".join(profile_lines))
    parts.extend(render_post(post, index, max_post_text_chars) for index, post in enumerate(posts, 1))
    return "\n\n".join(parts)[:max_chars]

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


def ingest_posts(config: Config) -> None:
    user_files = local_json_files(config.user_dir)
    content_files = local_json_files(config.content_dir)
    if not user_files:
        raise FileNotFoundError(f"No .jsonl/.json files found in {config.user_dir}")
    if not content_files:
        raise FileNotFoundError(f"No .jsonl/.json files found in {config.content_dir}")
    config.post_shards_dir.mkdir(parents=True, exist_ok=True)
    target_user_ids: set[str] = set()
    excluded_profile_fields = {"phone", "number_follow", "number_friend", "averageReact", "hobby"}
    with config.user_profiles_path.open("w", encoding="utf-8") as output:
        for path in user_files:
            scanned = kept = 0
            for row in tqdm(iter_json_records(path), desc=f"users:{path.name}"):
                scanned += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                user_id = compact_text(row.get("id"))
                if not user_id or user_id in target_user_ids:
                    continue
                target_user_ids.add(user_id)
                profile = {key: value for key, value in row.items() if key not in excluded_profile_fields}
                profile["id"] = user_id
                output.write(json.dumps(profile, ensure_ascii=False) + "\n")
                kept += 1
            print(f"{path.name}: profiles scanned={scanned:,}, kept={kept:,}")
    handles = [shard_path(config, index).open("w", encoding="utf-8") for index in range(config.post_shards)]
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
                if user_id not in target_user_ids or timestamp is None:
                    continue
                category = compact_text(row.get("article_type") or "facebook_content")
                text = str(row.get("content") or "")
                record = {
                    "user_id": user_id,
                    "category": category,
                    "source": "facebook",
                    "text": text,
                    "timestamp": format_vietnam_datetime(timestamp),
                }
                handle = handles[user_shard(user_id, config.post_shards)]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1; total_kept += 1
            print(f"{path.name}: scanned={scanned:,}, kept={kept:,}")
    finally:
        for handle in handles:
            handle.close()
    print(f"Target profiles={len(target_user_ids):,}; joined content={total_kept:,} "
          f"in {config.post_shards} JSONL shards")


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
    if not any(iter_post_shards(config)):
        raise FileNotFoundError(f"No post shards in {config.post_shards_dir}. Run ingest first.")
    aggregate: dict[str, dict[str, Any]] = {}
    for path in tqdm(list(iter_post_shards(config)), desc="select users"):
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
            item = aggregate.setdefault(user_id, {"count": 0, "text_posts": 0,
                "text_chars": 0, "min_ts": timestamp_ms, "max_ts": timestamp_ms})
            text = str(row.get("text") or "")
            if IB_PATTERN.search(text):
                continue
            item["count"] += 1
            item["text_posts"] += int(bool(text.strip())); item["text_chars"] += len(text)
            item["min_ts"] = min(item["min_ts"], timestamp_ms)
            item["max_ts"] = max(item["max_ts"], timestamp_ms)
    eligible = []
    for user_id, item in aggregate.items():
        history_days = (item["max_ts"] - item["min_ts"]) / 86_400_000
        if (item["count"] >= config.min_posts
                and item["text_chars"] >= config.min_text_chars
                and history_days >= config.min_history_days):
            eligible.append((user_id, item["count"], item["text_posts"], item["text_chars"],
                             history_days))
    metric_columns = (3, 2, 4, 1)
    weights = (0.4375, 0.25, 0.1875, 0.125)
    ranks = [percentile_ranks([float(row[column] or 0) for row in eligible]) for column in metric_columns]
    scored = [(sum(weight * vector[index] for weight, vector in zip(weights, ranks)), row)
              for index, row in enumerate(eligible)]
    scored.sort(key=lambda item: (-item[0], -item[1][3], -item[1][2], -item[1][1], item[1][0]))
    with config.selected_users_path.open("w", encoding="utf-8") as output:
        for rank, (score, row) in enumerate(scored[:config.top_k], 1):
            keys = ("user_id", "post_count", "text_posts", "text_chars", "history_days")
            record = {key: value for key, value in zip(keys, row)}
            record.update({"rank": rank, "score": score})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Eligible users={len(eligible):,}; selected={min(config.top_k, len(scored)):,}")


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
    for path in tqdm(shard_files, desc="prepare histories"):
        shard_index = int(path.stem.rsplit("-", 1)[-1])
        output_path = history_shard_path(config, shard_index)
        with output_path.open("w", encoding="utf-8") as output:
            posts_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for post in iter_local_jsonl(path):
                user_id = str(post["user_id"])
                if user_id in selected:
                    posts_by_user[user_id].append(post)
            for user_id, raw_posts in posts_by_user.items():
                posts = filter_posts(
                    raw_posts, min_post_text_chars=config.min_post_text_chars
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
                total_histories += 1
    print(f"Prepared histories={total_histories:,} in {config.history_shards_dir}")


def compact_profiles(config: Config) -> None:
    history_shards = list(iter_history_shards(config))
    if not history_shards:
        raise FileNotFoundError(f"No history shards in {config.history_shards_dir}. Run prepare first.")
    count = 0
    with config.compact_profiles_path.open("w", encoding="utf-8") as output:
        for path in tqdm(history_shards, desc="compact history shards"):
            for user in iter_local_jsonl(path):
                profile = assemble_profile(user, config.max_profile_chars, config.max_post_text_chars)
                record = {key: user[key] for key in ("user_id", "source", "post_count")}
                record.update({"compact_profile_chars": len(profile), "max_profile_chars": config.max_profile_chars,
                               "max_post_text_chars": config.max_post_text_chars, "profile_text": profile})
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
    return complete_prompt(prompt, LLMSettings(
        provider=config.llm_provider,
        local_endpoint=config.llm_endpoint,
        local_model=config.model,
        local_authorization=config.llm_authorization,
        openrouter_api_key=config.openrouter_api_key,
        openrouter_model=config.openrouter_model,
        timeout_seconds=config.llm_timeout_seconds,
    ))


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
                prompt = build_post_prompt(record["profile_text"], dimensions)
                prompt_path = save_prompt_log(config, chunk_index, prompt)
                try:
                    response = call_llm(prompt, config)
                    chunk_fields = sanitize_fields(parse_fields(response), dimensions, record["profile_text"])
                except (LLMUnauthorizedError, LLMCancelledError):
                    raise
                except Exception as error:
                    chunk_fields = sanitize_fields([], dimensions, record["profile_text"])
                    tqdm.write(
                        f"user={user_id} chunk={chunk_index}/{len(chunks)} "
                        f"LLM retries exhausted; marking {len(dimensions)} dimensions "
                        f"unsupported; error={error}"
                    )
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
            result = {key: record[key] for key in (
                "user_id", "source", "post_count", "compact_profile_chars"
            )}
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
            fields = persona.get("fields") or []
            for field in fields:
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "ingest", "prepare", "compact", "extract", "stats"), nargs="?", default="all")
    parser.add_argument("--content-dir", type=Path, default=Path("data/facebook/content"))
    parser.add_argument("--user-dir", type=Path, default=Path("data/facebook/user"))
    parser.add_argument("--work-dir", type=Path, default=Path("facebook_persona_fresh"))
    parser.add_argument("--schema-path", type=Path, default=Path("schema/dimension.json"))
    parser.add_argument("--top-k", type=int, default=50000)
    parser.add_argument("--min-history-days", type=float, default=30)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-llm-users", type=int, default=0)
    parser.add_argument("--post-shards", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(content_dir=project_path(args.content_dir), user_dir=project_path(args.user_dir),
                    work_dir=project_path(args.work_dir),
                    schema_path=project_path(args.schema_path),
                    top_k=args.top_k, max_rows_per_file=args.max_rows_per_file,
                    min_history_days=args.min_history_days,
                    max_llm_users=args.max_llm_users, post_shards=args.post_shards)
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


if __name__ == "__main__":
    main()
