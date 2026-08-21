"""Build schema-constrained personas from Vietnamese VOZ forum posts.

The pipeline is split into five resumable stages: ingest, prepare, compact,
extract and stats. Run ``python voz_extractor.py --help`` for usage.
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
from llm_client import LLMSettings, complete_prompt
from persona_coverage_chart import render_category_coverage_chart

PROJECT_ROOT = Path(__file__).resolve().parent
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
csv.field_size_limit(100_000_000)


def project_path(value: Path) -> Path:
    """Resolve relative paths from the project directory, not the current shell."""
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Config:
    post_dir: Path
    work_dir: Path
    schema_path: Path
    max_rows_per_file: int = 0
    top_k: int = 100
    min_posts: int = 5
    min_text_chars: int = 1000
    min_post_text_chars: int = 20
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

ASSIGNMENT_TYPES = {"direct", "structured_claim", "summary_inference", "unsupported"}
NULLISH_VALUES = {"", "null", "none", "n/a", "na", "unknown", "unsupported", "not applicable"}
def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars - 15].rstrip() + " ... [truncated]"
    return text

def parse_comment_date(value: Any) -> int | None:
    """Parse ISO-8601 timestamps, including VOZ offsets such as +0700."""
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", text):
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
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
        f"[post {index}]", f"timestamp: {post.get('timestamp') or 'unknown'}",
        f"thread_id: {post.get('thread_id') or 'unknown'}",
    ]
    lines.append(f"text: {compact_text(post_text(post), max_post_text_chars)}")
    return "\n".join(lines)

def assemble_profile(row: dict[str, Any], max_chars: int, max_post_text_chars: int) -> str:
    posts = row.get("posts") or []
    parts = ["Vietnamese VOZ forum member profile."]
    parts.extend(render_post(post, index, max_post_text_chars) for index, post in enumerate(posts, 1))
    return "\n\n".join(parts)[:max_chars]

def build_post_prompt(profile_text: str, dimensions: list[dict[str, Any]]) -> str:
    """Build a schema-constrained prompt for Vietnamese VOZ forum posts."""
    lines = [
        "You are mapping observable Vietnamese VOZ forum-post evidence to schema-constrained "
        "persona fields for one forum member. Fill attributes that are well supported "
        "by the posting history, and leave unsupported or identity-like claims null.",
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
        '"evidence": "<one short exact quote copied from POSTING HISTORY, or empty string>", '
        '"description": "<1-2 concrete sentences, or empty string>", '
        '"assignment_type": "direct|structured_claim|summary_inference|unsupported"}]}',
        "",
        "Allowed support:",
        "- direct: use when the member explicitly states the fact about themselves in post text.",
        "- structured_claim: use for repeated concrete non-sensitive claims supported by at least 2 distinct posts or threads.",
        "- summary_inference: use for non-sensitive interests, participation behavior, communication style, or expertise when a repeated pattern is visible across the posting history.",
        "- Overall writing style may support communication/cognitive-style "
        "dimensions only when the pattern is visible across at least 5 posts.",
        "- unsupported: use when evidence is absent, one-off, ambiguous, generic, "
        "or mainly about someone other than the forum member.",
        "",
        "Hard limits:",
        "- For age, gender, health, disability, ethnicity, religion, politics, "
        "income, family/household status, occupation, location, employment, and "
        "parenthood: assign a non-null value only from an explicit self-statement. "
        "Do not infer them from thread topic, quoted news, or other participants.",
        "- Do not attribute claims from quoted material or another participant to the member.",
        "- Do not infer personality inventories, values, worldview, MBTI, Big "
        "Five, HEXACO, clinical attributes, or mental-state attributes from "
        "ordinary forum posts unless the member explicitly states the "
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
        "verbatim from one of the posts.",
        "- Evidence must be an exact quote from POSTING HISTORY, not your reasoning, "
        "a paraphrase, or a summary. If you cannot copy an exact quote, return "
        "unsupported.",
        "- If you cannot copy an exact quote, return unsupported.",
        "- Do not append support counts, explanations, or labels to evidence. "
        "Evidence must be only text that appears in POSTING HISTORY.",
        "- description: 1-2 concrete Vietnamese sentences describing THIS forum member using details from their posts and threads. Describe the person; do not justify the label.",
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
        
    lines += ["", "POSTING HISTORY:", profile_text]
    
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


def ingest_posts(config: Config) -> None:
    files = local_csv_files(config.post_dir)
    if not files:
        raise FileNotFoundError(f"No .csv files found in {config.post_dir}")
    config.post_shards_dir.mkdir(parents=True, exist_ok=True)
    handles = [shard_path(config, index).open("w", encoding="utf-8") for index in range(config.post_shards)]
    total_kept = 0
    source_index = 0
    try:
        for path in files:
            scanned = kept = 0
            for row in tqdm(iter_csv_rows(path), desc=f"ingest:{path.name}"):
                scanned += 1
                source_index += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                user_id = compact_text(row.get("author"))
                timestamp = parse_comment_date(row.get("created_at"))
                if not user_id or timestamp is None:
                    continue
                category = compact_text(row.get("label") or row.get("source") or "voz")
                normalized = {"post_id": row.get("post_id"), "user_id": user_id,
                    "text": str(row.get("text") or ""), "thread_id": row.get("thread_id"),
                    "text_hash": row.get("text_hash"), "category": category, "timestamp": timestamp}
                stable_id = stable_post_id(normalized, category)
                record = {"post_id": stable_id,
                    "user_id": user_id, "author": user_id, "category": category,
                    "label": compact_text(row.get("label") or "voz"),
                    "source": compact_text(row.get("source") or "voz"),
                    "thread_id": compact_text(row.get("thread_id")),
                    "text_hash": compact_text(row.get("text_hash")),
                    "text": str(row.get("text") or ""),
                    "timestamp": format_vietnam_datetime(timestamp), "timestamp_ms": timestamp,
                    "source_index": source_index}
                handle = handles[user_shard(user_id, config.post_shards)]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1; total_kept += 1
            print(f"{path.name}: scanned={scanned:,}, kept={kept:,}")
    finally:
        for handle in handles:
            handle.close()
    print(f"Stored posts={total_kept:,} in {config.post_shards} JSONL shards")


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
        seen_post_ids: set[str] = set()
        for row in iter_local_jsonl(path):
            post_id = str(row.get("post_id") or "")
            if post_id in seen_post_ids:
                continue
            seen_post_ids.add(post_id)
            user_id = row["user_id"]
            item = aggregate.setdefault(user_id, {"count": 0, "categories": set(), "text_posts": 0,
                "text_chars": 0, "min_ts": row["timestamp_ms"], "max_ts": row["timestamp_ms"]})
            text = str(row.get("text") or "")
            item["count"] += 1; item["categories"].add(row.get("thread_id") or "Unknown thread")
            item["text_posts"] += int(bool(text.strip())); item["text_chars"] += len(text)
            item["min_ts"] = min(item["min_ts"], row["timestamp_ms"])
            item["max_ts"] = max(item["max_ts"], row["timestamp_ms"])
    eligible = []
    for user_id, item in aggregate.items():
        if item["count"] >= config.min_posts and item["text_chars"] >= config.min_text_chars:
            eligible.append((user_id, item["count"], len(item["categories"]), item["text_posts"], item["text_chars"],
                             (item["max_ts"] - item["min_ts"]) / 86_400_000))
    metric_columns = (4, 3, 2, 5, 1)
    weights = (0.35, 0.20, 0.20, 0.15, 0.10)
    ranks = [percentile_ranks([float(row[column] or 0) for row in eligible]) for column in metric_columns]
    scored = [(sum(weight * vector[index] for weight, vector in zip(weights, ranks)), row)
              for index, row in enumerate(eligible)]
    scored.sort(key=lambda item: (-item[0], -item[1][4], -item[1][2], -item[1][3], item[1][0]))
    with config.selected_users_path.open("w", encoding="utf-8") as output:
        for rank, (score, row) in enumerate(scored[:config.top_k], 1):
            keys = ("user_id", "post_count", "thread_count", "text_posts", "text_chars", "history_days")
            record = {key: value for key, value in zip(keys, row)}
            record.update({"rank": rank, "score": score})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Eligible users={len(eligible):,}; selected={min(config.top_k, len(scored)):,}")


def prepare_histories(config: Config) -> None:
    require_file(config.selected_users_path, "Run the prepare selection after ingest")
    with config.selected_users_path.open(encoding="utf-8") as source:
        selected_records = [json.loads(line) for line in source if line.strip()]
    selected = {str(row["user_id"]): row for row in selected_records}
    shard_files = list(iter_post_shards(config))
    with config.histories_path.open("w", encoding="utf-8") as output:
        for path in tqdm(shard_files, desc="prepare histories"):
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
                record = {"source": "voz_forum", "user_id": user_id, "rank": selected[user_id]["rank"],
                    "post_count": len(posts), "posts": posts}
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
    parser.add_argument("--post-dir", type=Path, default=Path("data/voz"))
    parser.add_argument("--work-dir", type=Path, default=Path("voz_persona_fresh"))
    parser.add_argument("--schema-path", type=Path, default=Path("schema/dimension.json"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-llm-users", type=int, default=0)
    parser.add_argument("--post-shards", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(post_dir=project_path(args.post_dir), work_dir=project_path(args.work_dir),
                    schema_path=project_path(args.schema_path),
                    top_k=args.top_k, max_rows_per_file=args.max_rows_per_file,
                    max_llm_users=args.max_llm_users, post_shards=args.post_shards)
    if config.post_shards < 1:
        raise ValueError("post-shards must be at least 1")
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
