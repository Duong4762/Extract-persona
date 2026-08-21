"""Build schema-constrained personas from Stack Overflow posting histories.

Stages: ingest, prepare, compact, extract and stats.
"""

from __future__ import annotations

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

from tqdm.auto import tqdm

from llm_client import LLMSettings, complete_prompt
from persona_coverage_chart import render_category_coverage_chart


PROJECT_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_TYPES = {"direct", "structured_claim", "summary_inference", "unsupported"}
NULLISH_VALUES = {"", "null", "none", "n/a", "na", "unknown", "unsupported", "not applicable"}
USER_ID_ALIASES = ("user_id", "owner_user_id", "OwnerUserId", "account_id")
POST_ID_ALIASES = ("post_id", "Id", "id")
POST_TYPE_ALIASES = ("post_type", "PostTypeId", "post_type_id")
TIMESTAMP_ALIASES = ("timestamp", "creation_date", "CreationDate")
TAGS_ALIASES = ("tags", "Tags")
TITLE_ALIASES = ("title", "Title")
TEXT_ALIASES = ("text", "body", "Body")
SCORE_ALIASES = ("score", "Score")
ACCEPTED_ALIASES = ("accepted", "is_accepted", "accepted_answer")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Config:
    input_dir: Path
    work_dir: Path
    schema_path: Path
    mapping_path: Path
    top_k: int = 100_000
    min_posts: int = 2
    min_text_chars: int = 200
    train_fraction: float = 0.8
    max_rows_per_file: int = 0
    max_posts: int = 90
    max_profile_chars: int = 70_000
    max_dims_per_chunk: int = 50
    max_llm_users: int = 0
    post_shards: int = 256
    llm_provider: str = os.environ.get("LLM_PROVIDER", "local")
    model: str = os.environ.get("LLM_MODEL", "Qwen3-14B")
    llm_endpoint: str = os.environ.get(
        "LLM_ENDPOINT", "http://203.113.152.4:7777/llm/v1/chat/completions"
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
        return self.work_dir / "personas.jsonl"

    @property
    def prompt_log_dir(self) -> Path:
        return self.work_dir / "prompt_log"

    @property
    def persona_stats_path(self) -> Path:
        return self.work_dir / "persona_stats.json"

    @property
    def persona_coverage_chart_path(self) -> Path:
        return self.work_dir / "persona_category_coverage.png"


def first_present(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in row and row[key] is not None:
            return row[key]
    return None


def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 15].rstrip() + " ... [truncated]"
    return text


def strip_html(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"<code.*?>.*?</code>", " [CODE] ", text, flags=re.I | re.S)
    text = re.sub(r"<.*?>", " ", text, flags=re.S)
    return compact_text(text)


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        candidates = value.replace("<", " ").replace(">", " ").split()
    else:
        candidates = []
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = compact_text(candidate)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


def normalize_timestamp(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    if number < 0:
        return None
    return number * 1000 if number < 10_000_000_000 else number


def post_type(row: dict[str, Any]) -> str:
    value = str(first_present(row, POST_TYPE_ALIASES) or "").strip().lower()
    if value in {"1", "question"}:
        return "question"
    if value in {"2", "answer"}:
        return "answer"
    return "post"


def normalize_post(row: dict[str, Any], user_id: str, source_index: int) -> dict[str, Any]:
    timestamp = normalize_timestamp(first_present(row, TIMESTAMP_ALIASES))
    accepted = first_present(row, ACCEPTED_ALIASES)
    normalized = {
        "user_id": user_id,
        "post_id": str(first_present(row, POST_ID_ALIASES) or ""),
        "post_type": post_type(row),
        "timestamp": timestamp,
        "date": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date().isoformat() if timestamp else None,
        "tags": parse_tags(first_present(row, TAGS_ALIASES)),
        "title": compact_text(first_present(row, TITLE_ALIASES)),
        "text": strip_html(first_present(row, TEXT_ALIASES)),
        "score": first_present(row, SCORE_ALIASES),
        "accepted": accepted if isinstance(accepted, bool) else None,
        "source_index": source_index,
    }
    if not normalized["post_id"]:
        identity = "|".join(str(normalized[key] or "") for key in (
            "user_id", "post_type", "timestamp", "title", "text"
        ))
        normalized["post_id"] = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return normalized


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if isinstance(value, dict):
                yield value


def input_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted({*root.rglob("*.jsonl"), *root.rglob("*.jsonl.gz")})


def shard_path(config: Config, index: int) -> Path:
    return config.post_shards_dir / f"posts-{index:04d}.jsonl"


def user_shard(user_id: str, shard_count: int) -> int:
    digest = hashlib.sha1(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % shard_count


def iter_shards(config: Config) -> Iterator[Path]:
    for index in range(config.post_shards):
        path = shard_path(config, index)
        if path.is_file():
            yield path


def ingest(config: Config) -> None:
    files = input_files(config.input_dir)
    if not files:
        raise FileNotFoundError(f"No .jsonl/.jsonl.gz files found in {config.input_dir}")
    config.post_shards_dir.mkdir(parents=True, exist_ok=True)
    handles = [shard_path(config, index).open("w", encoding="utf-8") for index in range(config.post_shards)]
    source_index = total = 0
    try:
        for path in files:
            scanned = kept = 0
            for row in tqdm(iter_jsonl(path), desc=f"ingest:{path.name}"):
                scanned += 1
                if config.max_rows_per_file and scanned > config.max_rows_per_file:
                    break
                rows = row.get("posts") if isinstance(row.get("posts"), list) else [row]
                grouped_user_id = str(row.get("user_id") or "").strip()
                for raw_post in rows:
                    if not isinstance(raw_post, dict):
                        continue
                    user_id = grouped_user_id or str(first_present(raw_post, USER_ID_ALIASES) or "").strip()
                    if not user_id:
                        continue
                    source_index += 1
                    post = normalize_post(raw_post, user_id, source_index)
                    handles[user_shard(user_id, config.post_shards)].write(
                        json.dumps(post, ensure_ascii=False) + "\n"
                    )
                    kept += 1
                    total += 1
            print(f"{path.name}: scanned={scanned:,}, kept_posts={kept:,}")
    finally:
        for handle in handles:
            handle.close()
    print(f"Stored posts={total:,} in {config.post_shards} JSONL shards")


def percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        result[index] = rank / len(values)
    return result


def select_users(config: Config) -> None:
    shards = list(iter_shards(config))
    if not shards:
        raise FileNotFoundError(f"No post shards in {config.post_shards_dir}. Run ingest first.")
    aggregate: dict[str, dict[str, Any]] = {}
    for path in tqdm(shards, desc="select users"):
        seen: set[str] = set()
        for post in iter_jsonl(path):
            post_id = str(post.get("post_id") or "")
            if post_id in seen:
                continue
            seen.add(post_id)
            user_id = str(post["user_id"])
            timestamp = post.get("timestamp")
            item = aggregate.setdefault(user_id, {
                "post_count": 0, "text_chars": 0, "tags": set(), "score_sum": 0.0,
                "min_ts": timestamp, "max_ts": timestamp,
            })
            item["post_count"] += 1
            item["text_chars"] += len(str(post.get("text") or "")) + len(str(post.get("title") or ""))
            item["tags"].update(post.get("tags") or [])
            try:
                item["score_sum"] += float(post.get("score") or 0)
            except (TypeError, ValueError):
                pass
            if timestamp is not None:
                item["min_ts"] = timestamp if item["min_ts"] is None else min(item["min_ts"], timestamp)
                item["max_ts"] = timestamp if item["max_ts"] is None else max(item["max_ts"], timestamp)

    eligible = []
    for user_id, item in aggregate.items():
        if item["post_count"] < config.min_posts or item["text_chars"] < config.min_text_chars:
            continue
        history_days = (
            (item["max_ts"] - item["min_ts"]) / 86_400_000
            if item["min_ts"] is not None and item["max_ts"] is not None else 0.0
        )
        eligible.append((user_id, item["post_count"], item["text_chars"], len(item["tags"]), history_days, item["score_sum"]))
    columns = (2, 1, 3, 4, 5)
    weights = (0.35, 0.25, 0.20, 0.10, 0.10)
    ranks = [percentile_ranks([float(row[column] or 0) for row in eligible]) for column in columns]
    scored = [(sum(weight * vector[index] for weight, vector in zip(weights, ranks)), row)
              for index, row in enumerate(eligible)]
    scored.sort(key=lambda item: (-item[0], -item[1][2], -item[1][1], item[1][0]))
    with config.selected_users_path.open("w", encoding="utf-8") as output:
        for rank, (score, row) in enumerate(scored[: config.top_k], start=1):
            keys = ("user_id", "post_count", "text_chars", "tag_count", "history_days", "score_sum")
            record = dict(zip(keys, row))
            record.update({"rank": rank, "score": score})
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Eligible users={len(eligible):,}; selected={min(config.top_k, len(scored)):,}")


def filter_posts(posts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    removed: dict[str, int] = defaultdict(int)
    for post in posts:
        key = (
            str(post.get("post_type") or ""), str(post.get("timestamp") or ""),
            compact_text(post.get("title")).lower(), compact_text(post.get("text")).lower(),
        )
        reason = None
        if not (compact_text(post.get("title")) or compact_text(post.get("text"))):
            reason = "empty_title_and_text"
        elif key in seen:
            reason = "duplicate_post"
        if reason:
            removed[reason] += 1
            continue
        seen.add(key)
        kept.append(post)
    kept.sort(key=lambda post: (post.get("timestamp") is None, post.get("timestamp") or 0, post.get("source_index") or 0))
    return kept, {
        "input_posts": len(posts), "kept_posts": len(kept),
        "removed_posts": len(posts) - len(kept), "removed_by_reason": dict(sorted(removed.items())),
    }


def split_posts(posts: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    split_index = max(1, min(int(len(posts) * fraction), len(posts) - 1))
    construction, validation = posts[:split_index], posts[split_index:]
    return construction, validation, {
        "method": "per_user_temporal", "train_fraction": fraction,
        "construction_post_count": len(construction), "validation_post_count": len(validation),
    }


def prepare(config: Config) -> None:
    require_file(config.selected_users_path, "Run ingest and prepare selection first")
    selected_rows = list(iter_jsonl(config.selected_users_path))
    selected = {str(row["user_id"]): row for row in selected_rows}
    with config.histories_path.open("w", encoding="utf-8") as output:
        for path in tqdm(list(iter_shards(config)), desc="prepare histories"):
            by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for post in iter_jsonl(path):
                user_id = str(post["user_id"])
                if user_id in selected:
                    by_user[user_id].append(post)
            for user_id, raw_posts in by_user.items():
                posts, filter_summary = filter_posts(raw_posts)
                if len(posts) < 2:
                    continue
                construction, validation, split = split_posts(posts, config.train_fraction)
                record = {
                    "source": "stackoverflow", "user_id": user_id, "rank": selected[user_id]["rank"],
                    "post_count": len(posts), "validation_post_count": len(validation),
                    "temporal_split": split, "post_filter_summary": filter_summary,
                    "posts": construction, "validation_posts": validation,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Prepared histories:", config.histories_path)


def render_post(post: dict[str, Any], index: int) -> str:
    return "\n".join([
        f"[post p{index:04d}]", f"date: {post.get('date') or 'unknown'}",
        f"type: {post.get('post_type') or 'post'}",
        f"tags: {', '.join(post.get('tags') or []) or '(none)'}",
        f"title: {post.get('title') or '(untitled)'}", f"score: {post.get('score', 'unknown')}",
        f"accepted: {str(post.get('accepted')).lower() if post.get('accepted') is not None else 'n/a'}",
        f"text: {compact_text(post.get('text'))}",
    ])


def compact_profiles(config: Config) -> None:
    require_file(config.histories_path, "Run the prepare stage first")
    count = 0
    with config.histories_path.open(encoding="utf-8") as source, config.compact_profiles_path.open("w", encoding="utf-8") as output:
        for line in tqdm(source, desc="compact profiles"):
            if not line.strip():
                continue
            user = json.loads(line)
            posts = user.get("posts") or []
            # Posts are chronological. Keep at most the newest max_posts first,
            # then drop complete older posts until the profile fits. This avoids
            # cutting a post mid-sentence and protects the most recent evidence.
            selected_posts = posts[-config.max_posts:] if config.max_posts else posts[:]
            candidate_post_count = len(selected_posts)
            rendered_posts = [render_post(post, index) for index, post in enumerate(selected_posts, 1)]
            dropped_for_size = 0
            profile = "\n\n".join(rendered_posts)
            while rendered_posts and len(profile) > config.max_profile_chars:
                rendered_posts.pop(0)
                selected_posts.pop(0)
                dropped_for_size += 1
                profile = "\n\n".join(rendered_posts)
            record = {
                "user_id": user["user_id"], "source": user["source"], "post_count": user["post_count"],
                "validation_post_count": user["validation_post_count"], "temporal_split": user["temporal_split"],
                "post_filter_summary": user["post_filter_summary"], "sampled_post_count": candidate_post_count,
                "profile_post_count": len(selected_posts),
                "posts_dropped_for_profile_size": dropped_for_size,
                "compact_profile_chars": len(profile), "profile_text": profile,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Compact profiles={count:,}; output={config.compact_profiles_path}")


def category_matches(category: str, pattern: str) -> bool:
    return category.startswith(pattern[:-1]) if pattern.endswith("*") else category == pattern


def load_supported_schema(config: Config) -> list[dict[str, Any]]:
    require_file(config.schema_path, "Provide --schema-path")
    require_file(config.mapping_path, "Provide --mapping-path")
    document = json.loads(config.schema_path.read_text(encoding="utf-8"))
    dimensions = document.get("dimensions") if isinstance(document, dict) else document
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError(f"Invalid schema: {config.schema_path}")
    mapping = json.loads(config.mapping_path.read_text(encoding="utf-8"))
    supported = {
        str(category)
        for evidence in mapping.get("evidence_categories", [])
        for category in evidence.get("schema_categories", [])
    }
    skipped = {str(category) for category in mapping.get("skip_by_default_schema_categories", [])}
    return [
        dimension for dimension in dimensions
        if any(category_matches(str(dimension.get("category") or ""), pattern) for pattern in supported)
        and not any(category_matches(str(dimension.get("category") or ""), pattern) for pattern in skipped)
    ]


def chunks_by_category(dimensions: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dimension in dimensions:
        grouped[str(dimension.get("category") or "Uncategorized")].append(dimension)
    return [items[start:start + size] for items in grouped.values() for start in range(0, len(items), size)]


def build_prompt(profile: str, dimensions: list[dict[str, Any]]) -> str:
    lines = [
        "You are extracting schema-constrained persona fields from one Stack Overflow user's public posting history.",
        "", "Return ONLY valid JSON with this shape:",
        '{"fields":[{"field_id":"<listed id>","value":"<allowed value or null>","confidence":0.0,'
        '"evidence":"<exact quote or empty>","description":"<1-2 sentences or empty>",'
        '"assignment_type":"direct|structured_claim|summary_inference|unsupported"}]}',
        "", "Rules:", "- Emit exactly one object per listed dimension.",
        "- value must be exactly one allowed value or null.",
        "- Every non-null value requires an exact evidence quote present in PROFILE.",
        "- Skills, tools, expertise and technical interests may be inferred from repeated demonstrated behavior.",
        "- Sensitive, demographic, medical, financial and psychological attributes require direct self-statements.",
        "- Unsupported fields must use value=null, confidence=0.0, evidence='', description='', assignment_type='unsupported'.",
        "- Return JSON only.", "", "DIMENSIONS:",
    ]
    for dimension in dimensions:
        allowed = " | ".join(str(value) for value in dimension.get("values", [])) or "(free value)"
        lines.append(
            f"- {dimension['id']} | {dimension.get('label', dimension['id'])} | "
            f"{dimension.get('description', '')} | [{allowed}]"
        )
    lines.extend(["", "PROFILE:", profile])
    return "\n".join(lines)


def parse_fields(text: str) -> list[dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    fields = payload.get("fields") if isinstance(payload, dict) else None
    return fields if isinstance(fields, list) else []


def unsupported_field(dimension: dict[str, Any]) -> dict[str, Any]:
    return {"field_id": str(dimension["id"]), "value": None, "confidence": 0.0,
            "evidence": "", "description": "", "assignment_type": "unsupported"}


def confidence_value(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def sanitize_fields(fields: list[dict[str, Any]], dimensions: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    by_id = {str(dimension["id"]): dimension for dimension in dimensions}
    result: dict[str, dict[str, Any]] = {}
    normalized_profile = compact_text(profile)
    for raw in fields:
        if not isinstance(raw, dict):
            continue
        field_id = str(raw.get("field_id") or "").strip()
        dimension = by_id.get(field_id)
        if dimension is None:
            continue
        value = raw.get("value")
        value = None if value is None or str(value).strip().casefold() in NULLISH_VALUES else str(value).strip()
        allowed = [str(item) for item in dimension.get("values", [])]
        if value is not None and allowed and value not in allowed:
            value = None
        assignment = str(raw.get("assignment_type") or "").strip()
        evidence = str(raw.get("evidence") or "").strip()
        supported = (
            value is not None and assignment in ASSIGNMENT_TYPES and assignment != "unsupported"
            and evidence and (evidence in profile or compact_text(evidence) in normalized_profile)
        )
        clean = {
            "field_id": field_id, "value": value, "confidence": confidence_value(raw.get("confidence")),
            "evidence": evidence, "description": str(raw.get("description") or "").strip(),
            "assignment_type": assignment,
        } if supported else unsupported_field(dimension)
        prior = result.get(field_id)
        if prior is None or (clean["value"] is not None and prior["value"] is None) or clean["confidence"] > prior["confidence"]:
            result[field_id] = clean
    return [result.get(str(dimension["id"])) or unsupported_field(dimension) for dimension in dimensions]


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
    config.prompt_log_dir.mkdir(parents=True, exist_ok=True)
    for path in config.prompt_log_dir.glob("*.txt"):
        if path.is_file():
            path.unlink()


def extract(config: Config) -> None:
    require_file(config.compact_profiles_path, "Run compact first")
    dimensions = load_supported_schema(config)
    chunks = chunks_by_category(dimensions, config.max_dims_per_chunk)
    done: set[str] = set()
    if config.personas_path.exists():
        done = {str(row["user_id"]) for row in iter_jsonl(config.personas_path)}
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
            fields: list[dict[str, Any]] = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                prompt = build_prompt(record["profile_text"], chunk)
                prompt_path = config.prompt_log_dir / f"prompt-{chunk_index:04d}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                started = time.perf_counter()
                try:
                    response = call_llm(prompt, config)
                    chunk_fields = sanitize_fields(parse_fields(response), chunk, record["profile_text"])
                except Exception as error:
                    chunk_fields = sanitize_fields([], chunk, record["profile_text"])
                    tqdm.write(
                        f"user={user_id} chunk={chunk_index}/{len(chunks)} "
                        f"LLM retries exhausted; marking {len(chunk)} dimensions "
                        f"unsupported; error={error}"
                    )
                fields.extend(chunk_fields)
                categories = sorted({str(item.get("category") or "Uncategorized") for item in chunk})
                tqdm.write(
                    f"user={user_id} chunk={chunk_index}/{len(chunks)} category={','.join(categories)} "
                    f"dimensions={len(chunk)} supported={sum(f['value'] is not None for f in chunk_fields)} "
                    f"elapsed={time.perf_counter() - started:.2f}s prompt_log={prompt_path.name}"
                )
            if len(fields) != len(dimensions):
                raise RuntimeError(f"Expected {len(dimensions)} fields, got {len(fields)}")
            output.write(json.dumps({
                "user_id": user_id, "source": "stackoverflow", "post_count": record["post_count"],
                "validation_post_count": record["validation_post_count"], "temporal_split": record["temporal_split"],
                "compact_profile_chars": record["compact_profile_chars"], "fields": fields,
            }, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            done.add(user_id)
            processed += 1
    print(f"New personas={processed}; output={config.personas_path}")


def stats(config: Config) -> None:
    require_file(config.personas_path, "Run extract first")
    dimensions = load_supported_schema(config)
    categories_by_id = {str(item["id"]): str(item.get("category") or "Uncategorized") for item in dimensions}
    schema_counts: dict[str, int] = defaultdict(int)
    for category in categories_by_id.values():
        schema_counts[category] += 1
    seen: set[str] = set()
    supported_total = 0
    supported_by_category: dict[str, int] = defaultdict(int)
    for persona in iter_jsonl(config.personas_path):
        user_id = str(persona.get("user_id") or "")
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        for field in persona.get("fields") or []:
            if isinstance(field, dict) and field.get("value") is not None:
                supported_total += 1
                supported_by_category[categories_by_id.get(str(field.get("field_id") or ""), "Unknown field category")] += 1
    persona_count = len(seen)
    category_totals = [
        {"category": category, "supported_dimension_count": count}
        for category, count in sorted(supported_by_category.items(), key=lambda item: (-item[1], item[0]))
    ]
    result = {
        "persona_count": persona_count, "schema_dimension_count": len(dimensions),
        "supported_dimension_count": supported_total,
        "average_supported_dimensions_per_persona": round(supported_total / persona_count, 4) if persona_count else 0.0,
        "categories": category_totals,
    }
    config.persona_stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chart_rows = [
        {"category": category, "coverage_percentage": (
            100.0 * supported_by_category.get(category, 0) / (persona_count * dimension_count)
            if persona_count and dimension_count else 0.0
        )}
        for category, dimension_count in schema_counts.items()
    ]
    chart_rows.sort(key=lambda row: (-row["coverage_percentage"], row["category"]))
    render_category_coverage_chart(
        chart_rows, config.persona_coverage_chart_path,
        f"Category Coverage Analysis (Stack Overflow Personas: {persona_count})",
    )
    print(f"Extracted personas: {persona_count:,}")
    print(f"Average supported dimensions/persona: {result['average_supported_dimensions_per_persona']:.4f}")
    print("Persona stats:", config.persona_stats_path)
    print("Persona coverage chart:", config.persona_coverage_chart_path)


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}. {hint}.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "ingest", "prepare", "compact", "extract", "stats"), nargs="?", default="all")
    parser.add_argument("--input-dir", type=Path, default=Path("data/stackoverflow"))
    parser.add_argument("--work-dir", type=Path, default=Path("stackoverflow_persona_fresh"))
    parser.add_argument("--schema-path", type=Path, default=Path("schema/dimensions.json"))
    parser.add_argument("--mapping-path", type=Path, default=Path("stackoverflow_persona_fresh/evidence_mapping/stackoverflow_evidence_mapping.json"))
    parser.add_argument("--top-k", type=int, default=100_000)
    parser.add_argument("--max-rows-per-file", type=int, default=0)
    parser.add_argument("--max-llm-users", type=int, default=0)
    parser.add_argument("--post-shards", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = Config(
        input_dir=project_path(args.input_dir), work_dir=project_path(args.work_dir),
        schema_path=project_path(args.schema_path), mapping_path=project_path(args.mapping_path),
        top_k=args.top_k, max_rows_per_file=args.max_rows_per_file,
        max_llm_users=args.max_llm_users, post_shards=args.post_shards,
    )
    if config.post_shards < 1:
        raise ValueError("post-shards must be at least 1")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    print("Work directory:", config.work_dir.resolve())
    if args.stage in {"all", "ingest"}:
        ingest(config)
    if args.stage in {"all", "prepare"}:
        select_users(config)
        prepare(config)
    if args.stage in {"all", "compact"}:
        compact_profiles(config)
    if args.stage in {"all", "extract"}:
        extract(config)
    if args.stage in {"all", "stats"}:
        stats(config)


if __name__ == "__main__":
    main()
