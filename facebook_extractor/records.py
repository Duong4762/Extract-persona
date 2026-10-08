"""Facebook content normalization, filtering, and profile rendering."""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


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
    return datetime.fromtimestamp(
        timestamp_ms / 1000, tz=VIETNAM_TIMEZONE
    ).isoformat(timespec="seconds")


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
        str(post.get("post_id") or ""),
        str(post.get("text_hash") or ""),
        str(post.get("thread_id") or ""),
        str(post.get("timestamp_ms") or ""),
        post_text(post).lower(),
    )


def filter_posts(
    posts: list[dict[str, Any]], *, min_post_text_chars: int
) -> list[dict[str, Any]]:
    """Keep advertising/commercial posts here -- a user whose own posting behavior is
    mostly selling/affiliate content is still a real persona to capture (see the
    social_engagement_style "Seller / affiliate" value), not noise to drop at this stage.
    ``build_post_prompt`` is responsible for restricting what that content may be used
    as evidence for, not this function.
    """
    kept, seen = [], set()
    for raw_post in posts:
        post = normalize_post_record(raw_post)
        if post.get("timestamp_ms") is None:
            continue
        if len(post_text(post)) < min_post_text_chars:
            continue
        if post_key(post) in seen:
            continue
        seen.add(post_key(post))
        kept.append(post)
    kept.sort(key=lambda row: (int(row.get("timestamp_ms") or 0), int(row.get("source_index") or 0)))
    return kept


def render_post(post: dict[str, Any], index: int, max_post_chars: int) -> str:
    lines = [
        f"[content {index}]",
        f"timestamp: {post.get('timestamp') or 'unknown'}",
        f"type: {post.get('category') or 'unknown'}",
        f"text: {compact_text(post_text(post), max_post_chars)}",
    ]
    return "\n".join(lines)


def assemble_profile(row: dict[str, Any], max_chars: int) -> str:
    """Render the declared profile plus every selected post into one text blob.

    Each post's text share of ``max_chars`` is derived from how many posts were
    selected, rather than a separate fixed per-post limit, so the per-post budget
    shrinks or grows with the actual number of posts the content-selection step kept.
    """
    posts = row.get("posts") or []
    profile = row.get("profile") or {}
    parts = ["Vietnamese Facebook user profile."]
    profile_lines = []
    for key in (
        "name", "about", "gender", "dateOfBirth", "birthday", "age_range",
        "hometown", "relationship_status", "religion", "education", "work",
        "isVietnamese",
    ):
        value = profile.get(key)
        if value not in (None, "", []):
            profile_lines.append(f"{key}: {compact_text(value, 3000)}")
    if profile_lines:
        parts.append("[declared profile]\n" + "\n".join(profile_lines))
    header = "\n\n".join(parts)
    if posts:
        remaining_chars = max(0, max_chars - len(header))
        max_post_chars = max(30, remaining_chars // len(posts))
    else:
        max_post_chars = 0
    parts.extend(render_post(post, index, max_post_chars) for index, post in enumerate(posts, 1))
    return "\n\n".join(parts)[:max_chars]
