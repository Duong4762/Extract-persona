"""Select a compact but temporally representative Facebook content history."""

import re
from typing import Any

from .filters import normalize_text

_TOI_WORD_PATTERN = re.compile(r"(?<!\w)toi(?!\w)")


def _with_t_abbreviation(markers: tuple[str, ...]) -> tuple[str, ...]:
    """Add a "t" abbreviation variant for every marker containing the "toi" pronoun."""
    expanded = list(markers)
    for marker in markers:
        if _TOI_WORD_PATTERN.search(marker):
            expanded.append(_TOI_WORD_PATTERN.sub("t", marker))
    return tuple(dict.fromkeys(expanded))


FIRST_PERSON_MARKERS = _with_t_abbreviation((
    "toi", "minh", "tui", "tao", "t", "chung toi", "chung minh", "bon toi",
    "cua toi", "cua minh", "theo toi", "voi toi", "doi voi toi",
    "toi dung", "toi da dung", "minh dung", "minh da dung",
    "toi mua", "minh mua", "toi can", "minh can", "toi muon", "minh muon",
    "toi thay", "minh thay", "toi nghi", "minh nghi", "toi cam thay",
    "minh cam thay", "toi nhan ra", "minh nhan ra", "toi lam", "minh lam",
    "t dung", "t da dung", "t mua", "t can", "t muon", "t thay", "t nghi",
    "t cam thay", "t nhan ra", "t lam", "t doc", "t nau", "t di lam",
    "t du lich",
    "toi doc", "minh doc", "toi nau", "minh nau", "toi di lam",
    "minh di lam", "toi du lich", "minh du lich",
    "con toi", "con minh", "be nha toi", "be nha minh", "vo toi", "vo minh",
    "chong toi", "chong minh", "nguoi yeu toi", "nguoi yeu minh",
    "con t", "be nha t", "vo t", "chong t", "nguoi yeu t", "me t", "bo t",
    "ba t", "gia dinh t", "nha t", "van phong t", "hoc sinh cua t",
    "cho nha t", "meo nha t", "thu cung nha t",
    "me toi", "me minh", "bo toi", "bo minh", "ba toi", "ba minh",
    "gia dinh toi", "gia dinh minh", "nha toi", "nha minh",
    "van phong toi", "van phong minh", "hoc sinh cua toi", "hoc sinh cua minh",
    "cho cua toi", "cho nha minh", "meo cua toi", "meo nha minh",
    "thu cung cua toi", "thu cung nha minh",
))

PREFERENCE_VALUE_MARKERS = _with_t_abbreviation((
    "toi thich", "minh thich", "toi rat thich", "minh rat thich", "toi yeu",
    "minh yeu", "toi ua", "minh ua", "toi khong thich", "minh khong thich",
    "toi ghet", "minh ghet", "toi muon", "minh muon", "toi can", "minh can",
    "toi uoc", "minh uoc", "toi khuyen nghi", "minh khuyen nghi", "nen mua",
    "khong nen mua", "rat dang mua", "yeu thich", "ua thich", "quan trong",
    "dang tien", "khong dang", "dang dong tien", "phi tien", "gia tri tot",
    "tiet kiem", "hop tui tien", "dat tien", "re", "gia re", "gia cao",
    "chat luong", "chat luong cao", "chat luong kem", "ben", "do ben",
    "ben lau", "chac chan", "mong manh", "thoai mai", "khong thoai mai",
    "an toan", "bao mat", "de dung", "kho dung", "de ve sinh", "kho ve sinh",
    "de cai dat", "kho cai dat", "de lap rap", "kho lap rap", "tien loi",
    "dang tin cay", "khong dang tin cay", "tiet kiem thoi gian",
    "tiet kiem khong gian", "nho gon", "di dong", "nhe", "chiu luc",
    "than thien moi truong", "khong doc hai", "huu co", "tu nhien",
    "khong huong lieu", "co mui", "khong mui",
))

HOBBY_INTEREST_MARKERS = _with_t_abbreviation((
    "so thich cua toi", "so thich cua minh", "dam me cua toi", "dam me cua minh",
    "toi dam me", "minh dam me", "thu vui cua toi", "thu vui cua minh",
    "toi hay choi", "minh hay choi", "toi choi", "minh choi", "toi tap",
    "minh tap", "toi luyen tap", "minh luyen tap", "toi hay xem", "minh hay xem",
    "toi hay nghe", "minh hay nghe", "toi hay doc", "minh hay doc",
    "toi suu tam", "minh suu tam", "thoi quen cua toi", "thoi quen cua minh",
    "toi danh thoi gian", "minh danh thoi gian", "toi thuong xuyen",
    "minh thuong xuyen", "so truong cua toi", "so truong cua minh",
    "toi gioi ve", "minh gioi ve", "toi la fan cua", "minh la fan cua",
    "toi hay di", "minh hay di", "cuoi tuan toi", "cuoi tuan minh",
    "thoi gian ranh toi", "thoi gian ranh minh",
))

WORK_OCCUPATION_MARKERS = _with_t_abbreviation((
    "cong viec cua toi", "cong viec cua minh", "toi lam viec", "minh lam viec",
    "toi lam nghe", "minh lam nghe", "nghe cua toi", "nghe cua minh",
    "toi lam o", "minh lam o", "toi dang lam", "minh dang lam",
    "cong ty toi", "cong ty minh", "sep toi", "sep minh",
    "dong nghiep toi", "dong nghiep minh", "toi phu trach", "minh phu trach",
    "toi quan ly", "minh quan ly", "chuc vu cua toi", "chuc vu cua minh",
    "toi kinh doanh", "minh kinh doanh", "toi khoi nghiep", "minh khoi nghiep",
    "du an cua toi", "du an cua minh", "khach hang cua toi", "khach hang cua minh",
    "toi lam chu", "minh lam chu", "nganh cua toi", "nganh cua minh",
    "toi lam tai", "minh lam tai", "deadline cua toi", "deadline cua minh",
))

OPINION_VIEWPOINT_MARKERS = _with_t_abbreviation((
    "theo quan diem cua toi", "theo quan diem cua minh", "quan diem cua toi",
    "quan diem cua minh", "toi cho rang", "minh cho rang", "toi tin rang",
    "minh tin rang", "toi nghi rang", "minh nghi rang", "ca nhan toi",
    "ca nhan minh", "toi ung ho", "minh ung ho", "toi phan doi", "minh phan doi",
    "toi khong dong y", "minh khong dong y", "toi dong y", "minh dong y",
    "toi tin tuong", "minh tin tuong", "quan diem ca nhan", "goc nhin cua toi",
    "goc nhin cua minh", "theo nhan dinh cua toi", "theo nhan dinh cua minh",
    "toi danh gia", "minh danh gia", "toi nhan dinh", "minh nhan dinh",
    "theo kinh nghiem cua toi", "theo kinh nghiem cua minh",
))

MARKER_GROUPS = (
    (FIRST_PERSON_MARKERS, 1.8),
    (PREFERENCE_VALUE_MARKERS, 1.6),
    (HOBBY_INTEREST_MARKERS, 1.7),
    (WORK_OCCUPATION_MARKERS, 1.7),
    (OPINION_VIEWPOINT_MARKERS, 1.7),
)


def _compile_markers(markers: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(marker) for marker in sorted(set(markers), key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


COMPILED_MARKER_GROUPS = tuple(
    (_compile_markers(markers), weight) for markers, weight in MARKER_GROUPS
)


def marker_score(text: str) -> float:
    normalized = normalize_text(text)
    return sum(
        len(pattern.findall(normalized)) * weight
        for pattern, weight in COMPILED_MARKER_GROUPS
    )


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def content_score(content: dict[str, Any]) -> float:
    text = str(content.get("text") or "")
    character_score = min(len(text) / 100, 5.0)
    word_count = len(re.findall(r"(?u)\b\w+\b", text))
    word_score = min(word_count / 60, 2.0)
    engagement = (
        _non_negative_int(content.get("like_count"))
        + 3 * _non_negative_int(content.get("share_count"))
        + 2 * _non_negative_int(content.get("comment_count"))
        + 2 * _non_negative_int(content.get("reply_count"))
    )
    engagement_score = min(engagement * 0.05, 1.0)
    return (
        character_score
        + word_score
        + engagement_score
        + marker_score(text)
    )


def timeline_sample_indices(
    contents: list[dict[str, Any]], sample_count: int
) -> list[int]:
    """Choose unused content nearest to evenly spaced history timestamps."""
    item_count = len(contents)
    if item_count <= 0 or sample_count <= 0:
        return []
    if sample_count >= item_count:
        return list(range(item_count))
    if sample_count == 1:
        return [(item_count - 1) // 2]

    timestamps = [int(content.get("timestamp_ms") or 0) for content in contents]
    first_timestamp, last_timestamp = timestamps[0], timestamps[-1]
    targets = [
        first_timestamp
        + (last_timestamp - first_timestamp) * index / (sample_count - 1)
        for index in range(sample_count)
    ]
    available = set(range(item_count))
    selected = []
    for target in targets:
        nearest = min(
            available,
            key=lambda index: (abs(timestamps[index] - target), index),
        )
        selected.append(nearest)
        available.remove(nearest)
    return sorted(selected)


def select_contents(
    contents: list[dict[str, Any]], *, max_contents: int, timeline_ratio: float = 0.4
) -> list[dict[str, Any]]:
    """Mix evenly distributed history with the highest-value remaining content."""
    if max_contents < 1:
        raise ValueError("max_contents must be at least 1")
    ordered = sorted(
        contents,
        key=lambda row: (int(row.get("timestamp_ms") or 0), str(row.get("text") or "")),
    )
    if len(ordered) <= max_contents:
        return ordered

    timeline_count = min(max_contents, max(0, round(max_contents * timeline_ratio)))
    timeline_indices = set(timeline_sample_indices(ordered, timeline_count))
    remaining_count = max_contents - len(timeline_indices)
    ranked_indices = sorted(
        (index for index in range(len(ordered)) if index not in timeline_indices),
        key=lambda index: (-content_score(ordered[index]), index),
    )
    selected_indices = timeline_indices.union(ranked_indices[:remaining_count])
    return [ordered[index] for index in sorted(selected_indices)]
