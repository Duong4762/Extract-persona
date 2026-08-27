"""Select a compact but temporally representative Facebook content history."""

import re
from typing import Any

from .filters import normalize_text

# Vietnamese equivalents of high-value persona evidence markers. Markers are
# stored without accents because ``normalize_text`` removes Vietnamese accents.
FIRST_PERSON_MARKERS = (
    "toi", "minh", "tui", "tao", "chung toi", "chung minh", "bon toi",
    "cua toi", "cua minh", "theo toi", "voi toi", "doi voi toi",
    "toi dung", "toi da dung", "minh dung", "minh da dung",
    "toi mua", "minh mua", "toi can", "minh can", "toi muon", "minh muon",
    "toi thay", "minh thay", "toi nghi", "minh nghi", "toi cam thay",
    "minh cam thay", "toi nhan ra", "minh nhan ra", "toi lam", "minh lam",
    "toi doc", "minh doc", "toi nau", "minh nau", "toi di lam",
    "minh di lam", "toi du lich", "minh du lich",
    "con toi", "con minh", "be nha toi", "be nha minh", "vo toi", "vo minh",
    "chong toi", "chong minh", "nguoi yeu toi", "nguoi yeu minh",
    "me toi", "me minh", "bo toi", "bo minh", "ba toi", "ba minh",
    "gia dinh toi", "gia dinh minh", "nha toi", "nha minh",
    "van phong toi", "van phong minh", "hoc sinh cua toi", "hoc sinh cua minh",
    "cho cua toi", "cho nha minh", "meo cua toi", "meo nha minh",
    "thu cung cua toi", "thu cung nha minh",
)

PREFERENCE_VALUE_MARKERS = (
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
)

COMPARISON_REASONING_MARKERS = (
    "tot hon", "kem hon", "te hon", "so voi", "tuong tu", "khac voi",
    "doi voi", "hay nhat", "tot nhat", "te nhat", "kem nhat", "nhieu nhat",
    "it nhat", "boi vi", "vi", "do do", "vi vay", "cho nen", "de ma",
    "ket qua la", "thay vi", "ly do", "uu diem", "nhuoc diem", "mat tot",
    "mat han che", "tuy nhien", "nhung", "mac du", "tru khi", "ngoai tru",
    "sau khi thu", "toi da thu", "minh da thu", "chung toi da thu",
    "thu nhieu loai", "sau khi dung", "sau khi su dung", "sau khi doc",
    "sau khi mac", "hoat dong tot hon", "dung tot", "chay tot",
    "khong hoat dong", "khong dung duoc", "khong hieu qua",
)

DOMAIN_DETAIL_MARKERS = (
    "cai dat", "lap dat", "da cai", "lap rap", "cau hinh", "thiet lap",
    "tuong thich", "sua chua", "da sua", "thay the", "dung cu", "oc vit",
    "gia do", "gan tuong", "kich thuoc", "do dai", "cm", "mm", "met",
    "gam", "kg", "chat lieu", "vai", "vai cotton", "da", "kim loai",
    "nhua", "go", "thep khong gi", "kich co", "vua", "rong", "chat",
    "cong suat", "dien ap", "von", "ampe", "pin", "sac", "bo sac",
    "bluetooth", "wifi", "usb", "hdmi", "phan mem", "ung dung",
    "ung dung di dong", "man hinh", "do phan giai", "camera", "ong kinh",
    "am thanh", "am tram", "am luong", "cong thuc", "nguyen lieu",
    "tap luyen", "bai tap", "hiep", "km", "calo", "huong dan su dung",
    "huong dan", "quy trinh", "chuong", "cot truyen", "nhan vat", "tac gia",
    "nguoi ke", "phien ban",
)

SENSITIVE_ADJACENT_MARKERS = (
    "dau", "dau lung", "dau co", "dau khop", "man tinh", "bac si", "y ta",
    "benh vien", "phong kham", "y te", "thuoc", "suc khoe", "lanh manh",
    "tri lieu", "nha tri lieu", "vat ly tri lieu", "lo au", "cang thang",
    "giac ngu", "mat ngu", "di ung", "tieu duong", "huyet ap", "viem khop",
    "chan thuong", "phau thuat", "phuc hoi", "tu the", "nep", "di chuyen",
    "xe lan", "khung tap di", "gay chong", "mang thai", "thai ky", "thai san",
    "cho con bu", "em be", "tre em", "tre nho", "thieu nien", "cha me",
    "nuoi day con", "ong ba", "ba ngoai", "ba noi", "ong ngoai", "ong noi",
    "nguoi cham soc", "nguoi cao tuoi", "nha tho", "kinh thanh", "cau nguyen",
    "ton giao",
)

MARKER_GROUPS = (
    (FIRST_PERSON_MARKERS, 1.8),
    (PREFERENCE_VALUE_MARKERS, 1.6),
    (COMPARISON_REASONING_MARKERS, 1.8),
    (DOMAIN_DETAIL_MARKERS, 1.0),
    (SENSITIVE_ADJACENT_MARKERS, 1.4),
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
