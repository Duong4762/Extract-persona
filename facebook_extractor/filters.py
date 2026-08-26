"""Rules for excluding commercial and advertising Facebook content."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase, remove Vietnamese accents, and collapse whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = normalized.replace("đ", "d")
    return re.sub(r"\s+", " ", normalized).strip()


PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)"
)
PRICE_PATTERN = re.compile(
    r"(?<!\d)\d+(?:[.,]\d+)*\s*(?:k|nghin|ngan|tr|trieu|d|dong|vnd)\b"
)
SHOP_URL_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+|shopee\.\S+|lazada\.\S+|tiki\.\S+|"
    r"chotot\.\S+|sendo\.\S+|tiktok\.com/\S+)"
)
CONTACT_PLATFORM_PATTERN = re.compile(r"\b(?:zalo|telegram|whatsapp|messenger)\b")
CONTACT_ACTION_PATTERN = re.compile(
    r"\b(?:lien he|nhan tin|inbox|ib|goi ngay|ket ban|add|hotline|sdt|so dien thoai)\b"
)
SALES_ACTION_PATTERN = re.compile(
    r"\b(?:dat hang|chot don|len don|order|mua ngay|chot ngay|nhan dat hang|"
    r"nhan order|nhan cung cap|ban si|ban le|si le|phan phoi|lam dai ly)\b"
)
QUOTATION_PATTERN = re.compile(
    r"\b(?:bao gia|nhan bao gia|xin bao gia|gui bao gia|bang gia|gia si|gia le|"
    r"gia dai ly|gia tot nhat|gia uu dai|gia canh tranh)\b"
)
PROMOTION_PATTERN = re.compile(
    r"\b(?:giam gia|khuyen mai|uu dai|sale off|flash sale|sale soc|gia soc|"
    r"gia huy diet|freeship|free ship|mien phi van chuyen|xa kho|thanh ly|"
    r"voucher|ma giam gia|tang kem|qua tang)\b"
)
DELIVERY_PATTERN = re.compile(
    r"\b(?:ship cod|ship toan quoc|giao hang toan quoc|giao hang tan noi|"
    r"duoc kiem hang|nhan hang thanh toan|thanh toan khi nhan hang)\b"
)
STOCK_PATTERN = re.compile(
    r"\b(?:con hang|san hang|hang co san|hang moi ve|mau moi ve|co san|ve hang|"
    r"sap het hang|chi con \d+|con duy nhat \d+)\b"
)
COMMERCIAL_HASHTAG_PATTERN = re.compile(
    r"#(?:sale|giamgia|khuyenmai|freeship|chotdon|order|giasi|bansi|banle|"
    r"muaban|thanhly|xakho)\b"
)
DIRECT_AD_PATTERNS = (
    re.compile(r"\bbao gia qua (?:zalo|inbox|messenger)\b"),
    re.compile(r"\blien he (?:zalo|inbox|messenger)\b"),
    re.compile(r"\b(?:zalo|inbox|messenger) de (?:bao gia|dat hang|chot don|tu van)\b"),
    re.compile(r"\binbox de (?:biet gia|nhan gia|chot don|dat hang)\b"),
    re.compile(r"\bnhan tin de (?:bao gia|dat hang|chot don|tu van)\b"),
    re.compile(r"\bgoi ngay de (?:dat hang|duoc tu van|nhan bao gia)\b"),
    re.compile(r"\bdat hang ngay hom nay\b"),
    re.compile(r"\bchot don ngay\b"),
    re.compile(r"\bso luong co han\b"),
    re.compile(r"\buu dai chi hom nay\b"),
    re.compile(r"\bnhan si so luong lon\b"),
    re.compile(r"\bcan tim dai ly\b"),
    re.compile(r"\btuyen dai ly\b"),
    re.compile(r"\bmo dai ly\b"),
)


def is_advertising(text: str) -> bool:
    """Return whether content has strong commercial/advertising signals."""
    normalized = normalize_text(text)
    if not normalized:
        return False

    if any(pattern.search(normalized) for pattern in DIRECT_AD_PATTERNS):
        return True
    if QUOTATION_PATTERN.search(normalized):
        return True
    if SALES_ACTION_PATTERN.search(normalized):
        return True
    if PROMOTION_PATTERN.search(normalized):
        return True
    if DELIVERY_PATTERN.search(normalized):
        return True
    if COMMERCIAL_HASHTAG_PATTERN.search(normalized):
        return True

    has_phone = bool(PHONE_PATTERN.search(normalized))
    has_price = bool(PRICE_PATTERN.search(normalized))
    has_url = bool(SHOP_URL_PATTERN.search(normalized))
    has_platform = bool(CONTACT_PLATFORM_PATTERN.search(normalized))
    has_contact_action = bool(CONTACT_ACTION_PATTERN.search(normalized))
    has_stock = bool(STOCK_PATTERN.search(normalized))

    if has_platform and has_contact_action:
        return True
    if has_platform and has_phone:
        return True
    if has_contact_action and has_phone:
        return True
    if has_price and has_contact_action:
        return True
    if has_price and has_platform:
        return True
    if has_price and has_stock:
        return True
    if has_url and (has_contact_action or has_price or has_stock):
        return True
    return False
