"""Turkish-aware tokenization shared by retrieval, abstention and verification."""

from __future__ import annotations

import re

STOPWORDS = set(
    "ve ile bir bu da de için ise ya veya mi mı mu mü ne kaç olan olarak en çok daha gibi her ama "
    "ancak şu o ben sen biz siz var yok değil kadar sonra önce içinde hangi kim nasıl ne zaman kadar "
    "mı mi lazım gerekir gerekiyor olur olarak bana benim bir".split()
)


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str, prefix: int | None = 5) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, keep the first `prefix` letters.

    Prefix truncation is a cheap stemmer for agglutinative Turkish: "yenilenir", "yenileme" and
    "yenilemeler" all become "yenil"."""
    tokens = [t for t in re.findall(r"[a-zçğıöşü0-9]+", turkish_lower(text)) if t not in STOPWORDS]
    return [t[:prefix] for t in tokens] if prefix else tokens


_FOLD = str.maketrans("çğıöşü", "cgiosu")


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", turkish_lower(text).translate(_FOLD))
