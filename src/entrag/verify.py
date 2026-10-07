"""Citation verification: is every claim in an answer actually supported by the sections it cites?

Three cheap, auditable checks per sentence (no second LLM call, so it can run on every answer):
1. every sentence must carry a citation, and every citation must point at a retrieved section;
2. every number in the sentence must appear in the cited text (the classic hallucination:
   a fluent answer with the wrong limit, deadline or amount);
3. enough of the sentence's content words must be supported by the cited text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Fact
from .retrieval import index_text
from .text import tokenize

NOT_FOUND = "Bu bilgi şirket belgelerinde yer almıyor."
CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d+(?:[.,:/]\d+)*")
_SENT_SPLIT = re.compile(r"(?<=[^\d.][.!?])\s+")


@dataclass
class Verdict:
    ok: bool
    issues: list[str] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def verify(answer: str, contexts: list[Fact], min_support: float = 0.5) -> Verdict:
    if answer.strip() == NOT_FOUND:
        return Verdict(True)
    issues: list[str] = []
    for sent in split_sentences(answer):
        cites = [int(c) for c in CITE_RE.findall(sent)]
        body = CITE_RE.sub("", sent)
        if not cites:
            issues.append(f"alıntısız cümle: '{body.strip()}'")
            continue
        bad = [c for c in cites if not 1 <= c <= len(contexts)]
        if bad:
            issues.append(f"geçersiz alıntı numarası: {bad}")
            continue
        cited = " ".join(index_text(contexts[c - 1]) for c in cites)
        cited_numbers = set(NUM_RE.findall(cited))
        for number in NUM_RE.findall(body):
            if number not in cited_numbers:
                issues.append(f"kaynakta olmayan sayı: {number}")
        words = [t for t in tokenize(body) if not t.isdigit()]
        if words:
            support = sum(t in set(tokenize(cited)) for t in words) / len(words)
            if support < min_support:
                issues.append(f"düşük kaynak desteği ({support:.2f}): '{body.strip()}'")
    return Verdict(not issues, issues)
