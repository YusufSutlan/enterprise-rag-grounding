"""Three retrievers over the section index and a rank-fusion hybrid.

- BM25 on stemmed terms (exact lexical match)
- LSA: TF-IDF followed by truncated SVD (a latent "semantic" space learned from the corpus)
- Hybrid: reciprocal rank fusion of both
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import FACTS, Fact
from .text import tokenize


def index_text(f: Fact) -> str:
    return f"{f.title} {f.text}"


class BM25:
    name = "BM25"

    def __init__(self, facts: list[Fact] = FACTS, k1: float = 1.5, b: float = 0.75):
        self.facts, self.k1, self.b = facts, k1, b
        self.docs = [Counter(tokenize(index_text(f))) for f in facts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = self.lengths.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.max_idf = max(self.idf.values())

    def term_weights(self, query: str) -> dict[str, float]:
        return {t: 1.0 for t in set(tokenize(query))}

    def scores(self, query: str, weights: dict[str, float] | None = None) -> np.ndarray:
        weights = weights or self.term_weights(query)
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term, w in weights.items():
            if term in self.idf:
                tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
                out += w * self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def coverage(self, query: str, doc_index: int) -> float:
        """Share of the query's information (idf mass) that the document actually contains.
        Terms unseen in the whole corpus count as maximally informative and unmatched."""
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        total = sum(self.idf.get(t, self.max_idf) for t in terms)
        hit = sum(self.idf[t] for t in terms if t in self.docs[doc_index])
        return hit / total


class LSA:
    name = "LSA"

    def __init__(self, facts: list[Fact] = FACTS, n_components: int = 24, seed: int = 0):
        self.vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True)
        tfidf = self.vec.fit_transform([index_text(f) for f in facts])
        self.svd = TruncatedSVD(n_components=min(n_components, tfidf.shape[1] - 1), random_state=seed)
        self.doc_vecs = self._norm(self.svd.fit_transform(tfidf))

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def scores(self, query: str) -> np.ndarray:
        q = self._norm(self.svd.transform(self.vec.transform([query])))
        return (q @ self.doc_vecs.T).ravel()


class BM25PRF(BM25):
    """BM25 with pseudo-relevance feedback: add the top terms of the best 3 documents."""

    name = "BM25 + PRF"

    def scores(self, query: str, weights=None) -> np.ndarray:
        base = super().scores(query)
        top = np.argsort(-base)[:3]
        feedback: Counter = Counter()
        for i in top:
            for t, c in self.docs[i].items():
                feedback[t] += c * self.idf[t]
        weights = self.term_weights(query)
        for t, _ in feedback.most_common(6):
            weights[t] = max(weights.get(t, 0.0), 0.4)
        return super().scores(query, weights)


@dataclass
class Hybrid:
    bm25: BM25
    lsa: LSA
    k: int = 60
    name: str = "Hybrid (BM25 + LSA, RRF)"

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.bm25.docs))
        for s in (self.bm25.scores(query), self.lsa.scores(query)):
            ranks = np.empty(len(s), dtype=int)
            ranks[np.argsort(-s)] = np.arange(len(s))
            fused += 1.0 / (self.k + ranks + 1)
        return fused


def search(retriever, query: str, k: int = 3) -> list[tuple[Fact, float]]:
    s = retriever.scores(query)
    return [(retriever.facts[i] if hasattr(retriever, "facts") else FACTS[i], float(s[i])) for i in np.argsort(-s)[:k]]
