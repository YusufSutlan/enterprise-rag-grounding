"""Offline evaluation: retrieval, abstention, citation verification, end-to-end.

Usage:
    python -m entrag.evaluate
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from .answer import abstention_score, answer_question, extractive_answer
from .corpus import FACTS, UNANSWERABLE
from .retrieval import BM25, LSA, BM25PRF, Hybrid, search
from .text import fold
from .verify import NUM_RE, verify

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"

QUESTIONS = [(f.qa[i][0], f.id, f.qa[i][1], i) for f in FACTS for i in (0, 1)]  # (q, gold id, value, qa index)
FABRICATED_NUMERIC = " Bu süre yönetici onayıyla {n} gün uzatılabilir [1]."
FABRICATED_TEXT = [
    " Bu kural yalnızca yurt dışı lokasyonlar için geçerlidir [1].",
    " İstisnalar genel müdürün kararıyla belirlenir [1].",
    " Uygulama ilgili sistemde otomatik olarak takip edilir [1].",
]


def retrievers():
    bm25 = BM25()
    lsa = LSA()
    return bm25, [bm25, lsa, BM25PRF(), Hybrid(bm25, lsa)]


def retrieval_eval(rets) -> dict:
    ids = [f.id for f in FACTS]
    out = {}
    for r in rets:
        hits, rr = {1: 0, 3: 0, 5: 0}, 0.0
        for q, gold, _, _ in QUESTIONS:
            ranked = [ids[i] for i in np.argsort(-r.scores(q))]
            for k in hits:
                hits[k] += gold in ranked[:k]
            rr += 1 / (ranked.index(gold) + 1)
        n = len(QUESTIONS)
        out[r.name] = {f"recall@{k}": v / n for k, v in hits.items()} | {"mrr": rr / n}
    return out


def abstention_eval(bm25: BM25, hybrid: Hybrid) -> dict:
    """Pick the abstention signal and threshold on a dev split, report on the untouched test split.

    Dev = first paraphrase of every section + the first 6 unanswerable questions;
    test = second paraphrase of every section + the last 6 unanswerable questions."""
    answerable = [q for q, *_ in QUESTIONS]
    qs = answerable + UNANSWERABLE
    y = np.array([1] * len(answerable) + [0] * len(UNANSWERABLE))
    dev = np.array([i for i, (_, _, _, qi) in enumerate(QUESTIONS) if qi == 0] + list(range(len(answerable), len(answerable) + 6)))
    test = np.array([i for i in range(len(qs)) if i not in set(dev)])

    names = ["BM25 top score", "idf coverage (top-3)", "LSA top cosine"]
    scores = {n: [] for n in names}
    for q in qs:
        ctx = [f for f, _ in search(hybrid, q, 3)]
        scores["BM25 top score"].append(abstention_score(bm25, q, ctx, "BM25 top score"))
        scores["idf coverage (top-3)"].append(abstention_score(bm25, q, ctx, "idf coverage (top-3)"))
        scores["LSA top cosine"].append(float(hybrid.lsa.scores(q).max()))
    scores = {n: np.array(v) for n, v in scores.items()}

    auroc_dev = {n: roc_auc_score(y[dev], scores[n][dev]) for n in names}
    auroc_test = {n: roc_auc_score(y[test], scores[n][test]) for n in names}
    chosen = max(auroc_dev, key=auroc_dev.get)

    def balanced(t):
        s = scores[chosen]
        return ((s[dev][y[dev] == 1] >= t).mean() + (s[dev][y[dev] == 0] < t).mean()) / 2

    thr = max(np.unique(scores[chosen][dev]), key=balanced)
    s_test = scores[chosen][test]
    return {
        "chosen_signal": chosen,
        "auroc_dev": auroc_dev,
        "auroc_test": auroc_test,
        "threshold": float(thr),
        "test_answerable_kept": float((s_test[y[test] == 1] >= thr).mean()),
        "test_unanswerable_rejected": float((s_test[y[test] == 0] < thr).mean()),
        "n_test": (int((y[test] == 1).sum()), int((y[test] == 0).sum())),
        "coverage": {"answerable": scores[chosen][y == 1].tolist(), "unanswerable": scores[chosen][y == 0].tolist()},
    }


def _other_number(rng: random.Random, exclude: set[str]) -> str:
    pool = sorted({n for f in FACTS for n in NUM_RE.findall(f.text)} - exclude)
    return rng.choice(pool)


def verification_eval(seed: int = 0) -> dict:
    """Run the verifier on correct answers and on answers corrupted in realistic ways."""
    rng = random.Random(seed)
    by_id = {f.id: f for f in FACTS}
    counts = {k: [0, 0] for k in ("clean extractive", "clean short", "wrong number", "wrong citation",
                                  "wrong number from the same section", "fabricated claim (with number)",
                                  "fabricated claim (no number)")}  # [flagged, total]

    for q, gold, value, _ in QUESTIONS:
        f = by_id[gold]
        others = rng.sample([x for x in FACTS if x.id != gold], 2)
        ctx = [f] + others
        clean = extractive_answer(q, ctx, BM25().idf)
        counts["clean extractive"][1] += 1
        counts["clean extractive"][0] += not verify(clean, ctx).ok

        short = f"{value[0].upper() + value[1:]} [1]."
        counts["clean short"][1] += 1
        counts["clean short"][0] += not verify(short, ctx).ok

        # Only corrupt answers that quote the gold section.
        if "[1]" not in clean:
            continue
        nums = NUM_RE.findall(clean.replace("[1]", ""))
        if nums:
            target = rng.choice(nums)
            new = _other_number(rng, set(NUM_RE.findall(f.text)))
            swapped = clean.replace(target, new, 1)
            counts["wrong number"][1] += 1
            counts["wrong number"][0] += not verify(swapped, ctx).ok
        own = sorted(set(NUM_RE.findall(f.text)) - set(NUM_RE.findall(clean)))
        if nums and own:  # a plausible, *grounded* but wrong number: outside what a lexical check can see
            swapped = clean.replace(rng.choice(nums), rng.choice(own), 1)
            counts["wrong number from the same section"][1] += 1
            counts["wrong number from the same section"][0] += not verify(swapped, ctx).ok
        wrong = clean.replace("[1]", "[2]")
        counts["wrong citation"][1] += 1
        counts["wrong citation"][0] += not verify(wrong, ctx).ok
        n = _other_number(rng, set(NUM_RE.findall(f.text)))
        counts["fabricated claim (with number)"][1] += 1
        counts["fabricated claim (with number)"][0] += not verify(clean + FABRICATED_NUMERIC.format(n=n), ctx).ok
        counts["fabricated claim (no number)"][1] += 1
        counts["fabricated claim (no number)"][0] += not verify(clean + rng.choice(FABRICATED_TEXT), ctx).ok
    return {k: {"flagged": v[0], "total": v[1], "rate": v[0] / v[1]} for k, v in counts.items()}


def _correct(q, value, hybrid, bm25, threshold, signal, bonus) -> tuple[bool, bool]:
    a = answer_question(q, hybrid, bm25, threshold, signal=signal, rank_bonus=bonus)
    return a.mode != "abstain" and fold(value) in fold(a.text), a.verdict.ok


def end_to_end(bm25: BM25, hybrid: Hybrid, threshold: float, signal: str) -> dict:
    """Extractive answerer. The rank bonus (how strongly sentence choice favours higher-ranked
    sections) is chosen on the dev paraphrases with abstention off, then everything is scored on
    the test paraphrases."""
    dev_q = [(q, v) for q, _, v, qi in QUESTIONS if qi == 0]
    test_q = [(q, v) for q, _, v, qi in QUESTIONS if qi == 1]
    grid = {b: float(np.mean([_correct(q, v, hybrid, bm25, -1e9, signal, b)[0] for q, v in dev_q])) for b in (0.01, 0.5, 1.0, 2.0, 4.0)}
    bonus = max(grid, key=grid.get)
    no_abstain = float(np.mean([_correct(q, v, hybrid, bm25, -1e9, signal, bonus)[0] for q, v in test_q]))
    res = [_correct(q, v, hybrid, bm25, threshold, signal, bonus) for q, v in test_q]
    in_top3 = float(np.mean([g in [f.id for f, _ in search(hybrid, q, 3)] for q, g, _, qi in QUESTIONS if qi == 1]))
    abstained = sum(answer_question(q, hybrid, bm25, threshold, signal=signal, rank_bonus=bonus).mode == "abstain" for q in UNANSWERABLE[6:])
    return {
        "rank_bonus_dev_grid": grid,
        "chosen_rank_bonus": bonus,
        "n_answerable": len(test_q),
        "gold_section_in_top3": in_top3,
        "answer_correct_without_abstention": no_abstain,
        "answer_correct_with_abstention": float(np.mean([r[0] for r in res])),
        "verified": float(np.mean([r[1] for r in res])),
        "n_unanswerable": len(UNANSWERABLE[6:]),
        "correctly_abstained": abstained / len(UNANSWERABLE[6:]),
    }


def plots(retr, abst, ver) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    names = list(retr)
    x, w = np.arange(len(names)), 0.26
    for j, metric in enumerate(["recall@1", "recall@3", "mrr"]):
        ax.bar(x + (j - 1) * w, [retr[n][metric] for n in names], w, label=metric)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval on 72 paraphrased questions over 36 sections")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "retrieval.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, max(abst['coverage']['answerable'] + abst['coverage']['unanswerable']) * 1.02, 26)
    ax.hist(abst["coverage"]["answerable"], bins, alpha=0.7, label="answerable (72)")
    ax.hist(abst["coverage"]["unanswerable"], bins, alpha=0.7, label="not in knowledge base (12)")
    ax.axvline(abst["threshold"], color="k", ls="--", label=f"threshold {abst['threshold']:.2f} (calibrated on dev)")
    ax.set_xlabel(f"abstention signal: {abst['chosen_signal']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "abstention.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    keys = list(ver)
    colors = ["#2ca02c" if k.startswith("clean") else "#d62728" for k in keys]
    ax.barh(keys[::-1], [ver[k]["rate"] for k in keys][::-1], color=colors[::-1])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("share of answers flagged by the verifier")
    ax.set_title("Citation verifier: green = should be 0, red = should be 1")
    for i, k in enumerate(keys[::-1]):
        ax.text(ver[k]["rate"] + 0.01, i, f"{ver[k]['rate']:.0%}  (n={ver[k]['total']})", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "verifier.png", dpi=130)
    plt.close(fig)


def main() -> None:
    bm25, rets = retrievers()
    hybrid = rets[-1]
    retr = retrieval_eval(rets)
    print(f"{'retriever':<28}{'R@1':>6}{'R@3':>6}{'R@5':>6}{'MRR':>6}")
    for n, m in retr.items():
        print(f"{n:<28}{m['recall@1']:>6.2f}{m['recall@3']:>6.2f}{m['recall@5']:>6.2f}{m['mrr']:>6.2f}")

    abst = abstention_eval(bm25, hybrid)
    print("\nAbstention AUROC dev :", {k: round(v, 3) for k, v in abst["auroc_dev"].items()})
    print("Abstention AUROC test:", {k: round(v, 3) for k, v in abst["auroc_test"].items()})
    print(f"Chosen on dev: '{abst['chosen_signal']}', threshold {abst['threshold']:.2f}: test answerable kept {abst['test_answerable_kept']:.2f}, "
          f"test unanswerable rejected {abst['test_unanswerable_rejected']:.2f}")  # fmt: skip

    ver = verification_eval()
    print("\nVerifier:")
    for k, v in ver.items():
        print(f"  {k:<34}{v['flagged']:>3}/{v['total']:<3} flagged ({v['rate']:.0%})")

    e2e = end_to_end(bm25, hybrid, abst["threshold"], abst["chosen_signal"])
    print("\nEnd-to-end (extractive answerer, test split):")
    for k, v in e2e.items():
        print(f"  {k:<36}{v if not isinstance(v, float) else round(v, 3)}")
    plots(retr, abst, ver)
    (ROOT / "reports" / "evaluation.json").write_text(
        json.dumps({"retrieval": retr, "abstention": {k: v for k, v in abst.items() if k != "coverage"},
                    "verifier": ver, "end_to_end": e2e}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
