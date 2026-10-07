from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from entrag.answer import NOT_FOUND, answer_question, build_prompt, extractive_answer
from entrag.corpus import FACTS, UNANSWERABLE
from entrag.retrieval import BM25, LSA, Hybrid, search
from entrag.text import fold, tokenize, turkish_lower
from entrag.verify import split_sentences, verify

BY_ID = {f.id: f for f in FACTS}


@pytest.fixture(scope="module")
def stack():
    bm25 = BM25()
    return bm25, Hybrid(bm25, LSA())


def test_corpus_is_consistent():
    assert len(FACTS) == 36 and len({f.id for f in FACTS}) == 36
    for f in FACTS:
        for question, value in f.qa:
            assert fold(value) in fold(f.text), (f.id, value)
    assert len(UNANSWERABLE) == 12


def test_turkish_text_helpers():
    assert turkish_lower("IŞIK İZİN") == "ışık izin"
    assert tokenize("yenilenir yenileme yenilemeler") == ["yenil"] * 3
    assert fold("Çalışan ŞİKÂYET") == "calisan sikâyet"


def test_retrieval_finds_the_right_section_for_a_paraphrase(stack):
    bm25, hybrid = stack
    top = [f.id for f, _ in search(bm25, "Laptopum ne zaman yenilenir?", 3)]
    assert "bt-laptop" in top


def test_sentence_splitter_keeps_ordinals_together():
    assert split_sentences("Kapanış ayın 5. iş günü yapılır [1]. Onay gerekir [1].") == [
        "Kapanış ayın 5. iş günü yapılır [1].",
        "Onay gerekir [1].",
    ]


def test_verifier_accepts_a_faithful_answer():
    ctx = [BY_ID["fi-avans"]]
    assert verify("Avans en fazla tahmini masrafın %80'ini kapsar [1].", ctx).ok


def test_verifier_flags_a_wrong_number():
    v = verify("Avans en fazla tahmini masrafın %90'ını kapsar [1].", [BY_ID["fi-avans"]])
    assert not v.ok and any("90" in i for i in v.issues)


def test_verifier_flags_missing_and_invalid_citations_and_unsupported_text():
    ctx = [BY_ID["fi-avans"], BY_ID["gt-kart"]]
    assert not verify("Avans masrafın %80'ini kapsar.", ctx).ok
    assert not verify("Avans masrafın %80'ini kapsar [5].", ctx).ok
    assert not verify("Kayıp kart 2 saat içinde bildirilir [1].", ctx).ok  # right words, wrong section
    assert verify(NOT_FOUND, ctx).ok


def test_verifier_cannot_see_a_grounded_but_wrong_number():
    """Documented limitation: a number that exists elsewhere in the cited section passes."""
    ctx = [BY_ID["fi-avans"]]
    assert verify("Avans, dönüşten sonraki 5 gün içinde kapatılmalıdır [1].", ctx).ok


def test_extractive_answer_quotes_a_cited_sentence():
    ctx = [BY_ID["bt-mobil"]]
    out = extractive_answer("Telefonum çalınırsa verilerin uzaktan silinmesi ne kadar sürede başlar?", ctx)
    assert out.endswith("[1].") and "1 saat" in out


class FakeClient:
    def __init__(self, text=None, error=None, stop_reason="end_turn"):
        self.calls = []

        def create(**kwargs):
            self.calls.append(kwargs)
            if error:
                raise error
            return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text or "")])

        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


Q = "Telefonum çalınırsa verilerin uzaktan silinmesi ne kadar sürede başlar?"


def test_llm_answer_is_used_when_verified(stack):
    bm25, hybrid = stack
    client = FakeClient("Uzaktan silme işlemi 1 saat içinde başlatılır [1].")
    a = answer_question(Q, hybrid, bm25, threshold=0.0, client=client)
    assert a.mode == "llm" and a.verdict.ok and "<bolumler>" in client.calls[0]["messages"][0]["content"]


def test_hallucinated_llm_answer_falls_back_to_extractive(stack):
    bm25, hybrid = stack
    a = answer_question(Q, hybrid, bm25, threshold=0.0, client=FakeClient("Silme işlemi 4 saat içinde başlar [1]."))
    assert a.mode == "extractive" and "1 saat" in a.text and any("4" in n for n in a.notes)


def test_api_outage_and_refusal_fall_back(stack):
    bm25, hybrid = stack
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    assert answer_question(Q, hybrid, bm25, 0.0, client=FakeClient(error=err)).mode == "extractive"
    assert answer_question(Q, hybrid, bm25, 0.0, client=FakeClient(stop_reason="refusal")).mode == "extractive"


def test_abstains_below_threshold(stack):
    bm25, hybrid = stack
    a = answer_question("Kripto para ile ödeme kabul ediyor muyuz?", hybrid, bm25, threshold=1e6)
    assert a.mode == "abstain" and a.text == NOT_FOUND


def test_prompt_marks_retrieved_text_as_data():
    p = build_prompt("soru?", [BY_ID["bt-vpn"]])
    assert p.startswith("<bolumler>") and "[1] Uzak Erişim (VPN)" in p and p.endswith("Soru: soru?")
