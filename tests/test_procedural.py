"""Curated procedure layer and claim-quote verification."""
from __future__ import annotations

from app import procedural
from app.legal_engine import _claims, speakable


def test_knowledge_base_loads_and_validates() -> None:
    assert procedural.KB["version"]
    assert len(procedural.PROCEDURES) >= 8


def test_every_procedure_is_usable() -> None:
    for procedure in procedural.PROCEDURES:
        assert len(procedure["steps"]) >= 3, procedure["id"]
        assert procedure["source"]["url"].startswith("https://")
        assert procedure["review_status"] in {"approved", "draft"}
        assert procedure["signals"], procedure["id"]


def test_matching_picks_the_right_procedure() -> None:
    match = procedural.best_match("police refused to register my fir at the thana")
    assert match is not None
    assert match["id"] == "police-fir"


def test_matching_ignores_unrelated_text() -> None:
    assert procedural.best_match("what is the weather in pune") is None


def test_word_boundaries_prevent_false_positive() -> None:
    # "first" must not trigger the FIR procedure.
    assert procedural.best_match("this is the first time I am asking") is None


def test_public_shape_never_hides_review_state() -> None:
    public = procedural.public(procedural.PROCEDURES[0])
    assert "review_status" in public
    assert public["reviewed"] is (public["review_status"] == "approved")
    assert public["steps"][0]["order"] == 1


def test_claims_require_an_exact_quote() -> None:
    sources = [{"id": "src-1", "text": "No person shall be deprived of life or personal liberty except according to procedure established by law."}]
    good, rejected = _claims([{"text": "Liberty is protected.", "source_id": "src-1", "quote": "except according to procedure established by law"}], sources)
    assert len(good) == 1 and not rejected

    bad, rejected_bad = _claims([{"text": "Liberty is protected.", "source_id": "src-1", "quote": "this sentence does not exist"}], sources)
    assert not bad
    assert rejected_bad == ["claim_0_not_exactly_supported"]


def test_claims_reject_unknown_source() -> None:
    clean, rejected = _claims([{"text": "x", "source_id": "invented", "quote": "anything"}], [])
    assert clean == []
    assert rejected


def test_speakable_strips_markdown_and_keeps_it_short() -> None:
    text = "## Heading\n\n- first point\n- second point with **bold** text that runs on and on " * 4
    spoken = speakable(text, 200)
    assert len(spoken) <= 201
    assert "#" not in spoken and "*" not in spoken
