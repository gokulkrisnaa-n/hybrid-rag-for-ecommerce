"""
The judge and labeling tools without OpenAI: request building, result parsing, scoring.

    pytest tests/test_judge.py -q
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from app.generation.context import ProductDetails, build_context
from app.schemas.product_models import ProductCard
from evals import judge
from evals.answers import answer_hash, product_blocks, render_answer
from evals.label_answers import load_labels, parse_input


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def card(pid: int, title: str) -> ProductCard:
    return ProductCard(product_id=pid, title=title, brand="adidas", price_min=Decimal("100"),
                       price_max=Decimal("100"), on_sale=False, in_stock=True, sizes_available=["9"],
                       url=f"https://kith.com/p/{pid}", category="footwear", subcategory=None, gender="mens")


def answer_row(**kw) -> dict:
    context = build_context([card(11, "Samba OG White"), card(22, "Gazelle Blue")],
                            {11: ProductDetails(materials=["leather"]), 22: ProductDetails(materials=["suede"])})
    base = {"case_id": "rec-samba", "message": "adidas samba og", "history": [], "filters": {},
            "kind": "recommend", "source": "llm", "answer_message": "The Samba OG White is a leather classic.",
            "picks": [{"product_id": 11, "title": "Samba OG White", "reason": "Leather upper."}],
            "comparison": None, "context": context, "checks": {"kind": True, "valid_ids": True}}
    return {**base, **kw}


def verdict(**overrides) -> dict:
    v = {c: {"reason": "ok", "verdict": "pass"} for c in judge.CRITERIA}
    v["comparison"] = v["conversation"] = {"reason": "n/a", "verdict": "na"}
    for c, val in overrides.items():
        v[c] = {"reason": f"{c} problem", "verdict": val}
    return v


def response_body(v: dict, status: str = "completed") -> dict:
    return {"status": status, "usage": {"input_tokens": 900, "output_tokens": 300,
                                        "output_tokens_details": {"reasoning_tokens": 200}},
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": json.dumps(v)}]}]}


# --- answers ----------------------------------------------------------------

def test_product_blocks_parse_the_generation_context():
    blocks = product_blocks(answer_row()["context"])
    assert set(blocks) == {11, 22}
    assert "Title: Samba OG White" in blocks[11] and "leather" in blocks[11]


def test_answer_hash_is_stable_and_tracks_the_answer():
    a = answer_row()
    assert answer_hash(a) == answer_hash(json.loads(json.dumps(a)))
    assert answer_hash(a) != answer_hash(answer_row(answer_message="Different text."))
    assert answer_hash(a) == answer_hash(answer_row(checks={}))       # rule results don't change the answer


def test_render_answer_includes_the_comparison_table():
    table = {"columns": ["Upper", "Price"], "rows": [{"product_id": 11, "title": "Samba", "values": ["Leather", "$100"]},
                                                     {"product_id": 22, "title": "Gazelle", "values": ["Suede", "$100"]}]}
    text = render_answer(answer_row(kind="compare", comparison=table))
    assert "| Product | Upper | Price |" in text and "| Gazelle | Suede | $100 |" in text


# --- requests ---------------------------------------------------------------

def test_request_body_uses_strict_structured_output_and_the_judge_model():
    body = judge.request_body(answer_row())
    assert body["model"] == "gpt-6.1-sol" and body["store"] is False
    assert body["reasoning"] == {"effort": "medium"}
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert set(fmt["schema"]["properties"]) == set(judge.CRITERIA)
    assert "<products>" in body["input"] and "Samba OG White" in body["input"] and "Leather upper." in body["input"]


def test_reasoning_effort_can_be_omitted(monkeypatch):
    monkeypatch.setenv("JUDGE_REASONING_EFFORT", "")
    from app.core.config import get_settings
    get_settings.cache_clear()
    assert "reasoning" not in judge.request_body(answer_row())


def test_batch_lines_skip_nothing_and_reject_duplicate_ids():
    lines = [json.loads(l) for l in judge.batch_lines([answer_row(), answer_row(case_id="other")])]
    assert [l["custom_id"] for l in lines] == ["rec-samba", "other"]
    assert all(l["url"] == "/v1/responses" and l["method"] == "POST" for l in lines)
    with pytest.raises(ValueError):
        judge.batch_lines([answer_row(), answer_row()])


def test_fixed_replies_are_not_judged():
    todo, skipped = judge.judgeable([answer_row(), answer_row(case_id="oos", source="template")])
    assert [r["case_id"] for r in todo] == ["rec-samba"] and [r["case_id"] for r in skipped] == ["oos"]


# --- results ----------------------------------------------------------------

def test_parse_batch_line_reads_a_verdict_and_usage():
    line = json.dumps({"custom_id": "rec-samba", "response": {"status_code": 200, "body": response_body(verdict())},
                       "error": None})
    r = judge.parse_batch_line(line)
    assert r["verdict"]["grounded"]["verdict"] == "pass"
    assert r["usage"] == {"input": 900, "output": 300, "reasoning": 200}


@pytest.mark.parametrize("line, expected", [
    ({"custom_id": "x", "response": {"status_code": 429, "body": {"error": {"message": "rate"}}}, "error": None}, "HTTP 429"),
    ({"custom_id": "x", "response": None, "error": {"code": "batch_expired"}}, "batch_expired"),
    ({"custom_id": "x", "response": {"status_code": 200, "body": response_body(verdict(), status="incomplete")}}, "incomplete"),
    ({"custom_id": "x", "response": {"status_code": 200, "body": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "{not json"}]}]}}}, "unparsable"),
    ({"custom_id": "x", "response": {"status_code": 200, "body": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}}}, "refusal"),
])
def test_bad_results_become_error_rows_not_crashes(line, expected):
    r = judge.parse_batch_line(json.dumps(line))
    assert "verdict" not in r and expected in r["error"]


def test_overall_fails_on_any_failed_or_skipped_required_criterion():
    assert judge.overall(verdict()) == (True, [])
    assert judge.overall(verdict(grounded="fail")) == (False, ["grounded"])
    assert judge.overall(verdict(safety="na")) == (False, ["safety"])     # judge skipped a required one
    assert judge.overall(verdict(comparison="fail"))[0] is False


def test_cohen_kappa():
    assert judge.cohen_kappa([(True, True), (False, False)] * 5) == pytest.approx(1.0)
    assert judge.cohen_kappa([(True, False), (False, True)] * 5) == pytest.approx(-1.0)
    assert judge.cohen_kappa([(True, True)] * 4) is None          # both always pass: chance explains everything
    assert judge.cohen_kappa([]) is None


def test_report_counts_agreement_on_matching_hashes_only(monkeypatch, tmp_path):
    monkeypatch.setattr(judge, "CASES_PATH", tmp_path / "missing.jsonl")
    good, bad = answer_row(), answer_row(case_id="rec-bad", answer_message="The Samba has a gum sole.")
    results = [{"case_id": "rec-samba", "verdict": verdict(), "usage": {}},
               {"case_id": "rec-bad", "verdict": verdict(grounded="fail"), "usage": {}}]
    labels = {answer_hash(good): {"label": "pass"}, answer_hash(bad): {"label": "pass", "note": "fine"},
              "stale0000000": {"label": "fail"}}
    text = judge.build_report([good, bad], [], results, {"model": "m", "prompt": "p", "mode": "sync",
                                                         "answers": "gen.jsonl"}, labels)
    assert "1/2" in text and "2 labeled answers" in text and "Agreement 50%" in text
    assert "`rec-bad`: you pass (fine); judge: grounded: grounded problem" in text
    assert "rules pass, judge fails 1" in text


# --- labeling ---------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("p", ("pass", "")), ("f wrong colour", ("fail", "wrong colour")), ("PASS  ok ", ("pass", "ok")),
    ("a", ("a", "")), ("q", ("q", "")),
])
def test_parse_input(text, expected):
    assert parse_input(text) == expected


@pytest.mark.parametrize("text", ["", "x", "q now"])
def test_parse_input_rejects_the_rest(text):
    with pytest.raises(ValueError):
        parse_input(text)


def test_later_label_replaces_earlier(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text('{"answer_hash": "abc", "label": "pass"}\n{"answer_hash": "abc", "label": "fail"}\n')
    assert load_labels(p)["abc"]["label"] == "fail"
