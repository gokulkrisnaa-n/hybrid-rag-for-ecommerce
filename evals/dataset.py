"""
The retrieval evaluation set: one JSON object per line in evals/ground_truth.jsonl.

    {"id": "samba-espadrille",
     "query": "samba espadrille",
     "filters": {},
     "relevant": {"8286625267840": 2, "8286850580608": 2, "7731170115712": 1},
     "tags": ["exact-name"],
     "notes": "Sambas first; espadrilles from other brands are not what was meant."}

Relevance is graded:
    2 = exactly what the shopper asked for
    1 = an acceptable alternative (right kind of product, not the exact one)
Anything not listed counts as not relevant.

JSON Lines rather than one big JSON file: one case per line keeps diffs readable
and lets the labeling tool append without rewriting the file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.query_models import SearchFilters, SearchQuery

DEFAULT_PATH = Path(__file__).parent / "ground_truth.jsonl"
PLAN_PATH = Path(__file__).parent / "query_plan.jsonl"


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    query: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    relevant: dict[int, int] = Field(description="product_id -> grade (1 or 2)")
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("relevant")
    @classmethod
    def _grades(cls, v: dict[int, int]) -> dict[int, int]:
        if not v:
            raise ValueError("a case needs at least one relevant product")
        bad = {pid: g for pid, g in v.items() if g not in (1, 2)}
        if bad:
            raise ValueError(f"grades must be 1 or 2, got {bad}")
        return v

    def to_search_query(self, limit: int) -> SearchQuery:
        return SearchQuery(text=self.query, filters=self.filters, limit=limit)

    def to_json_line(self) -> str:
        data = self.model_dump(mode="json", exclude_defaults=True)
        data["filters"] = self.filters.model_dump(mode="json", exclude_defaults=True)
        data["relevant"] = {str(k): v for k, v in self.relevant.items()}
        ordered = {k: data[k] for k in ("id", "query", "filters", "relevant", "tags", "notes") if k in data}
        return json.dumps(ordered, ensure_ascii=False)


class PlannedQuery(BaseModel):
    """A query waiting to be labeled: everything in an EvalCase except the judgments."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    query: str = Field(min_length=1)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="", description="grading guide: what counts as grade 2 and grade 1")


def load_plan(path: Path = PLAN_PATH) -> list[PlannedQuery]:
    if not path.exists():
        raise FileNotFoundError(f"No query plan at {path}")
    plan, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = PlannedQuery.model_validate_json(line)
        except ValueError as e:
            raise ValueError(f"{path.name} line {n}: {e}") from None
        if item.id in seen:
            raise ValueError(f"{path.name} line {n}: duplicate id {item.id!r}")
        seen.add(item.id)
        plan.append(item)
    return plan


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "case"


def load_cases(path: Path = DEFAULT_PATH) -> list[EvalCase]:
    """Load and validate every case. Errors name the line, so a bad edit is easy to find."""
    if not path.exists():
        return []
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except ValueError as e:
            raise ValueError(f"{path.name} line {n}: {e}") from None
        if case.id in seen:
            raise ValueError(f"{path.name} line {n}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def save_cases(cases: list[EvalCase], path: Path = DEFAULT_PATH) -> None:
    path.write_text("".join(c.to_json_line() + "\n" for c in cases), encoding="utf-8")