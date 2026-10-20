"""Shared adapter contract.

Every module adapter exposes one ``ADAPTER`` object. The frontend renders the
result payload generically, so a new module needs only an adapter plus one
registry entry — no frontend changes.

Result payload returned by ``Adapter.run`` (every key optional except ``summary``):

    summary        list[metric]                 headline cards
    metric_groups  list[{title, note?, metrics: list[metric]}]
    replay         {available, note, x_label, series: list[{key, label, values, log_scale?}]}
    images         list[{key, label, url, caption?, pixelated?}]
    compare        {before, after, label}      keys into ``images`` for a before/after slider
    plots          list[{key, label, url, caption?}]
    charts         list[{type: "bar", key, title, x_label?, y_label?, data: list[{label, value}], note?}]
    tables         list[{key, title, columns: list[{key, label, fmt?}], rows: list[dict], note?}]
    explanation    str
    explanation_source str
    notes          list[str]                    honesty / provenance notes shown to the user
    downloads      list[{label, url, filename}]

A ``metric`` is ``{key, label, value, fmt, unit?, baseline?, baseline_label?,
higher_is_better?, hint?}``. ``value`` is always a number/bool/str returned by the
module, or the string ``"not computed"``. Adapters must never invent numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.jsonutil import NOT_COMPUTED


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "description": self.description}


@dataclass(frozen=True)
class RunContext:
    input_path: Path
    job_id: str
    out_dir: Path
    sample_id: str | None = None


@dataclass(frozen=True)
class Adapter:
    run: Callable[[RunContext], dict[str, Any]]
    list_samples: Callable[[], list[Sample]]
    make_sample: Callable[[str, Path], Path]


def metric(
    key: str,
    label: str,
    value: Any,
    fmt: str = "float4",
    *,
    unit: str | None = None,
    baseline: Any = None,
    baseline_label: str | None = None,
    higher_is_better: bool | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """fmt: float2 | float4 | sci | pct | int | bool | text."""
    item: dict[str, Any] = {"key": key, "label": label, "value": value, "fmt": fmt}
    if unit:
        item["unit"] = unit
    if baseline_label:
        item["baseline"] = NOT_COMPUTED if baseline is None else baseline
        item["baseline_label"] = baseline_label
    if higher_is_better is not None:
        item["higher_is_better"] = higher_is_better
    if hint:
        item["hint"] = hint
    return item
