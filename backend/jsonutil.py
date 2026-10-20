"""JSON-safe conversion of module outputs. Never invents numeric metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.paths import RUNTIME_ROOT

NOT_COMPUTED = "not computed"


def maybe_float(value: Any) -> float | str:
    if value is None or isinstance(value, bool):
        return NOT_COMPUTED
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_COMPUTED
    if not np.isfinite(number):
        return NOT_COMPUTED
    return number


def maybe_int(value: Any) -> int | str:
    if value is None or isinstance(value, bool):
        return NOT_COMPUTED
    try:
        return int(value)
    except (TypeError, ValueError):
        return NOT_COMPUTED


def maybe_bool(value: Any) -> bool | str:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return NOT_COMPUTED


def maybe_text(value: Any) -> str:
    if value is None:
        return NOT_COMPUTED
    text = str(value).strip()
    return text or NOT_COMPUTED


def float_list(values: Any) -> list[float | None] | str:
    """Per-iteration history. Non-finite entries become None (gaps), never fabricated."""
    if values is None:
        return NOT_COMPUTED
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return NOT_COMPUTED
    if arr.size == 0:
        return NOT_COMPUTED
    return [float(x) if np.isfinite(x) else None for x in arr]


def save_gray_png(image: np.ndarray, dest: Path) -> Path:
    from skimage import io as skio

    dest.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(dest, (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8), check_contrast=False)
    return dest


def path_to_url(path: str | Path | None, runtime_root: Path = RUNTIME_ROOT) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        rel = target.resolve().relative_to(runtime_root.resolve())
    except ValueError:
        return None
    return "/outputs/" + rel.as_posix()


def download(label: str, path: str | Path | None) -> dict[str, str] | None:
    url = path_to_url(path)
    if url is None:
        return None
    return {"label": label, "url": url, "filename": Path(path).name}
