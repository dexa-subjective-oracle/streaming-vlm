from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from mlx import nn


def read_checkpoint(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.parameters(),
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if extra is not None:
        payload["extra"] = extra
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    payload = read_checkpoint(path)
    model.update(payload["model"])
    if optimizer_state is not None and "optimizer" in payload:
        optimizer_state.clear()
        optimizer_state.update(payload["optimizer"])
    return payload.get("extra")
