from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class StreamingArgs:
    """Track runtime state for append/shrink modes."""

    pos_mode: Literal["append", "shrink"] = "shrink"
    all_text: bool = False
    max_turns: Optional[int] = None
    text_sink: int = 0
    text_sliding_window: int = 0
    input_ids: Optional["ArrayLike"] = None
    video_grid_thw: Optional["ArrayLike"] = None
    second_per_grid_ts: Optional["ArrayLike"] = None
    last_cache_position: int = -1

    def __post_init__(self) -> None:
        if self.pos_mode not in {"append", "shrink"}:
            raise ValueError("pos_mode must be append or shrink")
