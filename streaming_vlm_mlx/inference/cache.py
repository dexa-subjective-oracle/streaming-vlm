from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StreamingCache:
    """Minimal cache wrapper for MLX attention."""

    key_cache: List["ArrayLike"] = field(default_factory=list)
    value_cache: List["ArrayLike"] = field(default_factory=list)
    position_cache: List["ArrayLike"] = field(default_factory=list)
    seen_tokens: int = 0

    def update(
        self,
        key_states: "ArrayLike",
        value_states: "ArrayLike",
        layer_idx: int,
    ) -> tuple["ArrayLike", "ArrayLike"]:
        self._ensure_capacity(layer_idx)
        self._ensure_capacity(layer_idx, target="value_cache")
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = self._concat(self.key_cache[layer_idx], key_states)
            self.value_cache[layer_idx] = self._concat(self.value_cache[layer_idx], value_states)
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_positions(self, position_ids: "ArrayLike", layer_idx: int) -> "ArrayLike":
        self._ensure_capacity(layer_idx, target="position_cache")
        if self.position_cache[layer_idx] is None:
            self.position_cache[layer_idx] = position_ids
        else:
            self.position_cache[layer_idx] = position_ids
        return position_ids

    def clear_from(self, keep_tokens: int) -> None:
        for idx, cached in enumerate(self.key_cache):
            if cached is None:
                continue
            self.key_cache[idx] = cached[:, :, -keep_tokens:, :]
        for idx, cached in enumerate(self.value_cache):
            if cached is None:
                continue
            self.value_cache[idx] = cached[:, :, -keep_tokens:, :]
        self.seen_tokens = keep_tokens

    def _ensure_capacity(self, layer_idx: int, target: str = "key_cache") -> None:
        store = getattr(self, target)
        while len(store) <= layer_idx:
            store.append(None)

    @staticmethod
    def _concat(left: "ArrayLike", right: "ArrayLike") -> "ArrayLike":
        from mlx import core as mx

        return mx.concatenate([left, right], axis=-2)
