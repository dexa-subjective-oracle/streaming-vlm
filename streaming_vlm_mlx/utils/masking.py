from __future__ import annotations

from typing import Optional

from mlx import core as mx


NEG_INF = -1e9


def full_causal_mask(seq_len: int) -> mx.array:
    tril = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.float32))
    neg = mx.ones_like(tril) * NEG_INF
    mask = mx.where(tril == 1, mx.zeros_like(tril), neg)
    return mask.reshape(1, 1, seq_len, seq_len)


def streaming_causal_mask(past_len: int, current_len: int) -> mx.array:
    if past_len == 0:
        return full_causal_mask(current_len)
    past_block = mx.zeros((current_len, past_len), dtype=mx.float32)
    tril = mx.tril(mx.ones((current_len, current_len), dtype=mx.float32))
    neg = mx.ones_like(tril) * NEG_INF
    current_block = mx.where(tril == 1, mx.zeros_like(tril), neg)
    mask = mx.concatenate([past_block, current_block], axis=-1)
    return mask.reshape(1, 1, current_len, past_len + current_len)
