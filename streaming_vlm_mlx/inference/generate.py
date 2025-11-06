from __future__ import annotations

from typing import Callable, Dict, Optional

from mlx import core as mx

from .cache import StreamingCache
from .streaming_args import StreamingArgs


def greedy_decode_step(
    model: Callable[[Dict[str, mx.array]], Dict[str, mx.array]],
    inputs: Dict[str, mx.array],
    cache: StreamingCache,
    streaming_args: StreamingArgs,
) -> Dict[str, mx.array]:
    outputs = model(inputs)
    next_token_logits = outputs["logits"][:, -1, :]
    next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
    new_ids = mx.concatenate([inputs["input_ids"], next_token], axis=-1)
    return {
        "sequences": new_ids,
        "logits": outputs["logits"],
        "cache": cache,
        "next_token": next_token,
    }
