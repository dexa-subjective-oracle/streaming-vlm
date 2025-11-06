from __future__ import annotations

from typing import List

from mlx import core as mx

from streaming_vlm_mlx.data.dataset import StreamingRecord
from streaming_vlm_mlx.inference.cache import StreamingCache
from streaming_vlm_mlx.inference.streaming_args import StreamingArgs
from streaming_vlm_mlx.models.tiny_decoder import TinyStreamingDecoder


def run_streaming_generation(
    model: TinyStreamingDecoder,
    record: StreamingRecord,
    *,
    prompt_token_id: int,
    bos_token_id: int,
    max_new_tokens: int | None = None,
) -> List[List[int]]:
    cache = StreamingCache()
    streaming_args = StreamingArgs(pos_mode="append")
    generations: List[List[int]] = []

    bos = mx.array([[bos_token_id]], dtype=mx.int32)
    model(bos, cache=cache, streaming_args=streaming_args)

    if record.previous_tokens:
        prev = mx.array(record.previous_tokens, dtype=mx.int32).reshape(1, -1)
        model(prev, cache=cache, streaming_args=streaming_args)

    for turn in record.turns:
        user_tokens = mx.array(turn.user_tokens, dtype=mx.int32).reshape(1, -1)
        model(user_tokens, cache=cache, streaming_args=streaming_args)
        start = mx.array([[prompt_token_id]], dtype=mx.int32)
        target_tokens = max_new_tokens or len(turn.assistant_tokens)
        generated = model.generate(
            start,
            max_new_tokens=target_tokens,
            cache=cache,
            streaming_args=streaming_args,
        )
        generations.append(generated[0].tolist())
    return generations
