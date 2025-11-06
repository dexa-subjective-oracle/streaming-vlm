from __future__ import annotations

from mlx import core as mx
from mlx import nn

from streaming_vlm_mlx.data.batching import IGNORE_INDEX


def language_modeling_loss(logits: mx.array, labels: mx.array) -> mx.array:
    vocab = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab)
    labels_flat = labels.reshape(-1)
    mask = labels_flat != IGNORE_INDEX
    mask_f = mask.astype(mx.float32)
    safe_labels = mx.where(mask, labels_flat, mx.zeros_like(labels_flat))
    log_probs = nn.log_softmax(logits_flat, axis=-1)
    gathered = mx.take_along_axis(log_probs, safe_labels.reshape(-1, 1), axis=-1).squeeze(-1)
    total = mx.maximum(mx.sum(mask_f), mx.array(1.0, dtype=mx.float32))
    loss = -mx.sum(gathered * mask_f) / total
    return loss
