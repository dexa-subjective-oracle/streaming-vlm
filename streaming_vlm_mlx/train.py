from __future__ import annotations

import argparse
from typing import Iterable

from mlx import core as mx
from mlx import optimizers as optim

from streaming_vlm_mlx.data.batching import Batch, collate_records
from streaming_vlm_mlx.data.dataset import StreamingRecord, load_records
from streaming_vlm_mlx.inference.streaming_args import StreamingArgs
from streaming_vlm_mlx.models.tiny_decoder import TinyDecoderConfig, TinyStreamingDecoder
from streaming_vlm_mlx.utils.loss import language_modeling_loss


def _loss_fn(model: TinyStreamingDecoder, batch: Batch) -> mx.array:
    outputs = model(batch.input_ids, streaming_args=StreamingArgs(pos_mode="append"))
    return language_modeling_loss(outputs["logits"], batch.labels)


def train_loop(
    model: TinyStreamingDecoder,
    optimizer: optim.Optimizer,
    records: Iterable[StreamingRecord],
    pad_token_id: int,
    bos_token_id: int,
    epochs: int,
) -> None:
    records = list(records)
    grad_fn = mx.value_and_grad(_loss_fn)
    global_step = 0
    for epoch in range(epochs):
        for record in records:
            batch = collate_records([record], pad_token_id, bos_token_id)
            loss, grads = grad_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            if global_step % 20 == 0:
                print(f"epoch={epoch} step={global_step} loss={float(loss):.4f}")
            global_step += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny streaming decoder with MLX.")
    parser.add_argument("--data", nargs="+", required=True, help="Path(s) to JSONL streaming records.")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--bos_token_id", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TinyDecoderConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = TinyStreamingDecoder(config)
    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    records = list(load_records(args.data))
    train_loop(
        model,
        optimizer,
        records,
        args.pad_token_id,
        args.bos_token_id,
        args.epochs,
    )


if __name__ == "__main__":
    main()
