from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from mlx import core as mx
from mlx import optimizers as optim

from streaming_vlm_mlx.data.dataset import ConversationTurn, StreamingRecord
from streaming_vlm_mlx.inference.loop import run_streaming_generation
from streaming_vlm_mlx.models.tiny_decoder import TinyDecoderConfig, TinyStreamingDecoder
from streaming_vlm_mlx.train import train_loop


CHARS = (
    ["<pad>", "<bos>"]
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + [" ", ":", "-", "=", ".", ",", "!", "\n", "~"]
)


def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab: Dict[str, int] = {}
    for idx, ch in enumerate(CHARS):
        vocab[ch] = idx
    inv_vocab = {idx: ch for ch, idx in vocab.items()}
    return vocab, inv_vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[ch] for ch in text]


def decode(tokens: List[int], inv_vocab: Dict[int, str]) -> str:
    return "".join(
        inv_vocab.get(tok, "?")
        for tok in tokens
        if inv_vocab.get(tok) not in {"<pad>", "<bos>"}
    )


def build_dataset(vocab: Dict[str, int]) -> List[StreamingRecord]:
    def user(text: str) -> List[int]:
        return encode(text, vocab)

    def assistant(text: str) -> List[int]:
        return encode("~" + text, vocab)

    records = [
        StreamingRecord(
            previous_tokens=encode("Previous clip: fast break\n", vocab),
            turns=[
                ConversationTurn(
                    time_start=0.0,
                    time_end=5.0,
                    user_tokens=user("Time=0-5 crowd builds\n"),
                    assistant_tokens=assistant("Player brings ball up the court.\n"),
                ),
                ConversationTurn(
                    time_start=5.0,
                    time_end=10.0,
                    user_tokens=user("Time=5-10 quick pass\n"),
                    assistant_tokens=assistant("Teammate cuts inside for the layup.\n"),
                ),
            ],
        ),
        StreamingRecord(
            previous_tokens=encode("Previous clip: defensive stop\n", vocab),
            turns=[
                ConversationTurn(
                    time_start=12.0,
                    time_end=18.0,
                    user_tokens=user("Time=12-18 reset offense\n"),
                    assistant_tokens=assistant("Guard signals the play and waits.\n"),
                ),
                ConversationTurn(
                    time_start=18.0,
                    time_end=24.0,
                    user_tokens=user("Time=18-24 corner action\n"),
                    assistant_tokens=assistant("Wing rotates out and drains the three.\n"),
                ),
            ],
        ),
    ]
    return records


def run_demo() -> None:
    vocab, inv_vocab = build_vocab()
    pad_token_id = vocab["<pad>"]
    bos_token_id = vocab["<bos>"]
    records = build_dataset(vocab)

    config = TinyDecoderConfig(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=4,
        num_heads=4,
    )
    model = TinyStreamingDecoder(config)
    optimizer = optim.AdamW(learning_rate=1e-3)

    import os

    epochs = int(os.getenv("EPOCHS", "150"))
    print("Training tiny streaming decoder on toy dataset...")
    train_loop(model, optimizer, records, pad_token_id, bos_token_id, epochs=epochs)

    print("\nRunning streaming generation on first record:")
    generations = run_streaming_generation(
        model,
        records[0],
        prompt_token_id=vocab["~"],
        bos_token_id=bos_token_id,
    )
    for idx, tokens in enumerate(generations):
        text = decode(tokens, inv_vocab)
        print(f"Turn {idx}: {text}")


if __name__ == "__main__":
    run_demo()
