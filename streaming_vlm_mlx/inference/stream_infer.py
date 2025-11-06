from __future__ import annotations

import argparse
import json
from pathlib import Path

from streaming_vlm_mlx.data.dataset import load_records
from streaming_vlm_mlx.data.tokenization import tokenize_record
from streaming_vlm_mlx.inference.loop import run_streaming_generation
from streaming_vlm_mlx.models.tiny_decoder import TinyDecoderConfig, TinyStreamingDecoder
from streaming_vlm_mlx.utils.checkpoint import read_checkpoint
from streaming_vlm_mlx.utils.tokenizer import StreamingTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming inference with an MLX checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by train.py.")
    parser.add_argument("--data", required=True, help="Converted JSONL file with streaming records.")
    parser.add_argument("--index", type=int, default=0, help="Which record to run.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer override. Defaults to value stored in checkpoint.")
    parser.add_argument("--prompt-text", default="\n", help="Text used to seed assistant decoding per turn.")
    parser.add_argument("--vision-token-text", default=None, help="Placeholder text appended to user turns.")
    parser.add_argument("--output-json", default=None, help="Optional path to dump JSONL results.")
    parser.add_argument("--output-vtt", default=None, help="Optional path to dump WebVTT subtitles.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional cap per turn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_checkpoint(args.checkpoint)
    extra = payload.get("extra", {})
    config_dict = extra.get("config", {})
    tokenizer_name = args.tokenizer or extra.get("tokenizer")
    if tokenizer_name is None:
        raise ValueError("Tokenizer identifier missing; pass --tokenizer.")
    tokenizer = StreamingTokenizer.from_pretrained(tokenizer_name)
    config = TinyDecoderConfig(
        vocab_size=config_dict.get("vocab_size", tokenizer.vocab_size),
        hidden_size=config_dict.get("hidden_size", 512),
        num_layers=config_dict.get("num_layers", 8),
        num_heads=config_dict.get("num_heads", 8),
        max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
        vision_token_id=config_dict.get("vision_token_id"),
    )
    model = TinyStreamingDecoder(config)
    model.update(payload["model"])

    records = list(load_records([args.data]))
    if args.index >= len(records):
        raise IndexError(f"Record index {args.index} out of range (total {len(records)})")
    record_text = records[args.index]
    record = tokenize_record(
        record_text,
        tokenizer,
        vision_token_text=args.vision_token_text,
    )

    prompt_tokens = tokenizer.encode(args.prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt text could not be tokenized")
    prompt_token_id = prompt_tokens[0]

    generations = run_streaming_generation(
        model,
        record,
        prompt_token_id=prompt_token_id,
        bos_token_id=tokenizer.bos_token_id,
        max_new_tokens=args.max_new_tokens,
    )

    segments = []
    for turn, tokens in zip(record_text.turns, generations):
        decoded = tokenizer.decode(tokens[1:])  # drop prompt id
        segments.append(
            {
                "start": turn.time_start,
                "end": turn.time_end,
                "text": decoded.strip(),
            }
        )

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_json).open("w", encoding="utf-8") as handle:
            for seg in segments:
                json.dump(seg, handle, ensure_ascii=False)
                handle.write("\n")

    if args.output_vtt:
        Path(args.output_vtt).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_vtt).open("w", encoding="utf-8") as handle:
            handle.write("WEBVTT\n\n")
            for seg in segments:
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                handle.write(f"{start} --> {end}\n{seg['text']}\n\n")

    if not args.output_json and not args.output_vtt:
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            print(f"[{start} - {end}] {seg['text']}")


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


if __name__ == "__main__":
    main()
