from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm

try:
    from streaming_vlm.data.lmm_dataset import LMMDataset, DataArguments  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LMMDataset = None  # type: ignore
    DataArguments = None  # type: ignore

from streaming_vlm_mlx.data.dataset import ConversationTurn, StreamingRecord

TIME_PATTERN = re.compile(r"Time=(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s")


def extract_turn_text(message: dict) -> str:
    pieces: List[str] = []
    for element in message.get("content", []):
        if isinstance(element, dict) and element.get("type") == "text":
            pieces.append(element.get("text", ""))
        elif isinstance(element, dict) and element.get("type") == "text_stream":
            stream = element.get("text_stream", [])
            words = [item[-1] for item in stream if isinstance(item, (list, tuple)) and len(item) >= 3]
            if words:
                pieces.append(" ".join(words))
    return " ".join(pieces).strip()


def parse_time_range(text: str) -> tuple[float, float]:
    match = TIME_PATTERN.search(text)
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def conversation_to_record(conversation: List[dict]) -> StreamingRecord:
    previous_text = ""
    turns: List[ConversationTurn] = []
    i = 0
    while i < len(conversation):
        user_msg = conversation[i]
        if user_msg.get("role") != "user":
            i += 1
            continue
        assistant_msg = conversation[i + 1] if i + 1 < len(conversation) else None
        for element in user_msg.get("content", []):
            if isinstance(element, dict) and "previous" in element:
                previous_text = element.get("previous", previous_text)
        user_text = extract_turn_text(user_msg)
        assistant_text = extract_turn_text(assistant_msg) if assistant_msg else ""
        time_start, time_end = parse_time_range(user_text)
        turns.append(
            ConversationTurn(
                time_start=time_start,
                time_end=time_end,
                user_text=user_text,
                assistant_text=assistant_text,
            )
        )
        i += 2
    return StreamingRecord(previous_text=previous_text, turns=turns)


def iter_conversations(
    annotation_paths: Sequence[str],
    *,
    max_samples: int | None = None,
) -> Iterable[StreamingRecord]:
    if (
        LMMDataset is not None
        and DataArguments is not None
        and os.environ.get("STREAMING_VLM_MLX_USE_LMMDATASET") == "1"
    ):
        data_args = DataArguments(train_annotation_paths=list(annotation_paths))
        kwargs = dict(train_annotation_paths=data_args.train_annotation_paths)
        dataset = LMMDataset(
            **kwargs,
            processor=None,
            return_conversation=False,
            with_context=data_args.with_context,
        )
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for idx in range(total):
            convo = dataset.load_conversation(idx)
            convo = [{"role": "previous text", "content": ""}] + convo
            yield conversation_to_record(convo)
    else:
        count = 0
        for path in annotation_paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if max_samples is not None and count >= max_samples:
                        return
                    if not line.strip():
                        continue
                    convo = json.loads(line)
                    yield conversation_to_record(convo)
                    count += 1


def write_records(records: Iterable[StreamingRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "previous_text": record.previous_text,
                "turns": [
                    {
                        "time_start": turn.time_start,
                        "time_end": turn.time_end,
                        "user_text": turn.user_text,
                        "assistant_text": turn.assistant_text,
                    }
                    for turn in record.turns
                ],
            }
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LMMDataset annotations to MLX streaming records.")
    parser.add_argument("--annotations", nargs="+", required=True, help="Paths to *_with_seeks.jsonl files.")
    parser.add_argument("--output", required=True, help="Output JSONL path for StreamingRecord entries.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on number of samples.")
    parser.add_argument("--data-root", default=None, help="Optional override for DATASET_PATH.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root:
        os.environ["DATASET_PATH"] = args.data_root
    output_path = Path(args.output)
    records = iter_conversations(args.annotations, max_samples=args.max_samples)
    write_records(tqdm(records, desc="Converting"), output_path)


if __name__ == "__main__":
    main()
