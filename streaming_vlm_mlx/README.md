# streaming_vlm_mlx prototype

## MVP task tracker

- [x] Encoder/decoder utilities: causal masks, positional offsets, and streaming cache integration.
- [x] Streaming generation API that wraps caching, next-token sampling, and greedy decode.
- [x] Dataset batching helpers that turn `StreamingRecord` objects into model-ready tensors (input ids, labels, masks).
- [x] Minimal training loop using MLX optimizers and gradient updates on the tiny decoder.
- [x] Command-line helpers for offline training and for step-by-step streaming inference on JSONL metadata.

## Quickstart

The `examples/toy_demo.py` script builds a character-level vocabulary, fabricates a tiny dataset, trains the MLX decoder, and then runs streaming generation turn by turn.

```bash
PYTHONPATH=. EPOCHS=400 python streaming_vlm_mlx/examples/toy_demo.py
```

Key knobs:

- Use the `EPOCHS` environment variable to change training iterations (default `150`).
- Adjust `TinyDecoderConfig` inside the script for hidden size / depth tweaks.
- `run_streaming_generation` automatically matches each turn's target length; pass `max_new_tokens` explicitly if you want to constrain output.

## Dataset Conversion Prototype

The script `streaming_vlm_mlx/scripts/convert_dataset.py` reads annotation files consumed by `streaming_vlm.data.lmm_dataset.LMMDataset` and emits lightweight JSONL records with per-chunk timestamps and text pairs (no tokenization yet). Example:

```bash
PYTHONPATH=. python streaming_vlm_mlx/scripts/convert_dataset.py \
  --annotations $DATASET_PATH/train_s12w24_with_seeks.jsonl \
  --output data/converted/train.jsonl \
  --max-samples 100
```

Set `--data-root` if `DATASET_PATH` is not already exported. Later stages consume these JSONL records and augment them with tokenizer IDs.

## Training with a Hugging Face Tokenizer

```bash
PYTHONPATH=. python streaming_vlm_mlx/train.py \
  --data data/converted/train.jsonl \
  --tokenizer Qwen/Qwen2.5-VL-7B-Instruct \
  --save-path checkpoints/tiny-streaming.pkl \
  --epochs 2 \
  --hidden_size 512 --num_layers 8 --num_heads 8
```

Key flags:

- `--vision_token_text "<vision>"` appends a placeholder token to each user turn.
- `--vision_token_id 151656` aligns with Qwen’s `<|video_pad|>` id, enabling replacement with real vision embeddings.
- `--resume-from` loads a previous `save_checkpoint` file to continue training.

## Streaming Inference

```bash
PYTHONPATH=. python streaming_vlm_mlx/inference/stream_infer.py \
  --checkpoint checkpoints/tiny-streaming.pkl \
  --data data/converted/train.jsonl \
  --index 0 \
  --output-json outputs/sample.jsonl \
  --output-vtt outputs/sample.vtt
```

The script reconstructs the model config from the checkpoint metadata, tokenizes the selected record, and emits per-chunk commentary as JSONL or WebVTT. Use `--prompt-text` to control the seed token and `--vision_token_text` to append vision placeholders during tokenization.

Refer to `streaming_vlm_mlx/docs/pipeline_usage.md` for the full end-to-end workflow (dataset conversion → training → streaming inference) and a built-in toy validation sequence.

This file tracks outstanding work while we iterate on the MLX rewrite. Update the checklist as pieces land.
