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

This file tracks outstanding work while we iterate on the MLX rewrite. Update the checklist as pieces land.
