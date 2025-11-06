# MLX Streaming MVP Requirements

## Goal
Given a 360p sports clip, generate per-chunk natural language commentary aligned to timestamps using an Apple silicon–friendly MLX model scaffolded in `streaming_vlm_mlx/`.

## High-Level Flow
1. **Dataset conversion**
   - Load existing sports commentary annotations via `streaming_vlm.data.lmm_dataset.LMMDataset`.
   - Export each sample to a simplified `StreamingRecord` JSONL file in `streaming_vlm_mlx` format (interleaved text, optional vision placeholder tokens).
   - Cache clip metadata (video path, timestamps) for inference playback.
2. **Tokenizer integration**
   - Reuse Qwen tokenizer (`Qwen2.5-VL` via `AutoProcessor`) to ensure compatibility with downloaded data.
   - Bridge HF tensors → NumPy/MLX arrays while preserving special tokens (`<|im_start|>`, `<|vision_start|>`, etc.).
3. **Model & training**
   - Extend `TinyStreamingDecoder` with:
     - Configurable attention/text/vision windows.
     - Contiguous RoPE updates.
     - Optional vision feature projection (placeholder first iteration).
   - Implement checkpoint save/load.
   - Provide a CLI to start training from the converted JSONL dataset.
4. **Streaming inference**
   - CLI that:
     - Loads a trained checkpoint.
     - Iterates chunk metadata (mirroring LMMDataset streaming order).
     - Generates commentary per chunk while pruning cache.
     - Saves output as WebVTT/JSON for inspection.
5. **Validation**
   - Run pipeline on a short 360p clip and compare commentary against ground truth or transcripts.
   - Record timing, memory usage, and sample outputs.

## Key Technical Tasks
- Write `scripts/convert_dataset.py` leveraging LMMDataset to emit MLX-friendly records.
- Implement tokenizer adapter using Qwen `AutoProcessor`.
- Enhance `TinyStreamingDecoder` with RoPE shifting, attention sinks, and hooking for vision tokens.
- Build `train_mlx.py` and `infer_stream.py` CLIs around the MLX core.
- Provide configuration files (YAML/JSON) to set window sizes, tokenizer paths, dataset roots.
- Document full pipeline in README with commands.

## Risks / Mitigations
- **Tokenizer mismatch**: ensure special token IDs align by directly using HF tokenizer; add unit tests.
- **Large dataset conversion time**: support filtering/subsampling (e.g., only a few games) for quick iteration.
- **Performance**: start with smaller model to verify correctness; profile and optimize later.
