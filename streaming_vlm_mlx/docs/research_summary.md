# StreamingVLM Paper Notes

## Core Ideas

- **Streaming-aware KV cache** keeps three regions:
  - Attention sink tokens (system prompt + previous text) that persist to stabilize attention.
  - Long text window (e.g., 512 tokens) for conversational memory.
  - Short vision window (e.g., 16 seconds of frames) for recent visual context.
- **Contiguous RoPE**: after evicting old tokens, shift rotary indices so remaining tokens stay in-distribution.
- **Training strategy**: use overlapped, full-attention chunks that mimic the inference retention pattern. Interleave vision and text tokens and supervise only aligned text positions (placeholder “…” when silent).
- **Data pipeline**: collect multi-hour sports broadcasts, segment with ASR + GPT rewriting/filtering, generate overlapped SFT samples plus a high-quality annealing subset focused on live actions.
- **Benchmark**: Inf-Streams-Eval with chunk (bounded) and infinite settings, measuring per-second commentary accuracy over multi-hour videos.

## MVP Implications

1. **Data alignment**: our MLX pipeline should convert existing LMMDataset outputs into the same interleaved token format, respecting attention sinks and per-second supervision.
2. **Cache management**: model must support attention sinks + configurable text/vision windows with contiguous positional updates.
3. **Tokenizer compatibility**: use the same tokenizer IDs as original datasets (Qwen2.5-VL) or a carefully designed substitute to preserve special tokens (`<|im_start|>`, `<|im_end|>`, etc.).
4. **Vision handling**: start with placeholder tokens for MVP, but leave hooks for frozen CLIP features so we can approximate the paper’s short vision window later.
5. **Evaluation**: implement chunked streaming inference with JSONL output so we can compare against references or human inspection per timestamp.
