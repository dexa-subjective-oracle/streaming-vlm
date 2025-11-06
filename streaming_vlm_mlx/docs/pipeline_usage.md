# StreamingVLM-MLX Pipeline Usage

## 1. Dataset Preparation

1. Download the annotation files referenced in `scripts/sft_stage_1.sh` (`Inf-Stream-Train`, `Live-WhisperX-526K`, etc.) under a shared root (e.g., `DATASET_PATH=/datasets/Inf-Stream-Train`).
2. Convert the JSONL annotations into MLX-friendly records:

   ```bash
   PYTHONPATH=. python streaming_vlm_mlx/scripts/convert_dataset.py \
     --annotations $DATASET_PATH/train_s12w24_with_seeks.jsonl \
     --output data/converted/train.jsonl \
     --max-samples 1000
   ```

   The output file contains per-chunk timestamps and raw user/assistant text; tokenization happens later.

## 2. Training

- **Quick toy pass** (tiny character model, fast sanity check):

  ```bash
  PYTHONPATH=. python streaming_vlm_mlx/train.py \
    --data streaming_vlm_mlx/examples/data/toy_record.jsonl \
    --tokenizer bert-base-uncased \
    --hidden_size 64 --num_layers 2 --num_heads 4 \
    --epochs 2 \
    --save-path streaming_vlm_mlx/examples/checkpoints/toy.pkl
  ```

- **Inf-Stream subset (recommended MVP run)**:

  ```bash
  PYTHONPATH=. python streaming_vlm_mlx/train.py \
    --data data/converted/train_inf_200.jsonl \
    --tokenizer Qwen/Qwen2.5-VL-7B-Instruct \
    --hidden_size 256 --num_layers 4 --num_heads 4 \
    --epochs 5 \
    --learning_rate 3e-3 \
    --vision_token_text "<|vision_place|>" \
    --vision_token_id 151656 \
    --save-path checkpoints/streaming_tiny.pkl
  ```

The checkpoint stores both model weights and the configuration (hidden size, number of layers, tokenizer, etc.) needed for inference.

## 3. Streaming Inference

```bash
PYTHONPATH=. python streaming_vlm_mlx/inference/stream_infer.py \
  --checkpoint checkpoints/streaming_tiny.pkl \
  --data data/converted/train.jsonl \
  --index 0 \
  --prompt-text "<|assistant|>" \
  --output-json outputs/sample.jsonl \
  --output-vtt outputs/sample.vtt
```

This renders per-chunk commentary. Swap `--index` to iterate through additional chunks or compose your own `StreamingRecord` for a new 360p clip.

## 4. Toy Validation

The commands above include a toy dataset path plus the Inf-Stream subset. Use the toy run for a fast end-to-end check, then switch to the Inf-Stream configuration for the real MVP.
