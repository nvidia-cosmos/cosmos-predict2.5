# Diffusers Inference Guide

This guide explains how to use `scripts/diffusers_inference.py` to run
Cosmos-Predict2.5 Diffusers pipelines for Text2World, Image2World, and
Video2World. Review the [Inference Guide](inference.md) for broader context.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), via:
    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Ensure sample assets (under `assets/base`) or your custom media/prompt files are accessible.

## Script Overview

`scripts/diffusers_inference.py` calls [`Cosmos2_5_PredictBasePipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos#diffusers.Cosmos2_5_PredictBasePipeline). Provide either a JSON bundle via `--input_path assets/base/sample.json` (recommended) or individual CLI flags. CLI arguments always override values loaded from the JSON file.

| Flag | Purpose |
| --- | --- |
| `--input_path` | Points either to a JSON asset (ending with `.json`) that contains `input_path`, optional `prompt_path`/`prompt`, and `negative_prompt` fields, or directly to conditioning media (image/video). Paths referenced inside JSON configs are resolved relative to the config file. |
| `--override-visual-input` | When `--input_path` references a JSON config, this flag overrides the media path defined inside that config. |
| `--output_path` | Output file (PNG for images, MP4 for videos). The script creates parent directories as needed. |
| `--num_output_frames` | Sets output length. Use `1` for image output and `>1` (default & recommended: `93`) for world (video) generation. |
| `--prompt` / `--prompt_path` | Overrides the prompt. |
| `--negative_prompt` / `--negative_prompt_path` | Provides custom safety guardrails. Defaults to a quality-focused negative prompt. |
| `--model_id` | Use nvidia/Cosmos-Predict2.5-14B for 14B or nvidia/Cosmos-Predict2.5-2B (default) for 2B base model |
| `--revision` | Use `diffusers/base/pre-trained` for pre-trained variant or `diffusers/base/post-trained` (default) for post-trained model variant |
| `--device`, `--device_map`, `--seed`, `--num_steps` | Advanced controls for model variant, placement, determinism, and sampling steps. |

Run `./scripts/diffusers_inference.py --help` to see the full Tyro-generated documentation.

## Ready-to-Run Asset Examples

Run these commands from the repository root. Each example targets a JSON asset that already contains the appropriate prompt and media reference.

### Text2World

```bash
./scripts/diffusers_inference.py \
  --input_path assets/base/bus_terminal_long.json \
  --output_path outputs/text2world_bus_terminal.mp4
```

### Image2World

```bash
./scripts/diffusers_inference.py \
  --input_path assets/base/robot_welding.json \
  --output_path outputs/image2world_robot_welding.mp4
```

### Video2World

```bash
./scripts/diffusers_inference.py \
  --input_path assets/base/sand_mining.json \
  --output_path outputs/video2world_sand_mining.mp4
```

### Text2Image

```bash
./scripts/diffusers_inference.py \
  --input_path assets/base/bus_terminal_long.json \
  --num_output_frames 1 \
  --output_path outputs/text2image_bus_terminal.png
```

## Supplying Custom Prompts

You can skip JSON configs and drive the pipeline directly from the CLI.

### Video2World

```bash
TEXT_PROMPT="The robot pours liquid into the cup which ignites into flames"

./scripts/diffusers_inference.py \
  --input_path assets/base/robot_pouring.mp4 \
  --prompt "$TEXT_PROMPT" \
  --output_path outputs/robot_pouring_overflow.mp4
```

## Tips

- `--input_path some_asset.json` assets can include `prompt_path` files; use this to separate long prompts from the CLI. Combine with `--override-visual-input custom.mp4` to reuse prompts but swap the conditioning media.
- Set `--seed` for reproducible generations. Omit it for additional diversity.
- Pass `--device_map balanced` (or other Hugging Face placements) if the model size requires multi-GPU sharding.
