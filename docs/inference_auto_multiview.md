# Auto Multiview Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2.5/auto/multiview models.

We recommend first reading the [Inference Guide](inference.md).

## Prerequisites

1. [Setup Guide](./setup.md)

## Example

Multiview inference requires a minimum of 8 GPUs with at least 80GB memory each.

Run multi-GPU inference with example asset:

```bash
torchrun --nproc_per_node=8 examples/multiview.py -i assets/multiview/urban_freeway.json -o outputs/multiview_video2world --inference-type=video2world
```

All variants require sample input videos. For Text2World, they are not used. For Image2World, only the first frame is used. For Video2World, the first 2 frames are used.

| Variant | Arguments |
| --- | --- |
| Text2World | `-o outputs/multiview_text2world --inference-type=text2world` |
| Image2World | `-o outputs/multiview_image2world --inference-type=image2world` |
| Video2World | `-o outputs/multiview_video2world --inference-type=video2world` |

### Outputs

#### multiview/text2world

<video src="https://github.com/user-attachments/assets/aae580f5-1379-4416-81ad-c863b51d4cf9" width="500" alt="multiview/text2world" controls></video>
