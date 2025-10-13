# Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2.5/base models.

<p align="center">
  <img width="500" alt="cosmos-predict-diagram" src="https://github.com/user-attachments/assets/8f436cdd-3d04-46ea-b333-d8e9ccdc6d9c">
</p>

## Prerequisites

1. [Setup Guide](setup.md)

## Example

Run inference with example asset:

```bash
python examples/inference.py -i assets/base/robot_pouring.json -o outputs/base_video2world --inference-type=video2world
```

To enable multi-GPU inference with 8 GPUs, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):

```bash
torchrun --nproc_per_node=8 examples/inference.py -i assets/base/robot_pouring.json -o outputs/base_video2world --inference-type=video2world
```

| Variant | Arguments |
| --- | --- |
| Text2World | `-o outputs/base_text2world --inference-type=text2world` |
| Image2World | `-o outputs/base_image2world --inference-type=image2world` |
| Video2World | `-o outputs/base_video2world --inference-type=video2world` |

To run all example assets:

```shell
torchrun --nproc_per_node=8 examples/inference.py -i assets/base/*.json -o outputs/base
```

To change the model, pass `--model`:

| Size | Arguments |
| --- | --- |
| 2B | `--model 2B/post-trained` |
| 14B | Coming soon! |

To see all available options:

```bash
python examples/inference.py --help
```

Parameters are specified as json:

```jsonc
{
  // Inference type: text2world, image2world, video2world
  "inference_type": "video2world",
  // Sample name
  "name": "robot_pouring",
  // Input prompt
  "prompt": "A robotic arm, primarily white with black joints and cables...",
  // Path to the input image/video file (not needed for text2world)
  "input_path": "robot_pouring.mp4"
}
```

### Outputs

#### text2world/snowy_stop_light

<video src="https://github.com/user-attachments/assets/6063060e-7873-4d56-99d1-3f231c535627" width="500" alt="text2world/snowy_stop_light" controls></video>

#### image2world/robot_welding

<video src="https://github.com/user-attachments/assets/260cf600-7b35-408e-9c33-2bb1a23cec2f" width="500" alt="image2world/robot_welding" controls></video>

#### video2world/sand_mining

<video src="https://github.com/user-attachments/assets/ca440ba6-50b1-4b63-b590-063e5e942b6a" width="500" alt="video2world/sand_mining" controls></video>

## Tips

### Multi-GPU

Context parallelism distributes inference across multiple GPUs, with each GPU generating a subset of the video frames.

* The number of GPUs should ideally be a divisor of the number of frames in the generated video.
* All GPUs should have the same model capacity and memory.
* Context parallelism works best with the 14B model where memory constraints are significant.
* Requires NCCL support and proper GPU interconnect for efficient communication.
* Significant speedup for video generation while maintaining the same quality.

### Prompt Engineering

For best results with Cosmos models, create detailed prompts that emphasize physical realism, natural laws, and real-world behaviors. Describe specific objects, materials, lighting conditions, and spatial relationships while maintaining logical consistency throughout the scene.

Incorporate photography terminology like composition, lighting setups, and camera settings. Use concrete terms like "natural lighting" or "wide-angle lens" rather than abstract descriptions, unless intentionally aiming for surrealism. Include negative prompts to explicitly specify undesired elements.

The more grounded a prompt is in real-world physics and natural phenomena, the more physically plausible and realistic the generated image will be.
