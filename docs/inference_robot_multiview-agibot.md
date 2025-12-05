# Robot Multiview-agibot Inference Guide

This guide provides instructions on running inference with the Cosmos-Predict2.5/robot/multiview-agibot model.

We recommend first reading the [Inference Guide](./inference.md).

## Prerequisites

1. [Setup Guide](./setup.md)

## Example

Run single-GPU inference with an example asset:
```bash
python examples/robot_multiview.py -i assets/robot_multiview-agibot/0.json \
    -o outputs/robot_multiview-agibot/ \
    --base-path=assets/robot_multiview-agibot/
```

For an explanation of all the available parameters run:

```
python examples/robot_multiview.py --help
```

Multi GPU (on all samples, note the wildcard):
```bash
NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS examples/robot_multiview.py --context_parallel_size=$NUM_GPUS \
    -i assets/robot_multiview-agibot/*.json \
    --base-path=assets/robot_multiview-agibot \
    -o outputs/robot_multiview-agibot/
```

You can override parameters, such as custom input/output directories: `--input-root=<input_dir> --output-dir=<output_dir>`. You can do the same with a custom text prompt (with `--prompt`), for example:

```bash
NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS examples/robot_multiview.py --context_parallel_size=$NUM_GPUS \
    -i assets/robot_multiview-agibot/0.json \
    --base-path=assets/robot_multiview-agibot \
    -o outputs/robot_multiview-agibot/myprompt \
    --prompt "The food on the table sets on fire"
```

NOTES:
* `camera_load_create_fn` is a string that represents a python function
  (`<module>.<name>`) that returns a function that will load the associated
  camera parameters for an input sample. Refer to the file
  `cosmos_predict2/robot_multiview.py` for further details.


### Outputs

<video src="https://github.com/user-attachments/assets/4e68e36c-eb16-40c5-bc01-6b344e0e2a83" width="500" alt="robot multiview agibot example 0" controls></video>

<video src="https://github.com/user-attachments/assets/ee91ee7d-cf0e-495d-ad82-ed68274964f1" width="500" alt="robot multiview agibot example 1" controls></video>

<video src="https://github.com/user-attachments/assets/cbc22b77-c31e-4943-a336-1ad3f7049d69" width="500" alt="robot multiview agibot example 2" controls></video>
