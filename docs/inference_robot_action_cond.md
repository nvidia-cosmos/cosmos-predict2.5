# Robot Action-Conditioned Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2.5/robot/action-cond models.

We recommend first reading the [Inference Guide](inference.md).

## Prerequisites

1. [Setup Guide](./setup.md)

## Example

Action conditioned inference does not yet support multi-GPU.

Run inference with example asset:

```bash
python examples/action_conditioned.py -i assets/action_conditioned/basic/inference_params.json -o outputs/action_conditioned/basic
```

## Configuration

The configuration is split into two parts:

1. **Setup Arguments** (`ActionConditionedSetupArguments`): Model-related configuration that typically stays the same across runs
   - `model`: Model variant to use (default: robot/multiview)
   - `context_parallel_size`: Context parallelism is not supported for action conditioned model. Set context_parallel_size to 1.
   - `output_dir`: Output directory for results
   - `config_file`: Model configuration file

2. **Inference Arguments** (`ActionConditionedInferenceArguments`): Per-run parameters that can vary
   - `input_root`: Root directory containing videos and annotations
   - `input_json_sub_folder`: Subdirectory containing JSON annotations
   - `chunk_size`: Action chunk size for processing
   - `guidance`: Guidance scale for generation
   - `action_load_fn`: Function to load action data
   - And many more...

## JSON Configuration File

Create a JSON file with your inference parameters:

```json
{
  "name": "my_inference",
  "input_root": "/path/to/input/data",
  "input_json_sub_folder": "annotations",
  "save_root": "/path/to/output",
  "chunk_size": 12,
  "guidance": 7,
  "camera_id": "base",
  "start": 0,
  "end": 100,
  "action_load_fn": "cosmos_predict2.action_conditioned.load_default_action_fn"
}
```

## Custom Action Loading

To use a custom action loading function, implement a function following this signature:

```python
def custom_action_load_fn():
    def load_fn(json_data: dict, video_path: str, args: ActionConditionedInferenceArguments) -> dict:
        # Your custom action loading logic here
        return {
            "actions": actions,  # numpy array of actions
            "initial_frame": img_array,  # first frame
            "video_array": video_array,  # full video
            "video_path": video_path,
        }
    return load_fn
```

Then specify it in your JSON config:

```json
{
  "action_load_fn": "my_module.custom_action_load_fn"
}
```

## Outputs
<video src="https://github.com/user-attachments/assets/35e3f671-5d0b-41c1-b3a1-f37eaf216f43" width="500" alt="action_conditioned" controls></video>

<video src="https://github.com/user-attachments/assets/b3a86f38-12dd-49c9-bda8-93a6584c5699" width="500" alt="action_conditioned" controls></video>

<video src="https://github.com/user-attachments/assets/8d598da3-6623-4cae-8980-64da81e3b54b" width="500" alt="action_conditioned" controls></video>
