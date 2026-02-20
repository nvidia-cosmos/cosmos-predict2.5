# Troubleshooting

## Issues

Also, check GitHub Issues for the [repository](https://github.com/orgs/nvidia-cosmos/repositories).

### Changing the cache directory
We download packages and checkpoints to the home directory by default. To change this, you can set the following variables.
This is also needed if you get `No space left on device` errors when setting up or downloading checkpoints.
```shell
export UV_CACHE_DIR=<new_dir>/.cache
export PIP_CACHE_DIR=<new_dir>/.cache
export HF_HOME=<new_dir>/checkpoints
```

### Missing Python.h

Error message: `fatal error: Python.h: No such file or directory`

This is fixed by [installing uv managed python](https://docs.astral.sh/uv/guides/install-python/#installing-python):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
```

Re-install the package:

```shell
rm -rf .venv
uv sync --extra=cu128
```

### CUDA driver version insufficient

**Fix:** Update NVIDIA drivers to latest version compatible with CUDA [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)

Check driver compatibility:

```shell
nvidia-smi | grep "CUDA Version:"
```

### PYTHONPATH conflicts in NVIDIA containers

When using `nvcr.io/nvidia/pytorch:25.xx-py3` containers, you will need to unset `PYTHONPATH` to be compatible with Python 3.10:

```shell
unset PYTHONPATH
```

### Out of Memory (OOM) errors

**Fix:** Use 2B models instead of 14B, multi-GPU, or reduce batch size/resolution

## Guide

### Logs

Logs are saved to `<output_dir>/*.log`.

### Profiling

To profile, pass the `--profile` flag. A [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) profile will be exported to `<output_dir>/profile.pyisession`.

View the profile:

```shell
pyinstrument --load=<output_dir>/profile.pyisession
```

Export the profile:

```shell
pyinstrument --load=<output_dir>/profile.pyisession -r html -o <output_dir>/profile.html
```

See [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html).
