# Troubleshooting

## Issues

Also check [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-predict2.5/issues). If filing a new issue, please upload the full output directory.

### CUDA driver version insufficient

**Fix:** Update NVIDIA drivers to latest version compatible with CUDA [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)

Check driver compatibility:

```shell
nvidia-smi | grep "CUDA Version:"
```

### Out of Memory (OOM) errors**

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
