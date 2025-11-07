default:
  just --list

package_name := `echo cosmos_* | tr '_' '-'`
module_name := `echo cosmos_*`

# Setup the repository
setup:
  uv tool install "pre-commit>=4.3.0"
  pre-commit install -c .pre-commit-config-base.yaml

default_cuda_name := "cu128"

# Install the repository
install cuda_name=default_cuda_name *args:
  echo {{cuda_name}} > .cuda-name
  uv sync --extra={{cuda_name}} {{args}}

# Run uv sync
_uv-sync *args:
  if [ ! -f .cuda-name ]; then \
    echo {{default_cuda_name}} > .cuda-name; \
  fi
  uv sync --extra=$(cat .cuda-name) {{args}}

# Run pre-commit
pre-commit *args: setup
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run pyrefly with the default config
_pyrefly *args: _uv-sync
  uv run --no-sync pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

# Run pyrefly with the src config
_pyrefly-src *args: _uv-sync
  uv run --no-sync pyrefly check -c pyrefly-src.toml --output-format=min-text {{args}}

# Run pyrefly
pyrefly *args: (_pyrefly args) (_pyrefly-src args)

# Run pyrefly and whitelist all errors
pyrefly-ignore *args: (pyrefly '--suppress-errors' args)

# Run linting and formatting
lint: pre-commit

# Test the install command
test-install:
  rm -f .cuda-name
  uv sync -q
  # Expect: "CUDA extra not installed..."
  -uv run --no-sync python -c "import {{module_name}}"
  just -f "{{source_file()}}" -d "$(pwd)" _uv-sync
  uv run --no-sync python -c "import {{module_name}}"

# Run a single test
test-single name *args: _uv-sync
  uv run --no-sync pytest --capture=no {{args}} {{name}}

# Run CPU tests
test-cpu *args: _uv-sync
  uv run --no-sync pytest --num-gpus=0 -n logical --maxprocesses=16 --levels=0 {{args}}

# Run 1-GPU tests
_test-gpu-1 *args: _uv-sync
  uv run --no-sync pytest --num-gpus=1 -n logical --levels=0 {{args}}

# Run 8-GPU tests
_test-gpu-8 *args: _uv-sync
  uv run --no-sync pytest --num-gpus=8 -n logical --levels=0 {{args}}

# Run GPU tests
test-gpu *args: (_test-gpu-1 args) (_test-gpu-8 args)

# Run custom pytest command
_pytest *args: _uv-sync
  uv run --no-sync pytest {{args}}

# Run tests
test *args: pyrefly (test-cpu args) (test-gpu args)

# List tests
test-list *args: _uv-sync
  uv run --no-sync pytest --collect-only -q {{args}}

# Print profile report
profile-print filename *args:
  uvx pyinstrument --load={{filename}} {{args}}

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions sentencepiece"

# Run licensecheck
_licensecheck *args:
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero {{args}}

# Run pip-licenses
_pip-licenses *args:
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: install _licensecheck _pip-licenses

# Run link-check
_link-check *args:
  pre-commit run -a --hook-stage manual link-check {{args}}

# Pre-release checks
release-check: license _link-check

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Run the docker container
_docker cuda_name base_image:
  #!/usr/bin/env bash
  set -euxo pipefail
  build_args="--build-arg=CUDA_NAME={{cuda_name}} --build-arg=BASE_IMAGE={{base_image}}"
  docker build $build_args .
  image_tag=$(docker build $build_args -q .)
  docker run \
    -it \
    --gpus all \
    --rm \
    -v .:/workspace \
    -v /workspace/.venv \
    $image_tag

# Run the CUDA 12.8 docker container.
docker-cu128: (_docker 'cu128' 'nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04')

# Run the CUDA 13.0 docker container.
docker-cu130: (_docker 'cu130' 'nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04')
