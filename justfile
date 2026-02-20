default:
  just --list

package_name := `echo cosmos_* | tr '_' '-'`
module_name := `echo cosmos_*`
short_name := `for dir in cosmos_*; do echo "${dir#cosmos_}"; done`

default_max_gpus := if arch() == "aarch64" {
  "4"
} else {
  "8"
}

# Setup the repository
setup:

default_cuda_name := if arch() == "aarch64" {
  "cu130"
} else {
  "cu128"
}

# Install the repository
install cuda_name=default_cuda_name *args: setup
  echo {{cuda_name}} > .cuda-name
  uv sync --extra={{cuda_name}} {{args}}

# Run uv sync
_uv-sync *args: setup
  if [ ! -f .cuda-name ]; then \
    echo {{default_cuda_name}} > .cuda-name; \
  fi
  uv sync --extra=$(cat .cuda-name) {{args}}

# Run a command in the package environment
run *args: _uv-sync
  uv run --no-sync {{args}}

# Setup pre-commit
_pre-commit-setup: setup
  uv tool install "pre-commit>=4.3.0"
  pre-commit install -c .pre-commit-config-base.yaml

# Run pre-commit
pre-commit *args: _pre-commit-setup
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

pytest_args := '-vv --instafail --durations=5 --force-regen'

# Run a single test
test-single name *args: _uv-sync
  uv run --no-sync pytest --manual --capture=no {{pytest_args}} {{args}} {{name}}

# Run CPU tests
test-cpu *args: _uv-sync
  uv run --no-sync pytest --num-gpus=0 -n logical --maxprocesses=16 --levels=0 {{pytest_args}} {{args}}

# Run GPU tests
test-gpu *args: _uv-sync
  #!/usr/bin/env bash
  set -euxo pipefail
  export MAX_GPUS={{default_max_gpus}}
  AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
  for num_gpus in 1 $MAX_GPUS; do
    if [ $num_gpus -gt $AVAILABLE_GPUS ]; then
      break
    fi
    args="{{pytest_args}} {{args}}"
    if [ $num_gpus -ne 1 ]; then
      # Only run coverage for single-GPU tests.
      # All multi-GPU tests should have a corresponding single-GPU smoke test, which has full coverage.
      args="$args --no-cov"
    fi
    uv run --no-sync pytest --num-gpus=$num_gpus -n logical --levels=0 $args
  done

# Run tests
test *args: pyrefly (test-cpu args) (test-gpu args)

coverage_args := '--cov-append --cov-report= --cov=' + module_name

# Initialize coverage
_coverage-init:
  rm -rf outputs/coverage

# Run tests with coverage
test-coverage *args: _coverage-init (test coverage_args args) (run 'coverage' 'xml')

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
_pip-licenses *args: install
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: _licensecheck _pip-licenses

# Run link-check
_link-check *args:
  pre-commit run -a --hook-stage manual link-check {{args}}

# Pre-release checks
release-check: license _link-check

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Run the docker container
_docker build_args='' run_args='':
  #!/usr/bin/env bash
  set -euxo pipefail
  docker build {{build_args}} .
  image_tag=$(docker build {{build_args}} -q .)
  docker run \
    -it \
    --runtime=nvidia \
    --ipc=host \
    --rm \
    -v .:/workspace \
    -v /workspace/.venv \
    -v /root/.cache:/root/.cache \
    -e HF_TOKEN="$HF_TOKEN" \
    {{run_args}} \
    $image_tag

# Run the CUDA 12.8 docker container.
docker-cu128 *run_args: (_docker '--build-arg=CUDA_NAME=cu128 --build-arg=BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04' run_args)

# Run the CUDA 13.0 docker container.
docker-cu130 *run_args: (_docker '-f docker/nightly.Dockerfile' run_args)
