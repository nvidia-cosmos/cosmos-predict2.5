default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Install the repository
install:
  uv sync --all-groups

# Run linting and formatting
lint: setup
  pre-commit run -a || pre-commit run -a

# Run tests
test: lint

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions sentencepiece"

# Update the license
license: install
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md
  pre-commit run --files ATTRIBUTIONS.md || true

# Pre-release checks
release-check:
  just license
  pre-commit run -a --hook-stage manual

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Run the docker container
docker *args:
  # https://github.com/astral-sh/uv-docker-example/blob/main/run.sh
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it $(docker build -q .)

# Initialize the repository by auto-fixing/whitelisting all existing issues
_init:
  pre-commit run -a --hook-stage manual ruff-fix || true

  # Run twice, since formatting will break some noqa comments
  pre-commit run -a --hook-stage manual ruff-noqa || true
  pre-commit run -a --hook-stage manual ruff-format || true
  pre-commit run -a --hook-stage manual ruff-noqa || true
  pre-commit run -a --hook-stage manual ruff-format || true

  pre-commit run -a || pre-commit run -a
