# Coverage for Shell Script Tests

## Overview

The test scripts in `tests/docs_test/` execute shell scripts that run Python code via `torchrun` and other commands. This document explains how coverage is collected for these tests.

## How It Works

Coverage collection for subprocess/distributed execution works through several components:

### 1. `sitecustomize.py`

Located at the **root of the package** (next to `pyproject.toml`), this file is automatically imported by Python when `PYTHONPATH` includes the current directory. It calls `coverage.process_startup()` to enable coverage tracking in distributed processes spawned by `torchrun`.

### 2. Shell Script Modifications

All test shell scripts have been updated to conditionally wrap Python commands with coverage when the `COVERAGE_ENABLED` environment variable is set:

```bash
# Enable coverage subprocess tracking if coverage is enabled
COVERAGE_RUN="torchrun"
if [ -n "$COVERAGE_ENABLED" ]; then
    export COVERAGE_PROCESS_START="$(pwd)/pyproject.toml"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    COVERAGE_RUN="coverage run --parallel-mode --source=cosmos_predict2 -m torch.distributed.run"
fi

$COVERAGE_RUN $TORCHRUN_ARGS examples/inference.py ...
```

**Key environment variables:**
- `COVERAGE_PROCESS_START`: Points to the coverage config file
- `PYTHONPATH`: Includes current directory so Python can find `sitecustomize.py`

**Note**: The scripts do NOT call `coverage combine` themselves. This is handled by pytest hooks to avoid conflicts when tests run in parallel.

### 3. Pytest Hooks

The `conftest.py` includes a `pytest_sessionfinish` hook that combines all `.coverage.*` files after tests complete. This ensures coverage data from all parallel test runs is properly merged before pytest-cov generates reports.

### 4. ScriptRunner Integration

The `ScriptRunner` fixture (in `cosmos-oss/cosmos_oss/fixtures/script.py`) automatically detects when pytest-cov is active and sets `COVERAGE_ENABLED=1` in the environment passed to shell scripts.

### 5. Coverage Configuration

The `pyproject.toml` contains coverage configuration:

```toml
[tool.coverage.run]
parallel = true
concurrency = ["thread", "multiprocessing"]
source = ["cosmos_predict2"]
sigterm = true
```

## Running Tests with Coverage

### Automatic (Recommended)

Simply run pytest with the `--cov` flag:

```bash
# Run all docs tests with coverage
pytest tests/docs_test.py --cov=cosmos_predict2 --cov-report=html

# Run specific test with coverage
pytest tests/docs_test.py::test_level_0 --cov=cosmos_predict2
```

The `ScriptRunner` will automatically detect coverage is enabled and configure the shell scripts appropriately.

### Manual Control

You can also manually control coverage by setting `COVERAGE_ENABLED`:

```bash
# Enable coverage manually
export COVERAGE_ENABLED=1
bash tests/docs_test/action_conditioned.sh

# Disable coverage
unset COVERAGE_ENABLED
bash tests/docs_test/action_conditioned.sh
```

## Understanding Coverage Output

### Parallel Mode

Coverage runs in `--parallel-mode`, which creates separate `.coverage.*` files for each process. These are automatically combined by the `pytest_sessionfinish` hook after all tests complete.

### Coverage Data Location

Coverage data is stored in `.coverage` files in the current working directory. When running pytest, this is typically the project root.

### Viewing Reports

```bash
# Generate HTML report
coverage html

# Generate terminal report
coverage report

# Generate XML report (for CI/CD)
coverage xml
```

## Troubleshooting

### No Coverage Data Collected

**Problem**: Running tests with `--cov` but no coverage data is generated (all lines show `hits="0"`).

**Solutions**:
1. Ensure `coverage` is installed: `uv pip install coverage[toml]`
2. **Verify `sitecustomize.py` exists at the root** of the package (next to `pyproject.toml`)
3. **Check `PYTHONPATH` includes current directory**: `echo $PYTHONPATH` should include `.` or `$(pwd)`
4. Verify `COVERAGE_PROCESS_START` points to the config file: should be absolute path or `$(pwd)/pyproject.toml`
5. Test if `sitecustomize.py` is being imported: add a print statement temporarily

### Coverage Not Tracking Subprocess Code

**Problem**: Main process code is tracked but distributed worker code is not.

**Solutions**:
1. Verify `sitecustomize.py` exists and is importable
2. Check `COVERAGE_PROCESS_START=pyproject.toml` is exported before running `torchrun`
3. Ensure `concurrency = ["thread", "multiprocessing"]` is in `pyproject.toml`

### Multiple .coverage.* Files Not Combining

**Problem**: Many `.coverage.*` files but `coverage report` shows incomplete data.

**Solutions**:
1. Verify the `pytest_sessionfinish` hook is running (check for "Combined X coverage data file(s)" message)
2. Run `coverage combine` manually to merge all data files
3. Check that `parallel = true` is in `[tool.coverage.run]`
4. Ensure pytest is not being killed prematurely (which would skip the sessionfinish hook)

## Architecture Notes

### Why sitecustomize.py?

Python automatically imports `sitecustomize.py` when starting up **if it's in the Python path**. By setting `PYTHONPATH` to include the current directory, we ensure this file is found and imported in all subprocesses spawned by `torchrun`, enabling coverage tracking without modifying the application code (`action_conditioned.py`, `inference.py`, etc.).

### Why COVERAGE_PROCESS_START?

The `COVERAGE_PROCESS_START` environment variable tells coverage which config file to use. This must be set before coverage starts in subprocesses.

### Why Parallel Mode?

Distributed training creates multiple Python processes. Each process needs its own coverage data file, which are later merged by pytest hooks or `coverage combine`.

## Files Modified

- `sitecustomize.py` - Auto-starts coverage in subprocesses (at package root)
- `conftest.py` - Pytest hook to combine coverage data after tests
- `pyproject.toml` - Coverage configuration and build includes
- `tests/docs_test/*.sh` - Conditional coverage wrapping with PYTHONPATH setup
- `cosmos-oss/cosmos_oss/fixtures/script.py` - Auto-detection of coverage mode
