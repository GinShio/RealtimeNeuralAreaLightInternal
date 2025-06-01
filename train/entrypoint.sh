#!/bin/bash
set -e
cd /workspace

# Install the required packages
sh -c "uv sync"

# Add torch to the python environment
export PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])"):$PYTHONPATH

# pass all arguments to the Python script
exec uv run src/main.py "$@"
