#!/bin/bash
set -e

# install dependencies
sh -c "uv sync"

# pass all arguments to the Python script
exec python src/main.py "$@"
