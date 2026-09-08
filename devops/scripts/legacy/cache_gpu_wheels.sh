#!/usr/bin/env bash
# Populate a local, untracked wheelhouse before an offline/retry-prone build.
# This is deliberately separate from docker build so interrupted downloads can
# be retried without invalidating dependency layers.
set -euo pipefail

cache_dir="${1:-.cache/wheels}"
mkdir -p "$cache_dir"
python3 -m pip download --dest "$cache_dir" --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
python3 -m pip download --dest "$cache_dir" --extra-index-url https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117
python3 -m pip hash "$cache_dir"/torch-*.whl > "$cache_dir"/SHA256SUMS.txt
