#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
echo "[070] inference benchmark (100 tokens)..."
./bin/bench 100 | tee logs/070_bench.log
