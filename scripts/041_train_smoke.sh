#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs models
echo "[041] running 5000-line training smoke test..."
./bin/train models 5000 | tee logs/041_train_smoke.log
