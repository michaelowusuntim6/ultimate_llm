#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p logs

VOCAB="${1:-/home/mike/Vocab/vocab_english.txt}"
FALLBACK="${2:-/home/mike/Vocab/byte_fallback_tokens.txt}"
PROSE="${3:-/home/mike/Vocab/dataset_prose.txt}"
NLINES="${4:-10000}"

echo "[011] running bin/tok_test on $PROSE (first $NLINES lines)"
bin/tok_test "$VOCAB" "$FALLBACK" "$PROSE" "$NLINES" | tee logs/011_test.log
