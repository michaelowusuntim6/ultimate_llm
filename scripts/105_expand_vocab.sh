#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build bin models logs
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"

echo "[105] compiling vocab expander..."
gcc $FLAGS -c src/030_shard.c        -o build/030_shard.o
gcc $FLAGS -c src/105_expand_vocab.c -o build/105_expand_vocab.o
gcc -flto build/030_shard.o build/105_expand_vocab.o -o bin/expand_vocab -lm -lpthread

echo "[105] scanning TinyStories and building models/vocab_expanded.txt..."
./bin/expand_vocab 2>&1 | tee logs/105_expand_vocab.log
echo "[105] expanded vocab size:"; wc -l models/vocab_expanded.txt
