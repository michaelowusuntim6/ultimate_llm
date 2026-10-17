#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"

echo "[030] compiling shard..."
gcc $FLAGS -c src/030_shard.c -o build/030_shard.o
gcc $FLAGS -c src/031_test_shard.c -o build/031_test_shard.o
gcc -flto build/030_shard.o build/031_test_shard.o -o bin/shard_test -lm -lpthread
echo "[030] OK -> bin/shard_test"
