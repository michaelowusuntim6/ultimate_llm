#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"
gcc $FLAGS -c src/010_tokenizer.c -o build/010_tokenizer.o
gcc $FLAGS -c src/020_sketch.c   -o build/020_sketch.o
gcc $FLAGS -c src/070_bench.c    -o build/070_bench.o
gcc -flto build/010_tokenizer.o build/020_sketch.o build/070_bench.o -o bin/bench -lm -lpthread
echo "[070] OK -> bin/bench"
