#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"
gcc $FLAGS -c src/010_tokenizer.c -o build/010_tokenizer.o
gcc $FLAGS -c src/020_sketch.c   -o build/020_sketch.o
gcc $FLAGS -c src/080_cli.c      -o build/080_cli.o
gcc -flto build/010_tokenizer.o build/020_sketch.o build/080_cli.o -o bin/sllm -lm -lpthread
echo "[080] OK -> bin/sllm"
