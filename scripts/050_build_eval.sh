#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"
gcc $FLAGS -c src/010_tokenizer.c -o build/010_tokenizer.o
gcc $FLAGS -c src/020_sketch.c -o build/020_sketch.o
gcc $FLAGS -c src/030_shard.c -o build/030_shard.o
gcc $FLAGS -c src/050_eval.c -o build/050_eval.o
gcc -flto build/010_tokenizer.o build/020_sketch.o build/030_shard.o build/050_eval.o -o bin/eval -lm -lpthread
echo "[050] OK -> bin/eval"
