#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"
LIBS="-lm -lpthread"

echo "[040] compiling tokenizer & shard dependencies..."
gcc $FLAGS -c src/010_tokenizer.c -o build/010_tokenizer.o
gcc $FLAGS -c src/020_sketch.c -o build/020_sketch.o
gcc $FLAGS -c src/030_shard.c -o build/030_shard.o

echo "[040] compiling trainer..."
gcc $FLAGS -c src/040_train.c -o build/040_train.o

echo "[040] linking bin/train"
gcc -flto build/010_tokenizer.o build/020_sketch.o build/030_shard.o build/040_train.o -o bin/train $LIBS
echo "[040] OK -> bin/train"
