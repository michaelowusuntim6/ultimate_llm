#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"

echo "[020] compiling sketch..."
gcc $FLAGS -c src/020_sketch.c -o build/020_sketch.o
gcc $FLAGS -c src/021_test_sketch.c -o build/021_test_sketch.o
gcc -flto build/020_sketch.o build/021_test_sketch.o -o bin/sketch_test -lm -lpthread
echo "[020] OK -> bin/sketch_test"
