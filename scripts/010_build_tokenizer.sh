#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build bin logs

FLAGS="-O3 -march=ivybridge -mtune=ivybridge \
       -fno-math-errno -fno-trapping-math -funroll-loops -flto \
       -Wall -Wextra -Wno-unused-parameter -std=c99"

echo "[010] compiling 010_tokenizer.c"
gcc $FLAGS -c src/010_tokenizer.c -o build/010_tokenizer.o 2>&1 | tee logs/010_build.log

echo "[010] compiling 011_test_tokenizer.c"
gcc $FLAGS -c src/011_test_tokenizer.c -o build/011_test_tokenizer.o 2>&1 | tee -a logs/010_build.log

echo "[010] linking bin/tok_test"
gcc -flto build/010_tokenizer.o build/011_test_tokenizer.o \
    -o bin/tok_test -lm -lpthread 2>&1 | tee -a logs/010_build.log

echo "[010] OK -> bin/tok_test"
