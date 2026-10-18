#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build bin models logs
FLAGS="-O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99"

echo "[039] compiling unigram extractor..."
gcc $FLAGS -c src/010_tokenizer.c        -o build/010_tokenizer.o
gcc $FLAGS -c src/030_shard.c            -o build/030_shard.o
gcc $FLAGS -c src/039_extract_unigrams.c -o build/039_extract_unigrams.o
gcc -flto build/010_tokenizer.o build/030_shard.o build/039_extract_unigrams.o \
    -o bin/extract_unigrams -lm -lpthread

echo "[039] extracting top 1000 unigrams from tinystories..."
./bin/extract_unigrams dataset/ts-ds/tinystories_train.txt models/top_unigrams.bin 1000 | tee logs/039_unigrams.log
echo "[039] OK -> models/top_unigrams.bin"
