# SLLM — Central Configuration
# Paper-style comments. Every key here is read by exactly one module.
# Change a value, rerun the owning script; nothing else moves.

## ── Paths ─────────────────────────────────────────────────────────────
# Absolute root. All other paths are relative to this unless stated absolute.
project_root: /home/mike/sllm

# Reference corpus + tokenizer assets (read-only; from /home/mike/Vocab).
data_dir: /home/mike/Vocab
vocab_file: /home/mike/Vocab/vocab_english.txt
fallback_file: /home/mike/Vocab/byte_fallback_tokens.txt

# Phase 8: TinyStories training corpus + expanded vocabulary.
train_corpus: dataset/ts-ds/tinystories_train.txt
expanded_vocab: models/vocab_expanded.txt

# Writable roots created by 001_init_dirs.sh.
build_dir:   build
model_dir:   models
log_dir:     logs
bin_dir:     bin

## ── Build / Target ────────────────────────────────────────────────────
# Ivy Bridge = AVX1 + SSE4.2. NOT AVX2. NOT FMA.
# -march=ivybridge is the ONLY correct arch flag for this CPU.
compiler: gcc
opt_flags: -O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -fno-trapping-math -funroll-loops -flto
warn_flags: -Wall -Wextra -Wno-unused-parameter -std=c99
link_flags: -lm -lpthread

## ── Tokenizer ─────────────────────────────────────────────────────────
# Trie over vocab_english.txt; byte fallback guarantees no OOV crash.
# hash: FNV-1a 32-bit, seeded per-context-view (see seeds below).
tokenizer_max_word_len: 64
tokenizer_use_trie: 1

## ── Context signatures ────────────────────────────────────────────────
# Four views per position. Each is a 32-bit FNV-1a hash of the listed inputs.
# Fewer views = faster but weaker. 4 is the sweet spot on 2C/4T.
signature_views: 4
sig_seed_0: 0x9E3779B1   # (t-1, t-2)
sig_seed_1: 0x85EBCA6B   # (t-1, t-3)
sig_seed_2: 0xC2B2AE35   # (t-1)
sig_seed_3: 0x27D4EB2F   # (t-1, pos_bucket)

## ── Sketch (parameter store) ──────────────────────────────────────────
# Open-addressing, linear probe, mmap-backed. Slot = 16 bytes
# (4 key + 4 pad + 4 fp32 logit + 2 u16 count + 2 u16 gen).
sketch_capacity_entries: 33554432      # 32 Mi slots × 16 B = 512 MB
sketch_slot_bytes: 16
sketch_load_factor_max: 0.90           # trigger eviction sweep above this
sketch_evict_count_threshold: 1        # drop entries with count <= this
sketch_evict_gen_window: 3             # only sweep gens older than cur-3
sketch_flush_every_sec: 30

## ── Prediction / training ─────────────────────────────────────────────
# Candidate union per signature. 24 × 4 views = ≤96 candidates per token.
top_k_per_signature: 24
candidate_union_cap: 96

# LC-SGD: η_eff = lr / (1 + decay * log1p(count))
learning_rate: 0.12
lr_count_decay_lambda: 0.35
grad_clip: 4.0

# Backoff mixture weights λ_0..λ_3 (learned by 4-float SGD).
backoff_lr: 0.02
backoff_init: 0.25

# Residue booster (layer 2). Off = pure layer-1 model.
residue_enabled: 1
residue_buckets: 16                    # residual = -log2 p(true), bucketed
residue_lr: 0.08

## ── Streaming / memory ────────────────────────────────────────────────
# Hard ceiling on resident mmap'd corpus. Guarantees zero OOM on 16 GB.
shard_window_mb: 64
shard_read_ahead: 2
max_epochs: 3
shuffle_seed: 0xC0FFEE

## ── Inference harness ─────────────────────────────────────────────────
infer_temperature: 0.85
infer_top_p: 0.92
infer_top_k: 40
infer_max_tokens: 256
infer_threads: 2                       # 2 physical cores; HT gives nothing here

## ── Benchmark ─────────────────────────────────────────────────────────
bench_warmup_iters: 3
bench_timed_iters: 10
bench_tokens_per_iter: 1000000
bench_baseline_cmd: ""                 # optional: path to llama.cpp binary for A/B
