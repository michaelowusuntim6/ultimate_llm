# Ultimate LLM — SLLM

**S**parse **L**attice **L**anguage **M**odel: a matmul-free language model that
predicts the next token through sparse signature lookups over an `mmap`'d
open-addressing sketch table. No BLAS, no GPU, no dense matrices — just FNV-1a
hashing, a trie tokenizer, and a single 2-core Ivy Bridge CPU.

---

## Why this exists

Every mainstream transformer spends its compute on dense matrix multiplies. SLLM
asks: what if a language model were *just* a giant associative memory? For each
context window we hash several "signatures", gather a small union of candidate
next-tokens, softmax over only those, and update only the touched weights with a
count-aware SGD rule. The result is a model that trains and infers at thousands
of tokens/second on commodity hardware, in pure C99, linking nothing but `-lm -lpthread`.

## Architecture

```
text ──▶ Tokenizer (trie + byte-fallback, case-folded)
     ──▶ 4 context signatures per position (FNV-1a, seeded)
     ──▶ Sketch lookup: Top-8 candidates per signature → union ≤ 96
     ──▶ Softmax over the candidate union
     ──▶ LC-SGD update of only the touched slots
     ──▶ Backoff mixture λ (normalized to the simplex)
```

| Subsystem | Design |
|---|---|
| **Tokenizer** | Trie over 231,427 lowercase words + 256 byte-fallback + 5 specials = 231,688 vocab. Longest-match walk; input case-folded so proper nouns stay whole tokens. |
| **Sketch** | 48-byte slots (`key, class_id, logit, count, gen, topk_classes[8]`), open addressing, linear probe, backed by `mmap` files pre-sized with `posix_fallocate`. |
| **Streaming loader** | Fixed 64 MB window, memmove partial-line carry — zero-OOM by construction. |
| **Trainer** | 4 seeded signatures/token; candidate union of Top-8-per-view ∪ unigram negatives; softmax; LC-SGD (`η = lr / (1 + decay·log1p(count))`); normalized backoff weights. |
| **Inference** | Same gather-score-softmax path with temperature sampling; pulls the full Top-8 list per signature. |

## Performance (measured on i5-3210M, AVX1-only)

| Metric | Value |
|---|---|
| Inference speed | **1801 tokens/sec** |
| Sketch insert throughput | 7.77M ops/s |
| Streaming throughput | 2837 MB/s @ 64 MB window |
| Held-out BPT | 3.63 bits/token |
| Top-1 accuracy | **64.3%** |
| Top-5 accuracy | 90.2% |
| Model footprint | ~1.5 GB (4 sketches × 384 MB) |
| Dependencies | libc, libm, pthread — nothing else |

## Build & run

Requirements: `gcc`, `make`, POSIX mmap. All binaries build with the exact flags
in [`CONFIG.md`](CONFIG.md): `-O3 -march=ivybridge -std=c99`.

```sh
# 0. Bootstrap + directories
sh scripts/000_bootstrap.sh
sh scripts/001_init_dirs.sh

# 1. Extract unigram negatives from the corpus
sh scripts/039_extract_unigrams.sh

# 2. Build + train (full corpus)
sh scripts/040_build_trainer.sh
./bin/train models 175926 > logs/full_train.log 2>&1 &

# 3. Evaluate held-out lines
sh scripts/050_build_eval.sh && ./bin/eval

# 4. Generate text
sh scripts/060_build_infer.sh
./bin/chat "The capital of France is" 15

# 5. Benchmark inference
sh scripts/070_bench.sh
```

Data assets live in `/home/mike/Vocab` (`vocab_english.txt`,
`byte_fallback_tokens.txt`, `dataset_prose.txt`). Paths are configurable in
[`CONFIG.md`](CONFIG.md).

## Repository layout

```
src/       C99 sources (tokenizer, sketch, shard, trainer, eval, infer, bench)
scripts/   numbered build/test/run scripts (run in numeric order)
models/    mmap'd sketch databases + extracted assets (gitignored)
build/     object files (gitignored)
bin/       compiled binaries (gitignored)
logs/      per-phase logs + FINAL_REPORT.md (gitignored)
```

## Status & roadmap

Phase 1 (this repo) proves the architecture end-to-end: every subsystem passes
its gate and training loss descends monotonically on the full corpus. Phase 2
upgrades shipped here close the two biggest gaps:

- **Case-folded tokenizer** — proper nouns no longer fragment into byte+subword.
- **Top-8 retrieval per signature** — generation now draws real candidates
  instead of a bag of random frequent words.

Remaining roadmap: richer top-K confidence ranking, Robin-Hood probing to cap
load-factor latency, and larger corpora for factual coherence.

## License

MIT. See [`ToDo_List.md`](ToDo_List.md) for the full execution ledger.
