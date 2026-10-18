#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Count unigram token frequencies over a corpus and persist the top-K token IDs.
   Output: binary array of uint32_t, exactly K entries (padded with last valid id). */

static int cmp_count_desc(const void *a, const void *b) {
    /* each element is a packed uint64: high 32 = count, low 32 = token id */
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    uint32_t cx = (uint32_t)(x >> 32), cy = (uint32_t)(y >> 32);
    if (cy > cx) return 1;
    if (cy < cx) return -1;
    return 0;
}

int main(int argc, char **argv) {
    const char *vocab   = "/home/mike/Vocab/vocab_english.txt";
    const char *fallback= "/home/mike/Vocab/byte_fallback_tokens.txt";
    const char *corpus  = (argc > 1) ? argv[1] : "/home/mike/Vocab/dataset_prose.txt";
    const char *outpath = (argc > 2) ? argv[2] : "models/top_unigrams.bin";
    int         K       = (argc > 3) ? atoi(argv[3]) : 1000;

    tokenizer_t tok;
    if (tok_init(&tok, vocab, fallback) != 0) { fprintf(stderr, "tok_init failed\n"); return 1; }
    int32_t n_vocab = tok.n_vocab;

    uint64_t *counts = calloc((size_t)n_vocab, sizeof(uint64_t));
    if (!counts) { fprintf(stderr, "OOM counts\n"); return 1; }

    shard_reader_t sr;
    if (shard_open(&sr, corpus, 64) != 0) return 1;

    char *line; size_t len;
    int32_t tokens[4096];
    long total_tokens = 0, lines = 0;
    while (shard_next_line(&sr, &line, &len) > 0) {
        int32_t n = tok_encode(&tok, line, (int32_t)len, tokens, 4096);
        for (int32_t i = 0; i < n; i++) {
            int32_t t = tokens[i];
            /* Only count dictionary word tokens; skip byte fallback + specials. */
            if (t >= 0 && t < tok.n_word_tokens) {
                counts[(size_t)t] += (1ULL << 32);   /* bump count field */
                total_tokens++;
            }
        }
        lines++;
    }
    shard_close(&sr);

    /* store token id in low bits */
    for (int32_t t = 0; t < n_vocab; t++)
        counts[(size_t)t] |= (uint64_t)(uint32_t)t;

    qsort(counts, (size_t)n_vocab, sizeof(uint64_t), cmp_count_desc);

    FILE *f = fopen(outpath, "wb");
    if (!f) { perror(outpath); return 1; }
    for (int i = 0; i < K; i++) {
        uint32_t id = (uint32_t)(counts[i] & 0xFFFFFFFFu);
        fwrite(&id, sizeof(id), 1, f);
    }
    fclose(f);

    printf("[unigrams] lines=%ld tokens=%ld vocab=%d -> wrote top %d ids to %s\n",
           lines, total_tokens, n_vocab, K, outpath);
    printf("[unigrams] top5: ");
    for (int i = 0; i < 5; i++) printf("%u ", (uint32_t)(counts[i] & 0xFFFFFFFFu));
    printf("\n");

    free(counts);
    tok_free(&tok);
    return 0;
}
