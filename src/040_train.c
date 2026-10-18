#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Count-based n-gram trainer: no SGD, no softmax. We just tally next-token
   frequencies per context signature into the Top-K slot lists. */

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed;
    h ^= t1; h *= 0x01000193u;
    h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "models";
    int max_lines = (argc > 2) ? atoi(argv[2]) : 175926;

    printf("[train] Loading BPE tokenizer...\n");
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/tokenizer_pure/vocab.json",
                      "/home/mike/Vocab/tokenizer_pure/merges.txt") != 0) return 1;

    printf("[train] Initializing 4 sketches (8M slots each)...\n");
    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "%s/view_%d.db", model_dir, v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};

    printf("[train] Opening corpus stream...\n");
    shard_reader_t sr;
    if (shard_open(&sr, "/home/mike/Vocab/dataset_prose.txt", 64) != 0) return 1;

    char *line; size_t len;
    int32_t tokens[4096];
    int lines_read = 0;
    size_t total_tokens = 0;

    printf("[train] Counting...\n");
    double t_start = now();

    while (shard_next_line(&sr, &line, &len) > 0 && lines_read < max_lines) {
        int32_t n_tok = tok_encode(&tok, line, (int32_t)len, tokens, 4096);
        if (n_tok < 6) { lines_read++; continue; }

        for (int32_t i = 5; i < n_tok; i++) {
            uint32_t t1 = (uint32_t)tokens[i-1], t2 = (uint32_t)tokens[i-2];
            uint32_t t3 = (uint32_t)tokens[i-3], t4 = (uint32_t)tokens[i-4];
            uint32_t t5 = (uint32_t)tokens[i-5];
            uint32_t true_c = (uint32_t)tokens[i];

            uint32_t sig[4];
            sig[0] = hash_sig(t1, t2, seeds[0]);
            sig[1] = hash_sig(t2, t3, seeds[1]);
            sig[2] = hash_sig(t1, t4, seeds[2]);
            sig[3] = hash_sig(t3, t5, seeds[3]);

            for (int v = 0; v < 4; v++) {
                sketch_slot_t *meta = sk_upsert(&sk[v], sig[v]);
                /* find true_c in topk list */
                int found = -1;
                for (int k = 0; k < 8; k++) if (meta->topk_classes[k] == true_c) { found = k; break; }
                if (found >= 0) {
                    if (meta->topk_counts[found] < 65535) meta->topk_counts[found]++;
                } else {
                    /* replace lowest-count slot (or first empty) */
                    int worst = 0;
                    for (int k = 1; k < 8; k++)
                        if (meta->topk_counts[k] < meta->topk_counts[worst]) worst = k;
                    meta->topk_classes[worst] = true_c;
                    meta->topk_counts[worst] = 1;
                }
                if (meta->class_id == 0 || meta->count == 0) meta->class_id = true_c;
                if (meta->count < 65535) meta->count++;
            }
        }

        lines_read++;
        total_tokens += n_tok;

        if (lines_read % 10000 == 0) {
            double elapsed = now() - t_start;
            double tps = (double)total_tokens / elapsed;
            printf("line %d | tokens %zu | %.0f tok/s | entries: %zu %zu %zu %zu\n",
                   lines_read, total_tokens, tps,
                   sk[0].n_entries, sk[1].n_entries, sk[2].n_entries, sk[3].n_entries);
            fflush(stdout);
            for (int v = 0; v < 4; v++)
                if (sk[v].n_entries > sk[v].capacity * 0.85)
                    sk_evict_sweep(&sk[v], 2, 1);
        }
    }

    printf("GATE PASS: Counting complete. Processed %d lines.\n", lines_read);
    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    shard_close(&sr);
    tok_free(&tok);
    return 0;
}
