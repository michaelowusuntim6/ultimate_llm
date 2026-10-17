#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOP_K_PER_SIG 24
#define UNION_CAP 96
#define NUM_LINES 2000 /* Eval on 2000 held-out lines */

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed; h ^= t1; h *= 0x01000193u; h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}

int main(void) {
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/vocab_english.txt", "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) return 1;

    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "models/view_%d.db", v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};
    float lambda[4] = {0.25f, 0.25f, 0.25f, 0.25f}; /* Simplified for eval */

    shard_reader_t sr;
    if (shard_open(&sr, "/home/mike/Vocab/dataset_prose.txt", 64) != 0) return 1;

    /* Skip first 5000 lines to simulate held-out eval */
    char *line; size_t len; int lines_read = 0;
    while (lines_read < 5000 && shard_next_line(&sr, &line, &len) > 0) { lines_read++; }

    int32_t tokens[2048];
    double total_log_prob = 0.0;
    size_t total_tokens = 0, top1_correct = 0, top5_correct = 0;
    lines_read = 0;

    while (shard_next_line(&sr, &line, &len) > 0 && lines_read < NUM_LINES) {
        int32_t n_tok = tok_encode(&tok, line, (int32_t)len, tokens, 2048);
        if (n_tok < 4) continue;

        for (int32_t i = 3; i < n_tok; i++) {
            uint32_t t1 = (uint32_t)tokens[i-1], t2 = (uint32_t)tokens[i-2];
            uint32_t t3 = (uint32_t)tokens[i-3], true_c = (uint32_t)tokens[i];
            uint32_t pos_bucket = (uint32_t)(i % 8);

            uint32_t sig[4];
            sig[0] = hash_sig(t1, t2, seeds[0]); sig[1] = hash_sig(t1, t3, seeds[1]);
            sig[2] = hash_sig(t1, 0, seeds[2]);  sig[3] = hash_sig(t1, pos_bucket, seeds[3]);

            uint32_t cand[UNION_CAP]; int n_cand = 0; cand[n_cand++] = true_c;
            for (int v = 0; v < 4; v++) {
                sketch_slot_t *meta = sk_get(&sk[v], sig[v]);
                if (meta && meta->count > 0) {
                    int found = 0; for(int j=0; j<n_cand; j++) if(cand[j] == meta->class_id) { found=1; break; }
                    if (!found) cand[n_cand++] = meta->class_id;
                }
            }
            for (int j = 0; j < 96 && n_cand < UNION_CAP; j++) {
                uint32_t u = (uint32_t)(rand() % tok.n_vocab);
                int found = 0; for(int k=0; k<n_cand; k++) if(cand[k] == u) { found=1; break; }
                if (!found) cand[n_cand++] = u;
            }

            float score[UNION_CAP], max_score = -1e9f;
            for (int c = 0; c < n_cand; c++) {
                score[c] = 0.0f;
                for (int v = 0; v < 4; v++) {
                    uint32_t k = fnv1a_u32_combine(sig[v], cand[c]);
                    sketch_slot_t *w = sk_get(&sk[v], k);
                    if (w) score[c] += lambda[v] * w->logit;
                }
                if (score[c] > max_score) max_score = score[c];
            }

            float p[UNION_CAP], sum_exp = 0.0f;
            for (int c = 0; c < n_cand; c++) { p[c] = expf(score[c] - max_score); sum_exp += p[c]; }
            float inv_sum = 1.0f / sum_exp;

            /* Find rank of true class */
            int rank = 1;
            float p_true = p[0] * inv_sum;
            for (int c = 1; c < n_cand; c++) {
                p[c] *= inv_sum;
                if (p[c] > p_true) rank++;
            }

            total_log_prob += log2f(p_true + 1e-9f);
            total_tokens++;
            if (rank == 1) top1_correct++;
            if (rank <= 5) top5_correct++;
        }
        lines_read++;
    }

    double bpt = -total_log_prob / (double)total_tokens;
    printf("EVAL RESULTS\n");
    printf("tokens: %zu\n", total_tokens);
    printf("Bits-per-token (BPT): %.4f\n", bpt);
    printf("Top-1 Acc: %.2f%%\n", (double)top1_correct / total_tokens * 100.0);
    printf("Top-5 Acc: %.2f%%\n", (double)top5_correct / total_tokens * 100.0);

    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    shard_close(&sr); tok_free(&tok);
    return 0;
}
