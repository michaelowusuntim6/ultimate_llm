#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define UNION_CAP 96

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed; h ^= t1; h *= 0x01000193u; h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t rng_state = 12345;
static inline uint32_t fast_rand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char **argv) {
    int bench_tokens = (argc > 1) ? atoi(argv[1]) : 100;

    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/vocab_english.txt", "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) return 1;

    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "models/view_%d.db", v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    uint32_t unigrams[1000];
    FILE *uf = fopen("models/top_unigrams.bin", "rb");
    if (uf) { if (fread(unigrams, sizeof(uint32_t), 1000, uf) != 1000) fprintf(stderr, "[bench] unigrams short read\n"); fclose(uf); }
    else { for (int i = 0; i < 1000; i++) unigrams[i] = (uint32_t)(rand() % tok.n_vocab); }

    float lambda[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    FILE *lf = fopen("models/lambdas.bin", "rb");
    if (lf) { if (fread(lambda, sizeof(float), 4, lf) != 4) fprintf(stderr, "[bench] lambdas short read\n"); fclose(lf); }

    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};
    const char *prompt = "The capital of France is";
    int32_t tokens[2048];
    int n_tok = tok_encode(&tok, prompt, (int32_t)strlen(prompt), tokens, 2048);
    if (n_tok < 5) { printf("[bench] prompt too short\n"); return 1; }

    double t0 = now();
    for (int step = 0; step < bench_tokens; step++) {
        int32_t i = n_tok;
        uint32_t t1 = (uint32_t)tokens[i-1], t2 = (uint32_t)tokens[i-2];
        uint32_t t3 = (uint32_t)tokens[i-3], t4 = (uint32_t)tokens[i-4], t5 = (uint32_t)tokens[i-5];

        uint32_t sig[4];
        sig[0] = hash_sig(t1, t2, seeds[0]); sig[1] = hash_sig(t2, t3, seeds[1]);
        sig[2] = hash_sig(t1, t4, seeds[2]); sig[3] = hash_sig(t3, t5, seeds[3]);

        uint32_t cand[UNION_CAP]; int n_cand = 0;
        for (int v = 0; v < 4; v++) {
            sketch_slot_t *meta = sk_get(&sk[v], sig[v]);
            if (meta && meta->count > 0) {
                int found = 0; for (int j = 0; j < n_cand; j++) if (cand[j] == meta->class_id) { found = 1; break; }
                if (!found) cand[n_cand++] = meta->class_id;
            }
        }
        while (n_cand < UNION_CAP) {
            uint32_t u = unigrams[fast_rand() % 1000];
            int found = 0; for (int j = 0; j < n_cand; j++) if (cand[j] == u) { found = 1; break; }
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

        float r = (float)(fast_rand() % 100000) / 100000.0f;
        float cum = 0.0f;
        uint32_t next_tok = cand[0];
        for (int c = 0; c < n_cand; c++) {
            p[c] *= inv_sum; cum += p[c];
            if (r <= cum) { next_tok = cand[c]; break; }
        }
        tokens[n_tok++] = (int32_t)next_tok;
    }
    double elapsed = now() - t0;
    double tps = bench_tokens / elapsed;

    printf("INFER BENCH: %.1f tokens/sec (%d tokens in %.4fs)\n", tps, bench_tokens, elapsed);

    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    tok_free(&tok);
    return 0;
}
