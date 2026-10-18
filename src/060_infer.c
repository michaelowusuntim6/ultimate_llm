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

/* Simple xorshift random number generator for sampling */
static uint32_t rng_state = 12345;
static inline uint32_t fast_rand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char **argv) {
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/vocab_english.txt", "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) return 1;

    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "models/view_%d.db", v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    /* Load unigrams for negative sampling during generation (same as trainer) */
    uint32_t unigrams[1000];
    FILE *uf = fopen("models/top_unigrams.bin", "rb");
    if (uf) {
        if (fread(unigrams, sizeof(uint32_t), 1000, uf) != 1000)
            fprintf(stderr, "[infer] top_unigrams.bin short read\n");
        fclose(uf);
    } else {
        /* Fallback if missing */
        for(int i=0; i<1000; i++) unigrams[i] = (uint32_t)(rand() % tok.n_vocab);
    }

    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};
    float lambda[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    FILE *lf = fopen("models/lambdas.bin", "rb");
    if (lf) {
        if (fread(lambda, sizeof(float), 4, lf) != 4)
            fprintf(stderr, "[infer] lambdas.bin short read; using defaults\n");
        fclose(lf);
    } else {
        fprintf(stderr, "[infer] no lambdas.bin; using defaults\n");
    }
    float temp = 1.0f;

    /* Setup default prompt or grab from args */
    const char *prompt = (argc > 1) ? argv[1] : "The";
    int max_gen = (argc > 2) ? atoi(argv[2]) : 50;

    /* TEMP HACK: lowercase the prompt so proper nouns match the lowercase vocab. */
    char lower_prompt[512];
    strncpy(lower_prompt, prompt, 511);
    lower_prompt[511] = '\0';
    for (int i = 0; lower_prompt[i]; i++) {
        if (lower_prompt[i] >= 'A' && lower_prompt[i] <= 'Z') lower_prompt[i] += 32;
    }

    int32_t tokens[2048];
    int n_tok = tok_encode(&tok, lower_prompt, (int32_t)strlen(lower_prompt), tokens, 2048);
    if (n_tok < 1) { printf("Empty prompt.\n"); return 1; }

    printf("Prompt: %s\n", prompt);
    printf("Generating %d tokens...\n", max_gen);

    char out_buf[8192];
    int out_len = 0;

    for (int step = 0; step < max_gen; step++) {
        int32_t i = n_tok;
        if (i < 5) break; /* Need at least 5 tokens for expanded context */

        uint32_t t1 = (uint32_t)tokens[i-1];
        uint32_t t2 = (uint32_t)tokens[i-2];
        uint32_t t3 = (uint32_t)tokens[i-3];
        uint32_t t4 = (uint32_t)tokens[i-4];
        uint32_t t5 = (uint32_t)tokens[i-5];
        uint32_t pos_bucket = (uint32_t)(i % 8);

        uint32_t sig[4];
        sig[0] = hash_sig(t1, t2, seeds[0]); sig[1] = hash_sig(t2, t3, seeds[1]);
        sig[2] = hash_sig(t1, t4, seeds[2]); sig[3] = hash_sig(t3, t5, seeds[3]);

        uint32_t cand[UNION_CAP]; int n_cand = 0;

        /* 1. Gather Top-8 from each context view */
        for (int v = 0; v < 4; v++) {
            sketch_slot_t *meta = sk_get(&sk[v], sig[v]);
            if (meta && meta->count > 0) {
                for (int j = 0; j < 8; j++) {
                    uint32_t cl = meta->topk_classes[j];
                    if (cl == 0) continue; /* empty slot */
                    int found = 0;
                    for (int q = 0; q < n_cand; q++) if (cand[q] == cl) { found = 1; break; }
                    if (!found) cand[n_cand++] = cl;
                }
            }
        }

        /* 2. Pad with unigrams (deduplicated) */
        while (n_cand < UNION_CAP) {
            uint32_t u = unigrams[fast_rand() % 1000];
            int found = 0; for(int j=0; j<n_cand; j++) if(cand[j] == u) { found=1; break; }
            if (!found) cand[n_cand++] = u;
        }

        /* 3. Score */
        float score[UNION_CAP], max_score = -1e9f;
        for (int c = 0; c < n_cand; c++) {
            score[c] = 0.0f;
            for (int v = 0; v < 4; v++) {
                uint32_t k = fnv1a_u32_combine(sig[v], cand[c]);
                sketch_slot_t *w = sk_get(&sk[v], k);
                if (w) score[c] += lambda[v] * w->logit;
            }
            score[c] /= temp; /* Apply temperature */
            if (score[c] > max_score) max_score = score[c];
        }

        /* 4. Softmax */
        float p[UNION_CAP], sum_exp = 0.0f;
        for (int c = 0; c < n_cand; c++) { p[c] = expf(score[c] - max_score); sum_exp += p[c]; }
        float inv_sum = 1.0f / sum_exp;

        /* 5. Sample */
        float r = (float)(fast_rand() % 100000) / 100000.0f;
        float cum = 0.0f;
        uint32_t next_tok = cand[0];
        for (int c = 0; c < n_cand; c++) {
            p[c] *= inv_sum;
            cum += p[c];
            if (r <= cum) { next_tok = cand[c]; break; }
        }

        /* 6. Append and decode */
        tokens[n_tok++] = (int32_t)next_tok;

        /* Inject space before word tokens, but avoid double spaces */
        if (next_tok < (uint32_t)tok.n_word_tokens && out_len > 0) {
            if (out_buf[out_len-1] != ' ') {
                out_buf[out_len++] = ' ';
            }
        }

        int32_t nt32 = (int32_t)next_tok;
        int32_t decoded = tok_decode(&tok, &nt32, 1, out_buf + out_len, 8192 - out_len);
        if (decoded > 0) out_len += decoded;
    }

    out_buf[out_len] = '\0';
    printf("\n--- OUTPUT ---\n%s\n--------------\n", out_buf);

    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    tok_free(&tok);
    return 0;
}
