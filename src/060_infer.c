#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define UNION_CAP 96

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed; h ^= t1; h *= 0x01000193u; h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}

static uint32_t rng_state = 12345;
static inline uint32_t fast_rand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

int main(int argc, char **argv) {
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/tokenizer_pure/vocab.json",
                      "/home/mike/Vocab/tokenizer_pure/merges.txt") != 0) return 1;

    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "models/view_%d.db", v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};
    /* Stupid-backoff weights per view: higher-order views get more trust. */
    float backoff_w[4] = {0.4f, 0.25f, 0.2f, 0.15f};
    float temp = 1.0f;

    const char *prompt = (argc > 1) ? argv[1] : "The";
    int max_gen = (argc > 2) ? atoi(argv[2]) : 50;

    int32_t tokens[4096];
    int n_tok = tok_encode(&tok, prompt, (int32_t)strlen(prompt), tokens, 4096);
    if (n_tok < 1) { printf("Empty prompt.\n"); return 1; }

    printf("Prompt: %s\nGenerating %d tokens...\n", prompt, max_gen);

    char out_buf[16384];
    int out_len = 0;
    /* echo the prompt first */
    { int dl = tok_decode(&tok, tokens, n_tok, out_buf, sizeof(out_buf)-1); if (dl>0) out_len = dl; }

    for (int step = 0; step < max_gen; step++) {
        int32_t i = n_tok;
        if (i < 5) break;

        uint32_t t1=(uint32_t)tokens[i-1], t2=(uint32_t)tokens[i-2];
        uint32_t t3=(uint32_t)tokens[i-3], t4=(uint32_t)tokens[i-4], t5=(uint32_t)tokens[i-5];
        uint32_t sig[4];
        sig[0]=hash_sig(t1,t2,seeds[0]); sig[1]=hash_sig(t2,t3,seeds[1]);
        sig[2]=hash_sig(t1,t4,seeds[2]); sig[3]=hash_sig(t3,t5,seeds[3]);

        /* gather candidates from all Top-K lists across views */
        uint32_t cand[UNION_CAP]; int n_cand = 0;
        float score[UNION_CAP];
        for (int v = 0; v < 4; v++) {
            sketch_slot_t *meta = sk_get(&sk[v], sig[v]);
            if (!meta || meta->count == 0) continue;
            for (int k = 0; k < 8; k++) {
                uint32_t cl = meta->topk_classes[k];
                if (cl == 0 || meta->topk_counts[k] == 0) continue;
                int found = -1;
                for (int j = 0; j < n_cand; j++) if (cand[j] == cl) { found = j; break; }
                float p = (float)meta->topk_counts[k] / (float)meta->count;
                if (found >= 0) score[found] += p * backoff_w[v];
                else if (n_cand < UNION_CAP) { cand[n_cand] = cl; score[n_cand] = p * backoff_w[v]; n_cand++; }
            }
        }
        if (n_cand == 0) break; /* no continuation known */

        /* softmax over scores with temperature, then sample */
        float mx = -1e30f;
        for (int c = 0; c < n_cand; c++) if (score[c]/temp > mx) mx = score[c]/temp;
        float sum = 0; float p[UNION_CAP];
        for (int c = 0; c < n_cand; c++) { p[c] = expf(score[c]/temp - mx); sum += p[c]; }
        float r = (float)(fast_rand() % 100000) / 100000.0f * sum;
        float cum = 0; uint32_t next = cand[0];
        for (int c = 0; c < n_cand; c++) { cum += p[c]; if (r <= cum) { next = cand[c]; break; } }

        tokens[n_tok++] = (int32_t)next;
        int dl = tok_decode(&tok, (const int32_t[]){(int32_t)next}, 1, out_buf + out_len, (int32_t)sizeof(out_buf) - out_len - 1);
        if (dl > 0) out_len += dl;
    }

    out_buf[out_len] = '\0';
    printf("\n--- OUTPUT ---\n%s\n--------------\n", out_buf);

    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    tok_free(&tok);
    return 0;
}
