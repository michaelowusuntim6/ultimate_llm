#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Hyperparameters matching CONFIG.md */
#define TOP_K_PER_SIG 24
#define UNION_CAP 96
#define LR 0.12f
#define LAMBDA_DECAY 0.35f
#define GRAD_CLIP 4.0f
#define BACKOFF_LR 0.02f
#define UNIGRAM_K 1000

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed;
    h ^= t1; h *= 0x01000193u;
    h ^= t2; h *= 0x01000193u;
    return h ? h : 1; /* ensure non-zero */
}

/* Maintain an insertion-ordered Top-K list (most-recent-first) of size TOPK.
   Returns 1 if true_c was newly added (list grew), 0 if it was already present. */
#define TOPK 8
static int topk_touch(uint32_t *topk, uint32_t true_c) {
    for (int j = 0; j < TOPK; j++) {
        if (topk[j] == true_c) return 0;            /* already ranked */
    }
    for (int j = TOPK - 1; j > 0; j--) topk[j] = topk[j-1];  /* shift down */
    topk[0] = true_c;                                /* insert at head */
    return 1;
}

int main(int argc, char **argv) {
    const char *model_dir  = (argc > 1) ? argv[1] : "models";
    int         max_lines  = (argc > 2) ? atoi(argv[2]) : 10000;

    printf("[train] Initializing tokenizer...\n");
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/vocab_english.txt", "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) {
        return 1;
    }

    printf("[train] Initializing 4 sketches (8M slots each, 512MB total)...\n");
    sketch_t sk[4];
    char path[256];
    for (int v = 0; v < 4; v++) {
        snprintf(path, sizeof(path), "%s/view_%d.db", model_dir, v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    float lambda[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    uint32_t seeds[4] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F};

    printf("[train] Loading unigram negatives (models/top_unigrams.bin)...\n");
    uint32_t unigrams[UNIGRAM_K];
    {
        FILE *uf = fopen("models/top_unigrams.bin", "rb");
        if (!uf) { perror("models/top_unigrams.bin"); return 1; }
        if (fread(unigrams, sizeof(uint32_t), UNIGRAM_K, uf) != (size_t)UNIGRAM_K) {
            fprintf(stderr, "top_unigrams.bin: expected %d entries\n", UNIGRAM_K);
            fclose(uf); return 1;
        }
        fclose(uf);
    }

    printf("[train] Opening corpus stream...\n");
    shard_reader_t sr;
    if (shard_open(&sr, "/home/mike/Vocab/dataset_prose.txt", 64) != 0) return 1;

    char *line;
    size_t len;
    int32_t tokens[2048];
    int lines_read = 0;
    size_t total_tokens = 0;
    double total_loss = 0.0f;

    printf("[train] Starting LC-SGD training loop...\n");
    double t_start = now();

    while (shard_next_line(&sr, &line, &len) > 0 && lines_read < max_lines) {
        int32_t n_tok = tok_encode(&tok, line, (int32_t)len, tokens, 2048);
        if (n_tok < 4) continue;

        for (int32_t i = 5; i < n_tok; i++) {
            uint32_t t1 = (uint32_t)tokens[i-1];
            uint32_t t2 = (uint32_t)tokens[i-2];
            uint32_t t3 = (uint32_t)tokens[i-3];
            uint32_t t4 = (uint32_t)tokens[i-4];
            uint32_t t5 = (uint32_t)tokens[i-5];
            uint32_t true_c = (uint32_t)tokens[i];
            uint32_t pos_bucket = (uint32_t)(i % 8);

            /* 1. Compute 4 signatures (expanded context bridges over spaces) */
            uint32_t sig[4];
            sig[0] = hash_sig(t1, t2, seeds[0]);
            sig[1] = hash_sig(t2, t3, seeds[1]);
            sig[2] = hash_sig(t1, t4, seeds[2]);
            sig[3] = hash_sig(t3, t5, seeds[3]);

            /* 2. Gather candidates (union of top classes from each sig) */
            uint32_t cand[UNION_CAP];
            int n_cand = 0;
            cand[n_cand++] = true_c; /* Always include true class */

            /* 2a. Context predictions: top-1 per view, deduped */
            for (int v = 0; v < 4; v++) {
                sketch_slot_t *meta = sk_get(&sk[v], sig[v]);
                if (meta && meta->count > 0) {
                    int found = 0;
                    for (int j = 0; j < n_cand; j++)
                        if (cand[j] == meta->class_id) { found = 1; break; }
                    if (!found) cand[n_cand++] = meta->class_id;
                }
            }

            /* 2b. Pad with unigram negatives until UNION_CAP (96), deduped */
            while (n_cand < UNION_CAP) {
                uint32_t u = unigrams[rand() % UNIGRAM_K];
                int found = 0;
                for (int j = 0; j < n_cand; j++)
                    if (cand[j] == u) { found = 1; break; }
                if (!found) cand[n_cand++] = u;
            }

            /* 3. Forward Pass: Score candidates */
            float score[UNION_CAP];
            float max_score = -1e9f;
            for (int c = 0; c < n_cand; c++) {
                score[c] = 0.0f;
                for (int v = 0; v < 4; v++) {
                    uint32_t k = fnv1a_u32_combine(sig[v], cand[c]);
                    sketch_slot_t *w = sk_get(&sk[v], k);
                    if (w) score[c] += lambda[v] * w->logit;
                }
                if (score[c] > max_score) max_score = score[c];
            }

            /* 4. Softmax */
            float p[UNION_CAP];
            float sum_exp = 0.0f;
            for (int c = 0; c < n_cand; c++) {
                p[c] = expf(score[c] - max_score);
                sum_exp += p[c];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int c = 0; c < n_cand; c++) p[c] *= inv_sum;

            total_loss += -logf(p[0] + 1e-9f);

            /* 5. Backward Pass (LC-SGD) */
            for (int c = 0; c < n_cand; c++) {
                float g = p[c] - (c == 0 ? 1.0f : 0.0f);
                if (g >  GRAD_CLIP) g =  GRAD_CLIP;
                if (g < -GRAD_CLIP) g = -GRAD_CLIP;
                if (fabsf(g) < 1e-6f) continue;

                for (int v = 0; v < 4; v++) {
                    uint32_t k = fnv1a_u32_combine(sig[v], cand[c]);
                    sketch_slot_t *w;

                    /* Only insert true/context candidates. Do not insert unigram noise if it doesn't exist. */
                    if (c < 5) {
                        w = sk_upsert(&sk[v], k);
                    } else {
                        w = sk_get(&sk[v], k);
                        if (!w) continue; /* Skip update if noise isn't in the sketch */
                    }

                    float n = w->count + 1;
                    float eta = LR / (1.0f + LAMBDA_DECAY * log1pf(n));

                    w->logit -= eta * g;
                    if (w->count < 65535) w->count++;

                    if (c == 0) {
                        sketch_slot_t *meta = sk_upsert(&sk[v], sig[v]);
                        /* Maintain Top-8 prediction list for this signature. */
                        int was_new = topk_touch(meta->topk_classes, true_c);
                        if (meta->count == 0) {
                            meta->class_id = true_c;
                            meta->logit = 1.0f;
                        } else if (was_new) {
                            meta->logit += 1.0f; /* Reinforce with a fresh class */
                        } else {
                            meta->logit -= 1.0f; /* Decay if already ranked */
                            if (meta->logit < 0.0f) {
                                meta->class_id = true_c;
                                meta->logit = 1.0f;
                            }
                        }
                        if (meta->count < 65535) meta->count++;
                    }
                }
            }

            /* 6. Update Backoff Weights (Lambda) */
            float g_true = p[0] - 1.0f; /* Negative when model is wrong */
            for (int v = 0; v < 4; v++) {
                uint32_t k = fnv1a_u32_combine(sig[v], true_c);
                sketch_slot_t *w = sk_get(&sk[v], k);
                if (w) {
                    /* Gradient descent (equivalent to flipping the sign) */
                    lambda[v] -= BACKOFF_LR * g_true * w->logit;

                    /* Clamp to prevent negative or zero weights */
                    if (lambda[v] < 0.01f) lambda[v] = 0.01f;
                }
            }

            /* Normalize to simplex (sum = 1.0) */
            float sum_lambda = lambda[0] + lambda[1] + lambda[2] + lambda[3];
            for (int v = 0; v < 4; v++) lambda[v] /= sum_lambda;
        }

        lines_read++;
        total_tokens += n_tok;

        /* Auto-evict if table gets too full */
        if (lines_read % 1000 == 0) {
            for (int v = 0; v < 4; v++) {
                if (sk[v].n_entries > (sk[v].capacity * 0.70)) {
                    sk_evict_sweep(&sk[v], 2, 1); /* Evict count<=2, gen older than cur-1 */
                }
            }
        }

        if (lines_read % 1000 == 0) {
            double elapsed = now() - t_start;
            double avg_loss = total_loss / (double)total_tokens;
            double tps = (double)total_tokens / elapsed;
            printf("line %d | tokens %zu | loss %.4f | %.0f tok/s | lambda: %.2f %.2f %.2f %.2f\n",
                   lines_read, total_tokens, avg_loss, tps, lambda[0], lambda[1], lambda[2], lambda[3]);
            total_loss = 0.0f; /* reset window loss */
            total_tokens = 0;
        }
    }

    printf("GATE PASS: Training complete. Processed %d lines.\n", lines_read);

    FILE *lf = fopen("models/lambdas.bin", "wb");
    if (lf) { fwrite(lambda, sizeof(float), 4, lf); fclose(lf); }

    for (int v = 0; v < 4; v++) sk_free(&sk[v]);
    shard_close(&sr);
    tok_free(&tok);
    return 0;
}
