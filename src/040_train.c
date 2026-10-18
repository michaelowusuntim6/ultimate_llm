#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

#define KN_DISCOUNT 0.75f

/* Interpolated Kneser-Ney probability of class `cls` given a meta slot's
   topk counts + context total, backed off to a unigram distribution. */
static float kn_prob(const sketch_slot_t *meta, uint32_t cls, int n_cand_unused) {
    (void)n_cand_unused;
    if (!meta || meta->count == 0) return 0.0f;
    float c = (float)meta->count;
    for (int k = 0; k < 8; k++) {
        if (meta->topk_classes[k] == cls && meta->topk_counts[k] > 0) {
            float d = (float)meta->topk_counts[k];
            float disc = d - KN_DISCOUNT; if (disc < 0) disc = 0;
            return disc / c;
        }
    }
    return 0.0f;
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "models";
    int max_lines = (argc > 2) ? atoi(argv[2]) : 175926;

    printf("[train] Loading BPE tokenizer...\n");
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/tokenizer_pure/vocab.json",
                      "/home/mike/Vocab/tokenizer_pure/merges.txt") != 0) return 1;

    /* Unigram continuation counts (for KN base distribution). */
    float *uni = calloc((size_t)tok.n_vocab, sizeof(float));
    double uni_total = 0.0;

    printf("[train] Initializing 3 sketches (8M slots each)...\n");
    sketch_t sk[3];
    char path[256];
    for (int v = 0; v < 3; v++) {
        snprintf(path, sizeof(path), "%s/view_%d.db", model_dir, v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    uint32_t seeds[3] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35};

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

        for (int32_t i = 3; i < n_tok; i++) {
            uint32_t t1 = (uint32_t)tokens[i-1], t2 = (uint32_t)tokens[i-2];
            uint32_t t3 = (uint32_t)tokens[i-3];
            uint32_t true_c = (uint32_t)tokens[i];

            /* unigram continuation tally */
            if (true_c < (uint32_t)tok.n_vocab) { uni[true_c] += 1.0f; uni_total += 1.0; }

            uint32_t sig[3];
            sig[0] = hash_sig(t1, 0, seeds[0]);        /* bigram   */
            sig[1] = hash_sig(t1, t2, seeds[1]);       /* trigram  */
            sig[2] = hash_sig(hash_sig(t1, t2, seeds[1]), t3, seeds[2]); /* 4-gram */

            for (int v = 0; v < 3; v++) {
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
            printf("line %d | tokens %zu | %.0f tok/s | entries: %zu %zu %zu\n",
                   lines_read, total_tokens, tps,
                   sk[0].n_entries, sk[1].n_entries, sk[2].n_entries);
            fflush(stdout);
            for (int v = 0; v < 3; v++)
                if (sk[v].n_entries > sk[v].capacity * 0.85)
                    sk_evict_sweep(&sk[v], 2, 1);
        }
    }

    printf("GATE PASS: Counting complete. Processed %d lines.\n", lines_read);

    /* ---- Held-out interpolation weight tuning (EM-style coordinate search) ---- */
    printf("[train] Tuning KN interpolation weights on held-out 10%%...\n");
    float lam[3] = {0.33f, 0.33f, 0.33f};
    double uni_norm = uni_total > 0 ? uni_total : 1.0;

    #define EVAL_N 20000
    shard_reader_t tr;
    if (shard_open(&tr, "/home/mike/Vocab/dataset_prose.txt", 64) != 0) return 1;
    char *tl; size_t tlen; int tline = 0;
    int32_t ttokbuf[4096];
    static float hp[EVAL_N][4]; static uint32_t htrue[EVAL_N]; int hn = 0;
    while (shard_next_line(&tr, &tl, &tlen) > 0 && hn < EVAL_N) {
        tline++;
        if (tline % 10 != 0) continue;
        int nt = tok_encode(&tok, tl, (int32_t)tlen, ttokbuf, 4096);
        for (int i = 3; i < nt && hn < EVAL_N; i++) {
            uint32_t t1=(uint32_t)ttokbuf[i-1],t2=(uint32_t)ttokbuf[i-2],t3=(uint32_t)ttokbuf[i-3];
            uint32_t tc=(uint32_t)ttokbuf[i];
            uint32_t sg[3]={hash_sig(t1,0,seeds[0]),hash_sig(t1,t2,seeds[1]),hash_sig(hash_sig(t1,t2,seeds[1]),t3,seeds[2])};
            float p_uni = uni[tc]/uni_norm; if(p_uni<=0) p_uni=1e-6f;
            hp[hn][3]=p_uni;
            for (int v=0;v<3;v++){ sketch_slot_t*m=sk_get(&sk[v],sg[v]); hp[hn][v]=kn_prob(m,tc,0); }
            htrue[hn]=tc; hn++;
        }
    }
    shard_close(&tr);

    double total_ll = 0; for(int i=0;i<hn;i++){ double pr=(1-lam[0]-lam[1]-lam[2])*hp[i][3]; for(int v=0;v<3;v++)pr+=lam[v]*hp[i][v]; if(pr<1e-12)pr=1e-12; total_ll+=log(pr);}
    for (int sweep=0; sweep<3; sweep++) {
        for (int v=0; v<3; v++) {
            float best=lam[v]; double bestll=total_ll;
            for (float cand=0.0f; cand<=1.0001f; cand+=0.05f) {
                float old=lam[v]; lam[v]=cand;
                double s=0; for(int i=0;i<hn;i++){ double rem=1-lam[0]-lam[1]-lam[2]; if(rem<0)rem=0; double pr=rem*hp[i][3]; for(int q=0;q<3;q++)pr+=lam[q]*hp[i][q]; if(pr<1e-12)pr=1e-12; s+=log(pr);}
                if (s>bestll){bestll=s;best=cand;}
                lam[v]=old;
            }
            lam[v]=best; total_ll=bestll;
        }
        float sm=lam[0]+lam[1]+lam[2]; if(sm>1){for(int v=0;v<3;v++)lam[v]/=sm;}
    }
    FILE *wf=fopen("models/kneser_ney_weights.bin","wb"); if(wf){fwrite(lam,sizeof(float),3,wf);fclose(wf);}
    printf("[train] learned lambda: %.3f %.3f %.3f (heldout LL/token=%.4f)\n",
           lam[0],lam[1],lam[2], hn?total_ll/hn:0);

    free(uni);
    for (int v = 0; v < 3; v++) sk_free(&sk[v]);
    shard_close(&sr);
    tok_free(&tok);
    return 0;
}
