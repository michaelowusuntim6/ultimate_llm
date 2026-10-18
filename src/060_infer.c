#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define UNION_CAP 256
#define KN_DISCOUNT 0.75f

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed; h ^= t1; h *= 0x01000193u; h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}

static uint32_t rng_state = 12345;
static inline uint32_t fast_rand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

/* discounted prob of cls from a meta slot (0 if unseen) */
static float kn_order(const sketch_slot_t *meta, uint32_t cls) {
    if (!meta || meta->count == 0) return 0.0f;
    float c = (float)meta->count;
    for (int k = 0; k < 8; k++)
        if (meta->topk_classes[k] == cls && meta->topk_counts[k] > 0) {
            float d = (float)meta->topk_counts[k];
            float disc = d - KN_DISCOUNT; if (disc < 0) disc = 0;
            return disc / c;
        }
    return 0.0f;
}

/* True interpolated Kneser-Ney with context-dependent backoff weight.
   meta->logit stores #distinct continuations (types); count stores tokens.
   gamma(h) = D * types(h) / count(h); P_KN(w|h) = disc/count + gamma * P_lower.
   A NULL context yields disc=0 and gamma=1 -> full weight to lower order. */
static double kn_rec(const sketch_slot_t *m[3], const float lam[3],
                     uint32_t w, int v, double p_uni) {
    if (v < 0) return p_uni;                       /* unigram base */
    const sketch_slot_t *meta = m[v];
    double p_high = kn_order(meta, w);             /* discounted term, 0 if unseen */
    double gamma = 1.0;                            /* default: pass all weight down */
    if (meta && meta->count > 0) {
        double types = meta->logit > 0 ? meta->logit : 1.0;
        gamma = (double)KN_DISCOUNT * types / (double)meta->count;
        if (gamma > 1.0) gamma = 1.0;
    }
    double mix = lam[v] >= 0 ? (double)lam[v] : 1.0; /* learned scale, fallback 1 */
    return p_high + gamma * mix * kn_rec(m, lam, w, v - 1, p_uni);
}

int main(int argc, char **argv) {
    tokenizer_t tok;
    if (tok_init(&tok, "/home/mike/Vocab/vocab_english.txt",
                      "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) return 1;

    sketch_t sk[3];
    char path[256];
    for (int v = 0; v < 3; v++) {
        snprintf(path, sizeof(path), "models/view_%d.db", v);
        if (sk_init(&sk[v], path, 8388608) != 0) return 1;
    }

    float lam[3] = {1.0f, 1.0f, 1.0f};
    FILE *wf = fopen("models/kneser_ney_weights.bin", "rb");
    if (wf) { float tmp[3]; if (fread(tmp, sizeof(float), 3, wf) == 3) { for(int v=0;v<3;v++) lam[v] = tmp[v] > 0 ? tmp[v] : 1.0f; } fclose(wf); }

    /* unigram distribution over top 100 */
    uint32_t top_u[100]; int n_top = 0;
    FILE *uf = fopen("models/top_unigrams.bin", "rb");
    if (uf) { n_top = (int)fread(top_u, sizeof(uint32_t), 100, uf); fclose(uf); }
    if (n_top < 100) { for (int i=n_top;i<100;i++) top_u[i]=(uint32_t)(i+1); n_top=100; }
    float uni_p[100]; double utot = 0;
    for (int i = 0; i < n_top; i++) { uni_p[i] = (float)(n_top - i); utot += uni_p[i]; } /* rank-weighted */
    if (utot <= 0) utot = 1;

    uint32_t seeds[3] = {0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35};
    float temp = 0.4f;   /* sharpen distribution so high-prob continuations win */

    const char *prompt = (argc > 1) ? argv[1] : "The";
    int max_gen = (argc > 2) ? atoi(argv[2]) : 50;

    int32_t tokens[4096];
    int n_tok = tok_encode(&tok, prompt, (int32_t)strlen(prompt), tokens, 4096);
    if (n_tok < 1) { printf("Empty prompt.\n"); return 1; }

    printf("Prompt: %s\nGenerating %d tokens...\n", prompt, max_gen);

    char out_buf[16384];
    int out_len = 0;
    { int dl = tok_decode(&tok, tokens, n_tok, out_buf, sizeof(out_buf)-1); if (dl>0) out_len = dl; }

    /* word-only context for signature formation (bridge over spaces/punct bytes) */
    static int32_t wctx[4096]; int nw = 0;
    for (int j = 0; j < n_tok; j++) if (tokens[j]>=0 && tokens[j]<tok.n_word_tokens) wctx[nw++]=tokens[j];

    for (int step = 0; step < max_gen; step++) {
        if (nw < 3) break;

        uint32_t t1=(uint32_t)wctx[nw-1], t2=(uint32_t)wctx[nw-2], t3=(uint32_t)wctx[nw-3];
        uint32_t sig[3];
        sig[0]=hash_sig(t1,0,seeds[0]);
        sig[1]=hash_sig(t1,t2,seeds[1]);
        sig[2]=hash_sig(hash_sig(t1,t2,seeds[1]),t3,seeds[2]);

        const sketch_slot_t *m[3];
        for (int v = 0; v < 3; v++) m[v] = sk_get(&sk[v], sig[v]);

        /* candidate set: seed with unigrams (word tokens only) */
        uint32_t cand[UNION_CAP]; int n_cand = 0;
        uint32_t special_base = (uint32_t)tok.n_vocab - 5;
        for (int j = 0; j < n_top && n_cand < UNION_CAP; j++) {
            uint32_t cl = top_u[j];
            if (cl >= (uint32_t)tok.n_word_tokens || cl >= special_base) continue; /* words only */
            int found=0; for(int q=0;q<n_cand;q++)if(cand[q]==cl){found=1;break;}
            if(!found) cand[n_cand++]=cl;
        }
        /* then add sketch Top-K candidates (word tokens only) */
        for (int v = 0; v < 3; v++) {
            if (!m[v] || m[v]->count == 0) continue;
            for (int k = 0; k < 8 && n_cand < UNION_CAP; k++) {
                uint32_t cl = m[v]->topk_classes[k];
                if (cl >= (uint32_t)tok.n_word_tokens || m[v]->topk_counts[k] == 0) continue;
                int found=0; for(int q=0;q<n_cand;q++)if(cand[q]==cl){found=1;break;}
                if(!found) cand[n_cand++]=cl;
            }
        }
        if (n_cand == 0) break;

        /* true interpolated KN recursion over the FULL candidate set */
        float score[UNION_CAP], mx = -1e30f;
        for (int c = 0; c < n_cand; c++) {
            uint32_t w = cand[c];
            double p_uni = 1e-9;
            for (int j = 0; j < n_top; j++) if (top_u[j]==w) { p_uni = (double)uni_p[j]/utot; break; }
            double s = kn_rec(m, lam, w, 2, p_uni);
            score[c] = (float)s;
            if (score[c] > mx) mx = score[c];
        }

        /* top-k restricted sampling: keep only the K highest-scoring candidates */
        #define TOPK 5
        int idx[UNION_CAP]; for (int c=0;c<n_cand;c++) idx[c]=c;
        for (int a=0;a<n_cand && a<TOPK+1;a++)
            for (int b=a+1;b<n_cand;b++)
                if (score[idx[b]] > score[idx[a]]) { int t=idx[a];idx[a]=idx[b];idx[b]=t; }
        int kk = n_cand < TOPK ? n_cand : TOPK;
        float p[TOPK], sum = 0;
        for (int c = 0; c < kk; c++) { p[c] = expf((score[idx[c]]-mx)/temp); sum += p[c]; }
        float r = (float)(fast_rand() % 100000) / 100000.0f * sum;
        float cum = 0; uint32_t next = cand[idx[0]];
        for (int c = 0; c < kk; c++) { cum += p[c]; if (r <= cum) { next = cand[idx[c]]; break; } }

        tokens[n_tok++] = (int32_t)next;
        if ((uint32_t)next < (uint32_t)tok.n_word_tokens && nw < 4096) wctx[nw++] = (int32_t)next;

        /* word-token space injection: insert a space before a dictionary word if
           the previous printed char isn't already whitespace. */
        int32_t nt = (int32_t)next;
        char piece[64]; int pl = tok_decode(&tok, &nt, 1, piece, sizeof(piece)-1);
        if (pl > 0) {
            piece[pl]='\0';
            int is_word = ((uint32_t)nt < (uint32_t)tok.n_word_tokens);
            int prev_space = (out_len>0 && (out_buf[out_len-1]==' '||out_buf[out_len-1]=='\t'||out_buf[out_len-1]=='\n'));
            if (is_word && out_len>0 && !prev_space)
                if (out_len+1<(int)sizeof(out_buf)) out_buf[out_len++]=' ';
            if (out_len + pl < (int)sizeof(out_buf)) { memcpy(out_buf+out_len, piece, pl); out_len += pl; }
        }
    }

    out_buf[out_len] = '\0';
    printf("\n--- OUTPUT ---\n%s\n--------------\n", out_buf);

    for (int v = 0; v < 3; v++) sk_free(&sk[v]);
    tok_free(&tok);
    return 0;
}
