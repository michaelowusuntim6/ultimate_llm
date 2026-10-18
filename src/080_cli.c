#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define UNION_CAP 256
#define KN_DISCOUNT 0.75f
#define TOPK 5
#define GEN_TOKENS 24

static inline uint32_t hash_sig(uint32_t t1, uint32_t t2, uint32_t seed) {
    uint32_t h = seed; h ^= t1; h *= 0x01000193u; h ^= t2; h *= 0x01000193u;
    return h ? h : 1;
}
static uint32_t rng_state = 0xC0FFEE;
static inline uint32_t fast_rand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}
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
static double kn_rec(const sketch_slot_t *m[3], const float lam[3],
                     uint32_t w, int v, double p_uni) {
    if (v < 0) return p_uni;
    const sketch_slot_t *meta = m[v];
    double p_high = kn_order(meta, w);
    double gamma = 1.0;
    if (meta && meta->count > 0) {
        double types = meta->logit > 0 ? meta->logit : 1.0;
        gamma = (double)KN_DISCOUNT * types / (double)meta->count;
        if (gamma > 1.0) gamma = 1.0;
    }
    double mix = lam[v] >= 0 ? (double)lam[v] : 1.0;
    return p_high + gamma * mix * kn_rec(m, lam, w, v - 1, p_uni);
}

/* generate up to GEN_TOKENS words from current context tokens[0..n_tok) */
static void generate(tokenizer_t *tok, sketch_t sk[3], const float lam[3],
                    const uint32_t top_u[100], int n_top, double utot,
                    const float uni_p[100], uint32_t seeds[3],
                    int32_t *tokens, int *n_tok_io) {
    int n_tok = *n_tok_io;
    uint32_t special_base = (uint32_t)tok->n_vocab - 5;
    /* word-only context for signatures */
    static int32_t wctx[4096]; int nw = 0;
    for (int j = 0; j < n_tok; j++) if (tokens[j]>=0 && tokens[j]<tok->n_word_tokens) wctx[nw++]=tokens[j];
    for (int step = 0; step < GEN_TOKENS; step++) {
        /* Pad short contexts with BOS(0) so signature hashing stays in-bounds
           and the recursion still falls back to lower orders / unigrams. */
        uint32_t t1 = nw>=1 ? (uint32_t)wctx[nw-1] : 0;
        uint32_t t2 = nw>=2 ? (uint32_t)wctx[nw-2] : 0;
        uint32_t t3 = nw>=3 ? (uint32_t)wctx[nw-3] : 0;
        uint32_t sig[3]={hash_sig(t1,0,seeds[0]),hash_sig(t1,t2,seeds[1]),hash_sig(hash_sig(t1,t2,seeds[1]),t3,seeds[2])};
        const sketch_slot_t *m[3]; for(int v=0;v<3;v++) m[v]=sk_get(&sk[v],sig[v]);

        /* Candidate set: ALWAYS seed with top unigrams (word tokens only), so a
           short/empty context still has something to score. */
        uint32_t cand[UNION_CAP]; int n_cand=0;
        for(int j=0;j<n_top&&n_cand<UNION_CAP;j++){uint32_t cl=top_u[j];if(cl>=(uint32_t)tok->n_word_tokens||cl>=special_base)continue;int f=0;for(int q=0;q<n_cand;q++)if(cand[q]==cl)f=1;if(!f)cand[n_cand++]=cl;}
        for(int v=0;v<3;v++){if(!m[v]||m[v]->count==0)continue;for(int k=0;k<8&&n_cand<UNION_CAP;k++){uint32_t cl=m[v]->topk_classes[k];if(cl>=(uint32_t)tok->n_word_tokens||m[v]->topk_counts[k]==0)continue;int f=0;for(int q=0;q<n_cand;q++)if(cand[q]==cl)f=1;if(!f)cand[n_cand++]=cl;}}
        if(n_cand==0) break;

        float score[UNION_CAP], mx=-1e30f;
        for(int c=0;c<n_cand;c++){uint32_t w=cand[c];double pu=1e-9;for(int j=0;j<n_top;j++)if(top_u[j]==w){pu=(double)uni_p[j]/utot;break;}score[c]=(float)kn_rec(m,lam,w,2,pu);if(score[c]>mx)mx=score[c];}

        int idx[UNION_CAP]; for(int c=0;c<n_cand;c++)idx[c]=c;
        for(int a=0;a<n_cand&&a<TOPK+1;a++)for(int b=a+1;b<n_cand;b++)if(score[idx[b]]>score[idx[a]]){int t=idx[a];idx[a]=idx[b];idx[b]=t;}
        int kk=n_cand<TOPK?n_cand:TOPK;
        float p[TOPK],sum=0; for(int c=0;c<kk;c++){p[c]=expf((score[idx[c]]-mx)/0.5f);sum+=p[c];}

        /* Sample, resampling until the token decodes to a non-empty string. */
        uint32_t next = cand[idx[0]]; char word[64]; int wl = 0;
        for (int attempt=0; attempt<kk*3; attempt++) {
            float r=(float)(fast_rand()%100000)/100000.0f*sum; float cum=0; uint32_t pick=cand[idx[0]];
            for(int c=0;c<kk;c++){cum+=p[c];if(r<=cum){pick=cand[idx[c]];break;}}
            int32_t pid=(int32_t)pick;
            int l=tok_decode(tok,&pid,1,word,sizeof(word)-1);
            if(l>0 && !(l==1 && (word[0]==' '||word[0]=='\t'))){ wl=l; next=pick; break; }
        }
        if(wl<=0){ /* last-resort: take highest-scoring candidate that decodes */
            int32_t pid=(int32_t)cand[idx[0]]; wl=tok_decode(tok,&pid,1,word,sizeof(word)-1); next=cand[idx[0]];
            if(wl<=0) break;
        }
        word[wl]='\0';

        int32_t nt=(int32_t)next;
        int is_word=((uint32_t)nt<(uint32_t)tok->n_word_tokens);
        if(is_word && step>0) printf(" ");
        printf("\033[0m%s", word); fflush(stdout);
        tokens[n_tok++]=nt;
        if(is_word && nw<4096) wctx[nw++]=nt;
        if(n_tok>=4096) break;
    }
    *n_tok_io = n_tok;
}

int main(void) {
    tokenizer_t tok;
    if (tok_init(&tok, "models/vocab_expanded.txt",
                      "/home/mike/Vocab/byte_fallback_tokens.txt") != 0) return 1;
    sketch_t sk[3]; char path[256];
    for (int v = 0; v < 3; v++) { snprintf(path,sizeof(path),"models/view_%d.db",v); if(sk_init(&sk[v],path,8388608)!=0) return 1; }
    float lam[3]={1,1,1}; FILE*wf=fopen("models/kneser_ney_weights.bin","rb");
    if(wf){float tmp[3];if(fread(tmp,sizeof(float),3,wf)==3)for(int v=0;v<3;v++)lam[v]=tmp[v]>0?tmp[v]:1.0f;fclose(wf);}
    uint32_t top_u[100]; int n_top=0; FILE*uf=fopen("models/top_unigrams.bin","rb");
    if(uf){n_top=(int)fread(top_u,sizeof(uint32_t),100,uf);fclose(uf);} if(n_top<100){for(int i=n_top;i<100;i++)top_u[i]=(uint32_t)i;n_top=100;}
    float uni_p[100]; double utot=0; for(int i=0;i<n_top;i++){uni_p[i]=(float)(n_top-i);utot+=uni_p[i];} if(utot<=0)utot=1;
    uint32_t seeds[3]={0x9E3779B1,0x85EBCA6B,0xC2B2AE35};

    /* persistent rolling context across turns */
    static int32_t ctx[4096]; static int ctx_n = 0;

    printf("\033[1;36m");
    printf("╭──────────────────────────────────────────────────────────╮\n");
    printf("│  Ultimate LLM (SLLM) — MatMul-Free Interactive CLI       │\n");
    printf("│  Type '/exit' to quit, '/clear' to reset context.        │\n");
    printf("╰──────────────────────────────────────────────────────────╯\n");
    printf("\033[0m");
    printf("\033[2m vocab=%d  model=trigram Kneser-Ney  press Ctrl-D or /exit to leave \033[0m\n\n", tok.n_vocab);

    char input[1024];
    while (1) {
        printf("\033[1;32m> \033[0m"); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input,"\n")] = 0;
        if (strcmp(input,"/exit")==0) break;
        if (strcmp(input,"/clear")==0) { ctx_n=0; printf("\033[2J\033[H"); continue; }
        if (strlen(input)==0) continue;

        int32_t pt[512];
        int pn = tok_encode(&tok, input, (int)strlen(input), pt, 512);
        if (pn < 1) { printf("Prompt too short.\n"); continue; }
        /* append prompt to rolling context */
        for (int i=0;i<pn && ctx_n<4096;i++) ctx[ctx_n++]=pt[i];

        printf("\033[1;34mAI:\033[0m "); fflush(stdout);
        generate(&tok, sk, lam, top_u, n_top, utot, uni_p, seeds, ctx, &ctx_n);
        printf("\n\n");
    }
    printf("\033[1;36mGoodbye!\033[0m\n");
    for (int v=0;v<3;v++) sk_free(&sk[v]);
    tok_free(&tok);
    return 0;
}
