#define _POSIX_C_SOURCE 200809L
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

/* Build models/vocab_expanded.txt = existing vocab_english.txt + all TinyStories
   words not already present. Words are whitespace-split, lowercased, stripped of
   surrounding non-letter chars. If total > MAX_VOCAB, drop words seen < MIN_FREQ. */

#define MAX_VOCAB 500000
#define MIN_FREQ 3
#define CAP 700000

typedef struct { char *w; int len; int freq; int base; } word_t;
static word_t *tbl; static size_t tmask, tn;

static uint32_t wh(const char *s, int n){uint32_t h=2166136261u;for(int i=0;i<n;i++){h^=(unsigned char)s[i];h*=0x01000193u;}return h?h:1;}
static int find_or_add(const char *s,int n,int create){
    size_t idx=wh(s,n)&tmask;
    while(tbl[idx].w){ if(tbl[idx].len==n&&memcmp(tbl[idx].w,s,n)==0) return (int)idx; idx=(idx+1)&tmask; }
    if(!create) return -1;
    char *c=malloc(n+1); memcpy(c,s,n); c[n]=0; tbl[idx].w=c; tbl[idx].len=n; tbl[idx].freq=0; tn++;
    return (int)idx;
}

/* extract core word letters from a whitespace token (strip leading/trailing junk) */
static int norm_word(char *tok, int tl, char *out){
    int s=0,e=tl; while(s<e && !isalpha((unsigned char)tok[s])) s++; while(e>s && !isalnum((unsigned char)tok[e-1]) && tok[e-1]!='\'') e--;
    int o=0; for(int i=s;i<e;i++){ char c=tok[i]; if(c=='\''){ if(o==0) continue; out[o++]=' '; /* map apostrophe to space to keep simple */ } else out[o++]=tolower((unsigned char)c);}
    /* collapse any internal spaces (from apostrophe mapping): just skip them */
    int f=0; for(int i=0;i<o;i++) if(out[i]!=' ') out[f++]=out[i];
    return f;
}

int main(void){
    size_t sz=1; while(sz<CAP*2) sz<<=1;
    tbl=calloc(sz,sizeof(word_t)); tmask=sz-1;

    /* seed with existing vocab (freq marked high so they're always kept) */
    FILE *vf=fopen("/home/mike/Vocab/vocab_english.txt","r");
    if(!vf){perror("vocab_english.txt");return 1;}
    char *ln=NULL; size_t lc=0; ssize_t l;
    while((l=getline(&ln,&lc,vf))>0){ while(l>0&&(ln[l-1]=='\n'||ln[l-1]=='\r')) l--; if(l<=0)continue; int id=find_or_add(ln,(int)l,1); if(id>=0){ tbl[id].freq=MIN_FREQ; tbl[id].base=1; } }
    fclose(vf); free(ln);
    fprintf(stderr,"[expand] seeded %zu base words\n",tn);

    /* stream TinyStories, count new words */
    shard_reader_t sr;
    if(shard_open(&sr,"dataset/ts-ds/tinystories_train.txt",64)!=0) return 1;
    char *line; size_t len; long lines=0;
    while(shard_next_line(&sr,&line,&len)>0){
        lines++;
        char *p=line; while(*p){ while(*p==' '||*p=='\t') p++; char *st=p; while(*p&&*p!=' '&&*p!='\t') p++; int tl=p-st; if(tl<=0)continue;
            char buf[128]; int nl=norm_word(st,tl<127?tl:127,buf); if(nl<=0)continue;
            int id=find_or_add(buf,nl,1); if(id>=0) tbl[id].freq++;
        }
    }
    shard_close(&sr);
    fprintf(stderr,"[expand] scanned %ld lines, %zu unique words\n",lines,tn);

    /* write expanded vocab: always keep base words; keep new words with freq>=MIN_FREQ */
    FILE *of=fopen("models/vocab_expanded.txt","wb");
    if(!of){perror("vocab_expanded.txt");return 1;}
    int written=0, newkept=0;
    for(size_t i=0;i<tmask+1;i++){
        if(!tbl[i].w) continue;
        int keep = tbl[i].base || (tbl[i].freq>=MIN_FREQ);
        if(keep){ fwrite(tbl[i].w,tbl[i].len,1,of); fputc('\n',of); written++; if(!tbl[i].base) newkept++; }
    }
    fclose(of);
    fprintf(stderr,"[expand] wrote %d words (%d new) to models/vocab_expanded.txt\n",written,newkept);
    return 0;
}
