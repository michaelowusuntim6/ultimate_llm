#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── GPT-2 byte <-> unicode tables ─────────────────────────────────────── */
static uint32_t bytes_to_unicode_tbl[256];   /* byte -> codepoint */
static int      unicode_to_bytes_cp[0x110000]; /* codepoint -> byte, -1 if none */
static int      tables_built = 0;

static void build_byte_tables(void) {
    if (tables_built) return;
    /* Printable bytes map to themselves; the remaining 68 bytes (controls +
       space + DEL + latin-1 gaps) map to codepoints 256..323 in ascending
       byte order. This matches tokenizer_pure/vocab.json's single-char tokens. */
    int printable[256]; for (int i=0;i<256;i++) printable[i]=0;
    for (int b = 33; b <= 126; b++) printable[b]=1;
    for (int b = 161; b <= 172; b++) printable[b]=1;
    for (int b = 174; b <= 255; b++) printable[b]=1;
    int k = 0;
    for (int b = 0; b < 256; b++) {
        if (printable[b]) bytes_to_unicode_tbl[b] = (uint32_t)b;
        else              bytes_to_unicode_tbl[b] = (uint32_t)(256 + k++);
    }
    for (int i = 0; i < 0x110000; i++) unicode_to_bytes_cp[i] = -1;
    for (int b = 0; b < 256; b++) unicode_to_bytes_cp[bytes_to_unicode_tbl[b]] = b;
    tables_built = 1;
}

/* Encode a UTF-8 string into the byte-level representation (one codepoint per
   entry via the table), returning raw original bytes. We operate directly on
   raw bytes: decode utf8 -> codepoints -> map each cp to its table byte. */
static int utf8_decode(const char *s, int len, int *pos, uint32_t *cp) {
    const unsigned char *u = (const unsigned char *)s;
    int i = *pos;
    if (i >= len) return 0;
    unsigned c = u[i++];
    if (c < 0x80) { *cp = c; *pos = i; return 1; }
    if ((c >> 5) == 0x6 && i < len) { *cp = ((c & 0x1F) << 6) | (u[i++] & 0x3F); *pos = i; return 1; }
    if ((c >> 4) == 0xE && i + 1 < len) { unsigned c1=u[i++],c2=u[i++]; *cp=((c&0x0F)<<12)|((c1&0x3F)<<6)|(c2&0x3F); *pos=i; return 1; }
    if ((c >> 3) == 0x1E && i + 2 < len) { unsigned c1=u[i++],c2=u[i++],c3=u[i++]; *cp=((c&0x07)<<18)|((c1&0x3F)<<12)|((c2&0x3F)<<6)|(c3&0x3F); *pos=i; return 1; }
    *cp = c; *pos = i; return 1;
}

static int encode_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* Map input text -> byte-level ids array (each element is a single-byte token id later). */
static int text_to_bytes(const char *s, int len, uint8_t **out_bytes, int *out_n) {
    build_byte_tables();
    uint8_t *buf = malloc((size_t)len + 1);
    if (!buf) return -1;
    int n = 0, pos = 0;
    while (pos < len) {
        uint32_t cp;
        if (!utf8_decode(s, len, &pos, &cp)) break;
        int b = (cp < 0x110000) ? unicode_to_bytes_cp[cp] : -1;
        buf[n++] = (uint8_t)(b >= 0 ? b : (unsigned char)s[pos-1]);
    }
    *out_bytes = buf; *out_n = n;
    return 0;
}

/* ── hash map for token string -> id ───────────────────────────────────── */
static uint32_t str_hash(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 0x01000193u; }
    return h ? h : 1;
}
static void hm_put(tokenizer_t *t, const char *s, size_t n, int32_t id) {
    uint32_t k = str_hash(s, n);
    size_t idx = k & t->hm_mask;
    while (t->hm_key[idx]) idx = (idx + 1) & t->hm_mask;
    t->hm_key[idx] = k; t->hm_val[idx] = id;
}
static int32_t hm_get(const tokenizer_t *t, const char *s, size_t n) {
    uint32_t k = str_hash(s, n);
    size_t idx = k & t->hm_mask;
    while (t->hm_key[idx]) {
        if (t->hm_key[idx] == k && strcmp(t->token_str[t->hm_val[idx]], s) == 0)
            return t->hm_val[idx];
        idx = (idx + 1) & t->hm_mask;
    }
    return -1;
}

/* ── JSON vocab loader (flat {"tok":id,...}) ───────────────────────────── */
static int load_vocab_json(tokenizer_t *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; } buf[sz] = '\0'; fclose(f);

    /* count entries */
    int cap = 16384;
    t->token_str = malloc(cap * sizeof(char*));
    t->token_bytes = malloc(cap * sizeof(uint8_t*));
    t->token_len = malloc(cap * sizeof(int32_t));
    t->token_text = malloc(cap * sizeof(char*));
    t->text_len = malloc(cap * sizeof(int32_t));
    int max_id = 0;
    /* collect max id first */
    {
        char *p = buf;
        while (*p) {
            char *q = strchr(p, ':'); if (!q) break;
            q++; while (*q==' '||*q=='\n'||*q=='\t') q++;
            int id = atoi(q); if (id > max_id) max_id = id;
            p = strchr(q, ','); if (!p) break; p++;
        }
    }
    t->n_vocab = max_id + 1;
    free(t->token_str); free(t->token_bytes); free(t->token_len);
    free(t->token_text); free(t->text_len);
    t->token_str = calloc((size_t)t->n_vocab, sizeof(char*));
    t->token_bytes = calloc((size_t)t->n_vocab, sizeof(uint8_t*));
    t->token_len = calloc((size_t)t->n_vocab, sizeof(int32_t));
    t->token_text = calloc((size_t)t->n_vocab, sizeof(char*));
    t->text_len = calloc((size_t)t->n_vocab, sizeof(int32_t));
    (void)cap;

    /* parse "key":value pairs from the flat JSON object */
    char *p = strchr(buf, '{'); if (!p) p = buf;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++;
        char *kstart = p; int esc = 0;
        while (*p) { if (*p == '"' && !esc) break; esc = (*p == '\\' && !esc); p++; }
        size_t klen = (size_t)(p - kstart);
        if (*p != '"') break;
        p++; /* past closing quote */
        while (*p && *p != ':') p++;
        if (!*p) break;
        p++;
        while (*p == ' ') p++;
        int id = atoi(p);
        if (klen > 0 && id >= 0 && id < t->n_vocab) {
            char *key = malloc(klen + 1);
            size_t o = 0;
            for (size_t i = 0; i < klen; ) {
                if (kstart[i]=='\\' && i+1<klen) {
                    char nx = kstart[i+1];
                    if (nx=='"'||nx=='\\'||nx=='/') key[o++]=nx, i+=2;
                    else if (nx=='u') { unsigned vv=0; for (int d=0; d<4 && i+2+d<klen; d++){ char hc=kstart[i+2+d]; vv=vv*16+(hc>='a'?hc-'a'+10:(hc>='A'?hc-'A'+10:hc-'0')); } o += (size_t)encode_utf8(vv, key+o); i+=6; }
                    else { key[o++]=kstart[i]; i++; }
                } else key[o++]=kstart[i++];
            }
            key[o]='\0';
            t->token_str[id] = key;
        }
        while (*p && *p != ',' && *p != '}') { if (*p == '"') { p++; while (*p && *p != '"') p++; } if (*p) p++; }
        if (*p == '}') break;
        if (*p == ',') p++;
    }
    free(buf);

    /* build hash map */
    size_t hc = 1; while (hc < (size_t)t->n_vocab * 2) hc <<= 1;
    t->hm_cap = hc; t->hm_mask = hc - 1;
    t->hm_key = calloc(hc, sizeof(uint32_t));
    t->hm_val = calloc(hc, sizeof(int32_t));
    for (int id = 0; id < t->n_vocab; id++)
        if (t->token_str[id]) hm_put(t, t->token_str[id], strlen(t->token_str[id]), id);

    /* predecode token strings: byte-level repr (for matching) + real UTF-8 (for output) */
    build_byte_tables();
    for (int id = 0; id < t->n_vocab; id++) {
        if (!t->token_str[id]) continue;
        uint8_t *b; int bn;
        if (text_to_bytes(t->token_str[id], (int)strlen(t->token_str[id]), &b, &bn) != 0) return -1;
        t->token_bytes[id] = b; t->token_len[id] = bn;
        /* real text: byte-level repr maps back to the ORIGINAL bytes */
        char *txt = malloc((size_t)bn + 1);
        for (int k = 0; k < bn; k++) txt[k] = (char)b[k];
        t->token_text[id] = txt; t->text_len[id] = bn;
    }
    /* byte -> single-byte token id */
    for (int b = 0; b < 256; b++) {
        char one[8] = {0}; int l = encode_utf8(bytes_to_unicode_tbl[b], one);
        one[l] = '\0';
        int32_t id = hm_get(t, one, (size_t)l);
        t->byte_to_id[b] = id;
    }
    /* space has no standalone byte-level token; use Ġ (id 37) as the marker.
       It decodes to a real space via bytes_to_unicode, and merges with word
       bases (e.g. Ġ + t -> Ġt) to form standard GPT-2 space-prefixed tokens. */
    if (t->byte_to_id[32] < 0) {
        int32_t gid = hm_get(t, "\xC4\xA0", 2); /* "Ġ" utf8 */
        if (gid >= 0) t->byte_to_id[32] = gid;
    }
    return 0;
}

static int load_merges(tokenizer_t *t, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    char *line = NULL; size_t cap = 0; ssize_t n;
    int mcap = 8192, mn = 0;
    t->merge_a = malloc(mcap*sizeof(int32_t));
    t->merge_b = malloc(mcap*sizeof(int32_t));
    t->merge_out = malloc(mcap*sizeof(int32_t));
    int first = 1;
    while ((n = getline(&line, &cap, f)) >= 0) {
        if (line[0] == '#') continue; /* skip "#version" header */
        (void)first;
        while (n>0 && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
        char *sp = strchr(line, ' '); if (!sp) continue;
        *sp='\0'; char *a=line, *b=sp+1;
        int ia = hm_get(t,a,strlen(a)), ib = hm_get(t,b,strlen(b));
        /* merged token = a ++ b */
        char cat[256]; size_t la=strlen(a), lb=strlen(b);
        if (la+lb >= sizeof(cat)) continue;
        memcpy(cat,a,la); memcpy(cat+la,b,lb); cat[la+lb]='\0';
        int io = hm_get(t,cat,la+lb);
        if (ia<0||ib<0||io<0) continue;
        if (mn==mcap){ mcap*=2; t->merge_a=realloc(t->merge_a,mcap*sizeof(int32_t)); t->merge_b=realloc(t->merge_b,mcap*sizeof(int32_t)); t->merge_out=realloc(t->merge_out,mcap*sizeof(int32_t)); }
        t->merge_a[mn]=ia; t->merge_b[mn]=ib; t->merge_out[mn]=io; mn++;
    }
    free(line); fclose(f);
    t->n_merges = mn;

    /* build rank map: (a<<14)|b -> merge index */
    size_t rc = 1; while (rc < (size_t)mn * 2 + 8) rc <<= 1;
    t->mr_cap = rc; t->mr_mask = rc - 1;
    t->mr_key = calloc(rc, sizeof(uint32_t));
    t->mr_val = calloc(rc, sizeof(int32_t));
    for (int m = 0; m < mn; m++) {
        uint32_t k = ((uint32_t)t->merge_a[m] << 14) | (uint32_t)t->merge_b[m];
        if (k == 0) k = 1; /* reserve 0 as empty marker */
        size_t idx = k & t->mr_mask;
        while (t->mr_key[idx]) idx = (idx + 1) & t->mr_mask;
        t->mr_key[idx] = k; t->mr_val[idx] = m;
    }
    return 0;
}

int tok_init(tokenizer_t *t, const char *vocab_json, const char *merges_txt) {
    memset(t, 0, sizeof(*t));
    build_byte_tables();
    if (load_vocab_json(t, vocab_json) != 0) return -1;
    if (load_merges(t, merges_txt) != 0) return -1;
    return 0;
}

void tok_free(tokenizer_t *t) {
    for (int i = 0; i < t->n_vocab; i++) { free(t->token_str[i]); free(t->token_bytes[i]); free(t->token_text[i]); }
    free(t->token_str); free(t->token_bytes); free(t->token_len);
    free(t->token_text); free(t->text_len);
    free(t->merge_a); free(t->merge_b); free(t->merge_out);
    free(t->hm_key); free(t->hm_val);
    for (int i = 0; i < t->wc_n; i++) { free(t->wc_str[i]); free(t->wc_toks[i]); }
    free(t->wc_str); free(t->wc_len); free(t->wc_toks); free(t->wc_ntok);
    memset(t, 0, sizeof(*t));
}

/* ── encoding: split into words, byte-initialize, apply merges ─────────── */
/* BPE a single word (raw bytes, optional leading space) into token ids. */
static int bpe_word(const tokenizer_t *t, const uint8_t *wb, int wlen,
                    int32_t *toks, int maxn) {
    int nt = 0;
    int start = 0;
    if (wlen > 0 && wb[0] == ' ') {
        int sp = 0; while (sp < wlen && wb[sp]==' ') sp++;
        int32_t gid = t->byte_to_id[32]; if (gid < 0) gid = 1;
        toks[nt++] = gid; start = sp;
    }
    for (int k = start; k < wlen && nt < maxn; k++) {
        int32_t id = t->byte_to_id[wb[k]]; if (id < 0) id = 1;
        toks[nt++] = id;
    }
    while (nt > 1) {
        int best = -1, bestrank = t->n_merges + 1;
        for (int k = 0; k + 1 < nt; k++) {
            if (toks[k] <= 4 || toks[k+1] <= 4) continue; /* skip specials/unk (ids 0-4) */
            uint32_t key = ((uint32_t)toks[k] << 14) | (uint32_t)toks[k+1];
            if (key == 0) key = 1;
            size_t idx = key & t->mr_mask;
            while (t->mr_key[idx]) {
                if (t->mr_key[idx] == key) { int r = t->mr_val[idx]; if (r < bestrank){bestrank=r;best=k;} break; }
                idx = (idx + 1) & t->mr_mask;
            }
        }
        if (best < 0) break;
        int outid = t->merge_out[bestrank];
        if (outid <= 4 || outid >= t->n_vocab) break;
        toks[best] = outid;
        memmove(toks+best+1, toks+best+2, (size_t)(nt-best-2)*sizeof(int32_t));
        nt--;
    }
    return nt;
}

/* cached word encode: returns pointer to token array + count, or NULL on miss. */
static int cache_get_or_build(const tokenizer_t *tc, const uint8_t *wb, int wlen,
                              const int32_t **out_toks, int *out_nt) {
    tokenizer_t *t = (tokenizer_t*)tc; /* cache is mutable bookkeeping */
    for (int i = 0; i < t->wc_n; i++)
        if (t->wc_len[i] == wlen && memcmp(t->wc_str[i], wb, (size_t)wlen) == 0) {
            *out_toks = t->wc_toks[i]; *out_nt = t->wc_ntok[i]; return 1;
        }
    int32_t tmp[512];
    int wl = wlen > 512 ? 512 : wlen;
    int nt = bpe_word(tc, wb, wl, tmp, 512);
    if (t->wc_n >= t->wc_cap) {
        int nc = t->wc_cap ? t->wc_cap*2 : 4096;
        t->wc_str = realloc(t->wc_str, (size_t)nc*sizeof(char*));
        t->wc_len = realloc(t->wc_len, (size_t)nc*sizeof(int32_t));
        t->wc_toks = realloc(t->wc_toks, (size_t)nc*sizeof(int32_t*));
        t->wc_ntok = realloc(t->wc_ntok, (size_t)nc*sizeof(int32_t));
        t->wc_cap = nc;
    }
    int slot = t->wc_n++;
    char *dup = malloc((size_t)wl + 1); memcpy(dup, wb, (size_t)wl); dup[wl]='\0';
    int32_t *arr = malloc((size_t)(nt>0?nt:1)*sizeof(int32_t)); memcpy(arr, tmp, (size_t)nt*sizeof(int32_t));
    t->wc_str[slot]=dup; t->wc_len[slot]=wl; t->wc_toks[slot]=arr; t->wc_ntok[slot]=nt;
    *out_toks = arr; *out_nt = nt; return 0;
}

int32_t tok_encode(const tokenizer_t *t, const char *s, int32_t n,
                   int32_t *out, int32_t cap) {
    uint8_t *allb; int alln;
    if (text_to_bytes(s, n, &allb, &alln) != 0) return -1;
    int o = 0, i = 0;
    while (i < alln) {
        int wstart = i;
        if (allb[i] == ' ') {
            while (i < alln && allb[i]==' ') i++;
            /* attach leading spaces to a following word; else emit them alone */
            if (!(i < alln && allb[i]!=' ' && allb[i]!='\t' && allb[i]!='\n')) {
                for (int k = wstart; k < i; k++) {
                    int32_t id = t->byte_to_id[allb[k]]; if (id<0) id=1;
                    if (o>=cap){free(allb);return -1;} out[o++]=id;
                }
                continue;
            }
        }
        while (i < alln && allb[i]!=' ' && allb[i]!='\t' && allb[i]!='\n') i++;
        int wlen = i - wstart; if (wlen > 512) wlen = 512;
        const int32_t *wtoks; int wnt;
        cache_get_or_build(t, allb + wstart, wlen, &wtoks, &wnt);
        for (int k = 0; k < wnt; k++) { if (o>=cap){free(allb);return -1;} out[o++]=wtoks[k]; }
        if (i == wstart) { /* lone whitespace/newline: emit its token, advance */
            int32_t id = t->byte_to_id[allb[i]]; if (id<0) id=1;
            if (o>=cap){free(allb);return -1;} out[o++]=id;
            i++;
        }
    }
    free(allb);
    return o;
}

int32_t tok_decode(const tokenizer_t *t, const int32_t *toks, int32_t n,
                   char *out, int32_t cap) {
    int o = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = toks[i];
        if (id < 0 || id >= t->n_vocab || !t->token_text[id]) return -1;
        int len = t->text_len[id];
        if (o + len > cap) return -1;
        memcpy(out+o, t->token_text[id], (size_t)len);
        o += len;
    }
    return o;
}
