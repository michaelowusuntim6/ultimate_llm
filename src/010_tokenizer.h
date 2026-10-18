#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * FNV-1a 32-bit hashing (kept for trainer signatures).
 * ───────────────────────────────────────────────────────────────────────────── */
static inline uint32_t fnv1a_u32(const void *data, size_t n, uint32_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = seed;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x01000193u; }
    return h;
}
static inline uint32_t fnv1a_u32_combine(uint32_t h, uint32_t v) {
    h ^= v; h *= 0x01000193u; return h;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Byte-level BPE tokenizer (GPT-2 style: Ġ encodes a leading space).
 * Vocab + merges are loaded from /home/mike/Vocab/tokenizer_pure/.
 * Tokens are matched with standard BPE merge application over byte chunks.
 * UTF-8 text is mapped to bytes via the GPT-2 byte<->unicode table so every
 * codepoint round-trips losslessly.
 * ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
    char    **token_str;   /* id -> token string (as stored in vocab.json) */
    uint8_t **token_bytes; /* id -> byte-level representation (for merge matching) */
    int32_t  *token_len;   /* id -> number of bytes in byte-level repr */
    char    **token_text;  /* id -> real UTF-8 text (for decoding output) */
    int32_t  *text_len;    /* id -> byte length of real UTF-8 text */
    int32_t   byte_to_id[256]; /* raw byte value -> single-byte token id */
    int32_t   n_vocab;

    /* Merge rules: pair of token ids -> resulting token id */
    int32_t  *merge_a, *merge_b, *merge_out;
    int32_t   n_merges;

    /* Rank lookup: key = (a<<14)|b -> merge index (lower = applied first) */
    uint32_t *mr_key;
    int32_t  *mr_val;
    size_t    mr_cap, mr_mask;

    /* word-level encode cache: raw-bytes -> token ids */
    char    **wc_str;   int32_t *wc_len;
    int32_t **wc_toks;  int32_t *wc_ntok;
    int       wc_n, wc_cap;

    /* id lookup for a token string (hash map, open addressing) */
    uint32_t *hm_key;      /* FNV hash of token string + 1 (0 = empty) */
    int32_t  *hm_val;      /* token id */
    size_t    hm_cap, hm_mask;
} tokenizer_t;

int  tok_init(tokenizer_t *t, const char *vocab_json, const char *merges_txt);
void tok_free(tokenizer_t *t);

/* Encode utf8 s[0..n) into out[0..cap). Returns count or -1 on overflow. */
int32_t tok_encode(const tokenizer_t *t, const char *s, int32_t n,
                   int32_t *out, int32_t cap);

/* Decode toks[0..n) into utf8 out[0..cap). Returns byte count or -1. */
int32_t tok_decode(const tokenizer_t *t, const int32_t *toks, int32_t n,
                   char *out, int32_t cap);

#endif
