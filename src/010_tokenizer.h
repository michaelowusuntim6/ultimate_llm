#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * FNV-1a 32-bit hash.
 *
 * Used by the trainer's signature builder (Layer 1 of SLLM) and exported here
 * so that training and inference use byte-identical context hashing.
 *
 * FNV-1a:  h = (h ^ byte) * 0x01000193
 * The CONFIG.md seeds are used as the initial state h0. FNV permits any seed.
 * ───────────────────────────────────────────────────────────────────────────── */
static inline uint32_t fnv1a_u32(const void *data, size_t n, uint32_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = seed;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}

static inline uint32_t fnv1a_u32_combine(uint32_t h, uint32_t v) {
    h ^= v;
    h *= 0x01000193u;
    return h;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Trie node — flat array, index 0 = root, 16 bytes/slot.
 * Children form a singly-linked sibling list hanging off `first_child`.
 * The root's first-character hop is bypassed via `root_child_idx[256]`.
 * ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
    int32_t token_id;      /* -1 if not a terminal */
    int32_t first_child;   /* -1 if leaf */
    int32_t next_sibling;  /* -1 if last sibling */
    uint8_t ch;            /* char from parent -> this node (root: unused) */
    uint8_t _pad[3];
} trie_node_t;

typedef struct {
    /* Trie */
    trie_node_t *nodes;
    int32_t      n_nodes;
    int32_t      cap_nodes;
    int32_t      root_child_idx[256];   /* -1 if no child with this first byte */

    /* Strings buffer (token text) */
    char        *strings;
    int64_t      strings_len;
    int64_t      strings_cap;

    /* offsets[t] .. offsets[t+1] = byte range of token t's text.
       length = n_vocab + 1. */
    uint32_t    *offsets;
    int32_t      n_vocab;
    int32_t      offsets_cap;

    /* Vocabulary layout (stable at runtime)
     *   [0,              n_word_tokens)      — word tokens (trie-inserted)
     *   [byte_base,      byte_base + 256)    — byte tokens (raw byte value)
     *   [n_vocab - 5,    n_vocab)            — 5 specials
     */
    int32_t      n_word_tokens;
    int32_t      n_byte_tokens;      /* always 256 */
    int32_t      n_special_tokens;   /* always 5   */
    int32_t      byte_base;          /* == n_word_tokens */
} tokenizer_t;

/* Load vocab_file (one token per line). Byte + special tokens are synthesized. */
int  tok_init(tokenizer_t *t, const char *vocab_file, const char *fallback_file);
void tok_free(tokenizer_t *t);

/* Encode s[0..n) into out[0..cap). Returns token count or -1 on overflow. */
int32_t tok_encode(const tokenizer_t *t, const char *s, int32_t n,
                   int32_t *out, int32_t cap);

/* Decode toks[0..n) into out[0..cap). Returns byte count or -1 on overflow. */
int32_t tok_decode(const tokenizer_t *t, const int32_t *toks, int32_t n,
                   char *out, int32_t cap);

/* Byte fallback token for the given raw byte. */
static inline int32_t tok_byte_id(const tokenizer_t *t, uint8_t b) {
    return t->byte_base + (int32_t)b;
}

/* Special token IDs (the last 5 entries of vocab, in this order). */
static inline int32_t tok_pad(const tokenizer_t *t) { return t->n_vocab - 5; }
static inline int32_t tok_bos(const tokenizer_t *t) { return t->n_vocab - 4; }
static inline int32_t tok_eos(const tokenizer_t *t) { return t->n_vocab - 3; }
static inline int32_t tok_unk(const tokenizer_t *t) { return t->n_vocab - 2; }
static inline int32_t tok_sep(const tokenizer_t *t) { return t->n_vocab - 1; }

#endif
