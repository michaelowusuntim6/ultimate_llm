#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKENIZER_MAX_WORD_LEN 64

/* ── grow helpers ──────────────────────────────────────────────────────── */
static int trie_grow(tokenizer_t *t, int32_t need) {
    if (t->n_nodes + need <= t->cap_nodes) return 0;
    int32_t newcap = t->cap_nodes ? t->cap_nodes * 2 : 4096;
    while (newcap < t->n_nodes + need) newcap *= 2;
    trie_node_t *nn = realloc(t->nodes, (size_t)newcap * sizeof(*nn));
    if (!nn) return -1;
    t->nodes = nn;
    t->cap_nodes = newcap;
    return 0;
}

static int strings_reserve(tokenizer_t *t, int64_t need) {
    if (t->strings_len + need <= t->strings_cap) return 0;
    int64_t newcap = t->strings_cap ? t->strings_cap : (int64_t)(1 << 20);
    while (newcap < t->strings_len + need) newcap *= 2;
    char *nn = realloc(t->strings, (size_t)newcap);
    if (!nn) return -1;
    t->strings = nn;
    t->strings_cap = newcap;
    return 0;
}

static int offsets_reserve(tokenizer_t *t, int32_t need) {
    if (t->offsets_cap >= need) return 0;
    int32_t newcap = t->offsets_cap ? t->offsets_cap * 2 : 4096;
    while (newcap < need) newcap *= 2;
    uint32_t *nn = realloc(t->offsets, (size_t)newcap * sizeof(*nn));
    if (!nn) return -1;
    t->offsets = nn;
    t->offsets_cap = newcap;
    return 0;
}

/* ── trie primitives ───────────────────────────────────────────────────── */
static int32_t trie_new_node(tokenizer_t *t, uint8_t ch) {
    if (trie_grow(t, 1) != 0) return -1;
    int32_t i = t->n_nodes++;
    t->nodes[i].token_id = -1;
    t->nodes[i].first_child = -1;
    t->nodes[i].next_sibling = -1;
    t->nodes[i].ch = ch;
    t->nodes[i]._pad[0] = t->nodes[i]._pad[1] = t->nodes[i]._pad[2] = 0;
    return i;
}

static int32_t trie_add_child(tokenizer_t *t, int32_t parent, uint8_t ch) {
    for (int32_t e = t->nodes[parent].first_child; e >= 0; e = t->nodes[e].next_sibling) {
        if (t->nodes[e].ch == ch) return e;
    }
    int32_t n = trie_new_node(t, ch);
    if (n < 0) return -1;
    t->nodes[n].next_sibling = t->nodes[parent].first_child;
    t->nodes[parent].first_child = n;
    return n;
}

static int trie_insert(tokenizer_t *t, const char *word, int32_t len, int32_t token_id) {
    int32_t state = 0;
    for (int32_t i = 0; i < len; i++) {
        int32_t next = trie_add_child(t, state, (uint8_t)word[i]);
        if (next < 0) return -1;
        state = next;
    }
    if (t->nodes[state].token_id < 0) t->nodes[state].token_id = token_id;
    return 0;
}

/* ── vocab loader ──────────────────────────────────────────────────────── */
/* One token per line. Leading <tab>-delimited suffix is stripped if present. */
static int load_vocab_file(tokenizer_t *t, const char *path, int is_word_file) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }

    char   *line = NULL;
    size_t  cap  = 0;
    ssize_t n;
    int32_t count = 0;

    while ((n = getline(&line, &cap, f)) >= 0) {
        /* strip trailing CR/LF */
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) n--;
        /* strip after first tab (some vocab files have "word\tfreq") */
        for (ssize_t k = 0; k < n; k++) {
            if (line[k] == '\t') { n = k; break; }
        }
        if (n <= 0) continue;
        if (is_word_file && n > TOKENIZER_MAX_WORD_LEN) continue;

        if (strings_reserve(t, n) != 0)                    { free(line); fclose(f); return -1; }
        if (offsets_reserve(t, t->n_vocab + 2) != 0)       { free(line); fclose(f); return -1; }

        t->offsets[t->n_vocab] = (uint32_t)t->strings_len;
        memcpy(t->strings + t->strings_len, line, (size_t)n);
        t->strings_len += n;

        int32_t tid = t->n_vocab++;
        if (is_word_file && trie_insert(t, line, (int32_t)n, tid) != 0) {
            free(line); fclose(f); return -1;
        }
        count++;
    }
    free(line);
    fclose(f);
    return count;
}

/* ── public API ────────────────────────────────────────────────────────── */
int tok_init(tokenizer_t *t, const char *vocab_file, const char *fallback_file) {
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < 256; i++) t->root_child_idx[i] = -1;

    /* Root node */
    if (trie_new_node(t, 0) != 0) return -1;

    /* Word tokens */
    if (load_vocab_file(t, vocab_file, 1) < 0) return -1;
    t->n_word_tokens = t->n_vocab;

    /* Byte tokens — synthesized as raw bytes 0x00..0xFF */
    t->byte_base      = t->n_vocab;
    t->n_byte_tokens  = 256;
    for (int i = 0; i < 256; i++) {
        if (strings_reserve(t, 1) != 0)              return -1;
        if (offsets_reserve(t, t->n_vocab + 2) != 0) return -1;
        t->offsets[t->n_vocab] = (uint32_t)t->strings_len;
        t->strings[t->strings_len++] = (char)(uint8_t)i;
        t->n_vocab++;
    }

    /* Special tokens — synthesized with standard names */
    {
        static const char *specials[5] = {"<pad>", "<bos>", "<eos>", "<unk>", "<sep>"};
        t->n_special_tokens = 5;
        for (int i = 0; i < 5; i++) {
            int32_t len = (int32_t)strlen(specials[i]);
            if (strings_reserve(t, len) != 0)            return -1;
            if (offsets_reserve(t, t->n_vocab + 2) != 0) return -1;
            t->offsets[t->n_vocab] = (uint32_t)t->strings_len;
            memcpy(t->strings + t->strings_len, specials[i], (size_t)len);
            t->strings_len += len;
            t->n_vocab++;
        }
    }

    /* End sentinel for offsets */
    if (offsets_reserve(t, t->n_vocab + 1) != 0) return -1;
    t->offsets[t->n_vocab] = (uint32_t)t->strings_len;

    /* Precompute root-level first-char jump table */
    for (int32_t e = t->nodes[0].first_child; e >= 0; e = t->nodes[e].next_sibling) {
        t->root_child_idx[t->nodes[e].ch] = e;
    }

    /* fallback_file is intentionally ignored for now — byte tokens are
       synthesized as raw byte values, specials are hardcoded. If a future
       phase needs custom special names, parse fallback_file here. */
    (void)fallback_file;
    return 0;
}

void tok_free(tokenizer_t *t) {
    free(t->nodes);    t->nodes = NULL;
    free(t->strings);  t->strings = NULL;
    free(t->offsets);  t->offsets = NULL;
    t->n_nodes = t->n_vocab = 0;
    t->strings_len = 0;
}

int32_t tok_encode(const tokenizer_t *t, const char *s, int32_t n,
                   int32_t *out, int32_t cap) {
    const trie_node_t *nodes      = t->nodes;
    const int32_t     *root_idx   = t->root_child_idx;
    const int32_t      byte_base  = t->byte_base;
    int32_t i = 0, o = 0;

    while (i < n) {
        if (o >= cap) return -1;
        int32_t c0 = (uint8_t)s[i];
        /* Case-fold: vocab is lowercase, so fold input to match. This keeps
           proper nouns like "France" as a single word token instead of
           fragmenting into byte + subword. */
        if (c0 >= 'A' && c0 <= 'Z') c0 += 32;
        int32_t start = root_idx[c0];

        if (start < 0) {                       /* no word starts with this byte */
            out[o++] = byte_base + c0;
            i++;
            continue;
        }

        int32_t state    = start;
        int32_t last_tok = nodes[state].token_id;   /* could be -1 */
        int32_t last_pos = i + 1;

        int32_t pos = i + 1;
        while (pos < n) {
            int32_t c = (uint8_t)s[pos];
            if (c >= 'A' && c <= 'Z') c += 32;   /* case-fold continuation bytes */
            int32_t e = nodes[state].first_child;
            while (e >= 0 && nodes[e].ch != c) e = nodes[e].next_sibling;
            if (e < 0) break;
            state = e;
            if (nodes[state].token_id >= 0) {
                last_tok = nodes[state].token_id;
                last_pos = pos + 1;
            }
            pos++;
        }

        if (last_tok >= 0) { out[o++] = last_tok; i = last_pos; }
        else               { out[o++] = byte_base + c0; i++; }
    }
    return o;
}

int32_t tok_decode(const tokenizer_t *t, const int32_t *toks, int32_t n,
                   char *out, int32_t cap) {
    const int32_t byte_base = t->byte_base;
    const int32_t byte_end  = byte_base + 256;
    int32_t o = 0;

    for (int32_t i = 0; i < n; i++) {
        int32_t tok = toks[i];
        if (tok < 0 || tok >= t->n_vocab) return -1;

        if (tok >= byte_base && tok < byte_end) {
            if (o >= cap) return -1;
            out[o++] = (char)(uint8_t)(tok - byte_base);
            continue;
        }
        uint32_t start = t->offsets[tok];
        uint32_t end   = t->offsets[tok + 1];
        int32_t len    = (int32_t)(end - start);
        if (o + len > cap) return -1;
        memcpy(out + o, t->strings + start, (size_t)len);
        o += len;
    }
    return o;
}
