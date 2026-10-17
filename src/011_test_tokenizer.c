#define _POSIX_C_SOURCE 200809L
#include "010_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *vocab_file    = (argc > 1) ? argv[1] : "/home/mike/Vocab/vocab_english.txt";
    const char *fallback_file = (argc > 2) ? argv[2] : "/home/mike/Vocab/byte_fallback_tokens.txt";
    const char *prose_file    = (argc > 3) ? argv[3] : "/home/mike/Vocab/dataset_prose.txt";
    int n_lines               = (argc > 4) ? atoi(argv[4]) : 10000;

    tokenizer_t tok;
    double t0 = now();
    if (tok_init(&tok, vocab_file, fallback_file) != 0) {
        fprintf(stderr, "tok_init failed\n");
        return 1;
    }
    double t_init = now() - t0;

    printf("INIT n_vocab=%d n_word=%d n_byte=%d n_special=%d "
           "strings=%.2f MB trie_nodes=%d init=%.3fs\n",
           tok.n_vocab, tok.n_word_tokens, tok.n_byte_tokens, tok.n_special_tokens,
           (double)tok.strings_len / 1e6, tok.n_nodes, t_init);

    FILE *f = fopen(prose_file, "r");
    if (!f) { perror(prose_file); tok_free(&tok); return 1; }

    char   *line    = NULL;
    size_t  cap     = 0;
    ssize_t len;
    int32_t toks[8192];
    char    decoded[8192];

    long long total_bytes = 0, total_tokens = 0;
    int n_processed = 0, n_bad = 0;

    double t1 = now();
    while ((len = getline(&line, &cap, f)) >= 0 && n_processed < n_lines) {
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) len--;
        if (len == 0) continue;
        if (len >= 8192) continue;

        int32_t nt = tok_encode(&tok, line, (int32_t)len, toks, 8192);
        if (nt < 0) { n_bad++; continue; }

        int32_t nb = tok_decode(&tok, toks, nt, decoded, 8192);
        if (nb != len || memcmp(line, decoded, (size_t)len) != 0) {
            if (n_bad < 5)
                fprintf(stderr, "MISMATCH line %d: in=%d out=%d\n",
                        n_processed, (int)len, nb);
            n_bad++;
            continue;
        }
        total_bytes  += len;
        total_tokens += nt;
        n_processed++;
    }
    double elapsed = now() - t1;

    free(line);
    fclose(f);
    tok_free(&tok);

    double ratio = total_tokens ? (double)total_bytes / (double)total_tokens : 0.0;
    double tps   = elapsed > 0 ? (double)total_tokens / elapsed : 0.0;
    double bps   = elapsed > 0 ? (double)total_bytes  / elapsed : 0.0;

    printf("ROUNDTRIP lines=%d bytes=%lld tokens=%lld ratio=%.2f bad=%d\n",
           n_processed, total_bytes, total_tokens, ratio, n_bad);
    printf("THROUGHPUT tok/s=%.2fM bytes/s=%.2fM elapsed=%.3fs\n",
           tps / 1e6, bps / 1e6, elapsed);

    if (n_bad == 0 && tps / 1e6 >= 5.0) {
        printf("GATE PASS: 100%% round-trip; %.2fM tok/s (>= 5M)\n", tps / 1e6);
        return 0;
    }
    printf("GATE FAIL: bad=%d tok/s=%.2fM\n", n_bad, tps / 1e6);
    return 1;
}
