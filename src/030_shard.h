#ifndef SHARD_H
#define SHARD_H

#include <stdio.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Shard Reader — Zero-OOM streaming text loader.
 * Uses a fixed-size buffer (window). Reads blocks, yields lines, and memmoves
 * any partial lines to the front of the buffer for the next chunk.
 * ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
    FILE   *f;
    char   *buf;
    size_t  cap;       /* Total buffer capacity (window_mb * 1024 * 1024) */
    size_t  len;       /* Valid bytes currently in buffer */
    size_t  pos;       /* Current read offset for yielding lines */
} shard_reader_t;

/* Opens file and allocates the fixed window buffer. */
int  shard_open(shard_reader_t *sr, const char *path, size_t window_mb);

/* Yields the next line. Modifies buf in place (replaces newline with \0).
   Returns length of line, 0 on EOF, -1 on error. */
int  shard_next_line(shard_reader_t *sr, char **line, size_t *line_len);

/* Frees buffer and closes file. */
void shard_close(shard_reader_t *sr);

#endif
