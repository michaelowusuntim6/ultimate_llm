#define _POSIX_C_SOURCE 200809L
#include "030_shard.h"
#include <stdlib.h>
#include <string.h>

int shard_open(shard_reader_t *sr, const char *path, size_t window_mb) {
    memset(sr, 0, sizeof(*sr));
    sr->f = fopen(path, "rb");
    if (!sr->f) {
        perror(path);
        return -1;
    }

    sr->cap = window_mb * 1024 * 1024;
    sr->buf = malloc(sr->cap);
    if (!sr->buf) {
        fprintf(stderr, "OOM: failed to allocate %zu MB for shard window\n", window_mb);
        fclose(sr->f);
        return -1;
    }

    /* Initial fill */
    sr->len = fread(sr->buf, 1, sr->cap, sr->f);
    return 0;
}

int shard_next_line(shard_reader_t *sr, char **line, size_t *line_len) {
    while (1) {
        /* If buffer is empty or fully consumed, refill */
        if (sr->pos >= sr->len) {
            if (feof(sr->f)) return 0; /* EOF */

            /* Move any leftover bytes (a partial line) to the front */
            if (sr->pos > 0 && sr->pos < sr->len) {
                size_t leftover = sr->len - sr->pos;
                memmove(sr->buf, sr->buf + sr->pos, leftover);
                sr->len = leftover;
            } else {
                sr->len = 0;
            }
            sr->pos = 0;

            /* Read next chunk into the remaining space */
            size_t space = sr->cap - sr->len;
            if (space == 0) {
                /* Buffer full, no newline found. Extremely long line. */
                fprintf(stderr, "shard_next_line: line exceeds window size\n");
                return -1;
            }

            size_t nread = fread(sr->buf + sr->len, 1, space, sr->f);
            sr->len += nread;
            
            if (sr->len == 0) return 0; /* EOF */
        }

        /* Search for newline in the current valid buffer */
        char *start = sr->buf + sr->pos;
        char *nl = memchr(start, '\n', sr->len - sr->pos);

        if (nl) {
            /* Found a line */
            *line = start;
            *line_len = nl - start;
            sr->pos += (*line_len) + 1; /* skip newline */
            
            /* Strip trailing CR */
            if (*line_len > 0 && (*line)[*line_len - 1] == '\r') {
                (*line_len)--;
            }
            (*line)[*line_len] = '\0'; /* Null-terminate in place */
            return (int)(*line_len);
        } else {
            /* No newline found. Refill buffer. */
            if (feof(sr->f)) {
                /* Yield the final partial line at EOF */
                *line = start;
                *line_len = sr->len - sr->pos;
                sr->pos = sr->len;
                if (*line_len > 0) {
                    (*line)[*line_len] = '\0';
                }
                return (int)(*line_len);
            }
            /* Trigger refill by setting pos to len */
            sr->pos = sr->len;
        }
    }
}

void shard_close(shard_reader_t *sr) {
    if (sr->f) fclose(sr->f);
    if (sr->buf) free(sr->buf);
    memset(sr, 0, sizeof(*sr));
}
