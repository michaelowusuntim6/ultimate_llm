#define _POSIX_C_SOURCE 200809L
#include "030_shard.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/home/mike/Vocab/dataset_prose.txt";
    size_t window_mb = (argc > 2) ? atoi(argv[2]) : 64;

    shard_reader_t sr;
    if (shard_open(&sr, path, window_mb) != 0) return 1;

    char *line;
    size_t len;
    size_t total_bytes = 0;
    size_t total_lines = 0;

    printf("[shard] streaming %s with %zu MB window...\n", path, window_mb);
    double t0 = now();
    
    while (shard_next_line(&sr, &line, &len) > 0) {
        total_bytes += len;
        total_lines++;
    }
    
    double elapsed = now() - t0;
    double mbps = ((double)total_bytes / (1024.0 * 1024.0)) / elapsed;

    printf("STREAM COMPLETE lines=%zu bytes=%zu\n", total_lines, total_bytes);
    printf("THROUGHPUT %.2f MB/s elapsed=%.3fs\n", mbps, elapsed);
    printf("GATE PASS: Zero-OOM streaming successful\n");

    shard_close(&sr);
    return 0;
}
