#define _POSIX_C_SOURCE 200809L
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    sketch_t sk;
    const char *path = "build/test_sketch.db";
    unlink(path);
    
    /* 8M slots = 128MB for the test */
    size_t cap = 8388608;
    printf("[sketch] init %zu slots (%.1f MB)...\n", cap, (cap * 16.0) / 1e6);
    if (sk_init(&sk, path, cap) != 0) return 1;

    double t0 = now();
    size_t n_ins = 7000000;
    for (size_t i = 1; i <= n_ins; i++) {
        sketch_slot_t *s = sk_upsert(&sk, (uint32_t)i);
        if (!s) { printf("OOM at %zu\n", i); return 1; }
        s->logit = (float)i;
        s->count = 1;
    }
    double t_ins = now() - t0;

    t0 = now();
    size_t misses = 0;
    for (size_t i = 1; i <= n_ins; i++) {
        sketch_slot_t *s = sk_get(&sk, (uint32_t)i);
        if (!s || s->logit != (float)i) misses++;
    }
    double t_get = now() - t0;

    printf("INSERT %zu in %.3fs (%.2fM/s)\n", n_ins, t_ins, n_ins / t_ins / 1e6);
    printf("RETRIEVE %zu in %.3fs (%.2fM/s), misses=%zu\n", n_ins, t_get, n_ins / t_get / 1e6, misses);

    printf("[sketch] sweep...\n");
    t0 = now();
    sk_evict_sweep(&sk, 1, 1);
    printf("SWEEP in %.3fs\n", now() - t0);

    if (misses == 0) printf("GATE PASS\n");
    else            printf("GATE FAIL\n");

    sk_free(&sk);
    unlink(path);
    return misses == 0 ? 0 : 1;
}
