#define _POSIX_C_SOURCE 200809L
#include "020_sketch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int sk_init(sketch_t *sk, const char *path, size_t capacity) {
    memset(sk, 0, sizeof(*sk));
    if (capacity & (capacity - 1)) {
        fprintf(stderr, "sketch capacity must be a power of 2\n");
        return -1;
    }
    sk->capacity = capacity;
    sk->mask = capacity - 1;
    sk->current_gen = 1;
    strncpy(sk->path, path, sizeof(sk->path) - 1);

    sk->fd = open(path, O_RDWR | O_CREAT, 0644);
    if (sk->fd < 0) { perror("open"); return -1; }

    size_t bytes = capacity * sizeof(sketch_slot_t);
    /* Pre-allocate physical blocks: sparse files (ftruncate) cause massive
       page-in thrashing under MAP_SHARED on this box. posix_fallocate is a
       cancellation point, so a large allocation can still be interrupted. */
    int falloc_res;
    do {
        falloc_res = posix_fallocate(sk->fd, 0, (off_t)bytes);
    } while (falloc_res == EINTR);
    if (falloc_res != 0) {
        fprintf(stderr, "posix_fallocate failed: %d\n", falloc_res);
        close(sk->fd); return -1;
    }

    sk->slots = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, sk->fd, 0);
    if (sk->slots == MAP_FAILED) { perror("mmap"); close(sk->fd); return -1; }
    
    /* If the file was newly created, memory is zeroed. We don't count existing entries yet. */
    return 0;
}

void sk_free(sketch_t *sk) {
    if (sk->slots && sk->slots != MAP_FAILED) {
        msync(sk->slots, sk->capacity * sizeof(sketch_slot_t), MS_SYNC);
        munmap(sk->slots, sk->capacity * sizeof(sketch_slot_t));
    }
    if (sk->fd >= 0) close(sk->fd);
    memset(sk, 0, sizeof(*sk));
}

static inline size_t hash_to_slot(uint32_t key, size_t mask) {
    /* fast integer hash for the table itself */
    key ^= key >> 16; key *= 0x7feb352dU; key ^= key >> 15; key *= 0x846ca68bU; key ^= key >> 16;
    return key & mask;
}

sketch_slot_t* sk_get(sketch_t *sk, uint32_t key) {
    if (key == 0) return NULL;
    size_t idx = hash_to_slot(key, sk->mask);
    for (size_t probes = 0; probes < sk->capacity; probes++) {
        sketch_slot_t *s = &sk->slots[idx];
        if (s->key == key) return s;
        if (s->key == 0) return NULL; /* empty slot ends the chain */
        idx = (idx + 1) & sk->mask;
    }
    return NULL;
}

sketch_slot_t* sk_upsert(sketch_t *sk, uint32_t key) {
    if (key == 0) key = 1; /* ensure 0 is reserved for empty */
    size_t idx = hash_to_slot(key, sk->mask);
    for (size_t probes = 0; probes < sk->capacity; probes++) {
        sketch_slot_t *s = &sk->slots[idx];
        if (s->key == key) {
            s->gen = sk->current_gen;
            return s;
        }
        if (s->key == 0) { /* insert here */
            s->key = key;
            s->gen = sk->current_gen;
            sk->n_entries++;
            return s;
        }
        idx = (idx + 1) & sk->mask;
    }
    return NULL; /* full */
}

void sk_evict_sweep(sketch_t *sk, uint16_t count_threshold, uint16_t gen_window) {
    size_t evicted = 0;
    uint16_t cutoff_gen = sk->current_gen > gen_window ? sk->current_gen - gen_window : 0;
    
    /* Naive full sweep. For 32M slots, this touches 512MB, taking ~0.2s on Ivy Bridge.
       Acceptable since it triggers rarely (when load factor > 0.90). */
    for (size_t i = 0; i < sk->capacity; i++) {
        sketch_slot_t *s = &sk->slots[i];
        if (s->key != 0 && s->count <= count_threshold && s->gen < cutoff_gen) {
            s->key = 0;
            s->logit = 0.0f;
            s->count = 0;
            s->gen = 0;
            evicted++;
            sk->n_entries--;
        }
    }
    sk->current_gen++;
    fprintf(stderr, "[sketch] sweep: evicted %zu, live %zu, gen -> %u\n", evicted, sk->n_entries, sk->current_gen);
}
