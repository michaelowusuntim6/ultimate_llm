#ifndef SKETCH_H
#define SKETCH_H

#include <stdint.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Sketch Slot — exactly 16 bytes. Cache-line friendly (4 slots per 64B line).
 * key == 0 represents an EMPTY slot. Ensure signatures never hash to 0.
 * ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
    uint32_t key;       /* FNV-1a signature (0 = empty) */
    uint32_t class_id;  /* The most confident next-token for this signature */
    float    logit;     /* W[sig][class] score */
    uint16_t count;     /* occurrence count (capped) */
    uint16_t gen;       /* generation stamp for eviction */
    uint32_t topk_classes[8]; /* Top-8 predicted next-tokens for this signature */
} sketch_slot_t;

typedef struct {
    int         fd;
    sketch_slot_t *slots;
    size_t      capacity;     /* Must be a power of 2 */
    size_t      mask;         /* capacity - 1 */
    size_t      n_entries;
    uint16_t    current_gen;
    char        path[256];
} sketch_t;

int  sk_init(sketch_t *sk, const char *path, size_t capacity);
void sk_free(sketch_t *sk);
sketch_slot_t* sk_get(sketch_t *sk, uint32_t key);
sketch_slot_t* sk_upsert(sketch_t *sk, uint32_t key);
void sk_evict_sweep(sketch_t *sk, uint16_t count_threshold, uint16_t gen_window);

#endif
