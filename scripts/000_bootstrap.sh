#!/bin/sh
# 000_bootstrap.sh — Phase A bootstrap (POSIX sh; no bashisms)
# Spec: ToDo_List.md PHASE A. Asserts toolchain + AVX1-only CPU, writes logs/000_bootstrap.log.

ROOT=/home/mike/sllm
LOG="$ROOT/logs/000_bootstrap.log"

mkdir -p "$ROOT/logs"
: > "$LOG"

say() {
    printf '%s\n' "$*" | tee -a "$LOG"
}

fail() {
    printf 'FATAL: %s\n' "$*" | tee -a "$LOG" >&2
    exit 1
}

say "=== SLLM bootstrap $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

# --- 1. Toolchain presence -------------------------------------------------
for tool in gcc python3 make; do
    command -v "$tool" >/dev/null 2>&1 || fail "missing required tool: $tool"
    say "found: $tool -> $(command -v $tool)"
done

# --- 2. mmap availability --------------------------------------------------
# mmap is a syscall exposed via <sys/mman.h>; compile+link a probe with the
# exact CONFIG.md flags to prove headers and libc support are present.
PROBE_C=$(mktemp /tmp/sllm_mmap_probe_XXXXXX.c)
cat > "$PROBE_C" <<'EOF'
#include <sys/mman.h>
#include <stdio.h>
int main(void) {
    void *p = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    printf("mmap ok\n");
    return 0;
}
EOF
PROBE_BIN=$(mktemp /tmp/sllm_mmap_probe_XXXXXX)
gcc -O3 -march=ivybridge -mtune=ivybridge -fno-math-errno -fno-trapping-math \
    -funroll-loops -flto -Wall -Wextra -Wno-unused-parameter -std=c99 \
    "$PROBE_C" -o "$PROBE_BIN" -lm -lpthread 2>>"$LOG" \
    || fail "mmap probe failed to compile with CONFIG.md flags"
"$PROBE_BIN" >>"$LOG" 2>&1 || fail "mmap runtime probe failed"
say "mmap: available (probe compiled+ran with CONFIG.md flags)"
rm -f "$PROBE_C" "$PROBE_BIN"

# --- 3. Compiler version ---------------------------------------------------
say "--- gcc --version ---"
gcc --version 2>&1 | tee -a "$LOG"
say "gcc -dumpversion: $(gcc -dumpversion)"

# --- 4. CPU feature flags --------------------------------------------------
CPUINFO=/proc/cpuinfo
[ -r "$CPUINFO" ] || fail "cannot read /proc/cpuinfo"

MODEL_NAME=$(grep -m1 '^model name' "$CPUINFO" | cut -d: -f2- | sed 's/^ //')
say "cpu model: $MODEL_NAME"

has_flag() {
    grep -m1 -E '^flags[[:space:]]*:' "$CPUINFO" | tr ' ' '\n' | grep -qx "$1"
}

for f in avx avx2 fma sse4_2; do
    if has_flag "$f"; then
        say "flag $f: PRESENT"
    else
        say "flag $f: ABSENT"
    fi
done

# --- 5. Hard assertions (do NOT touch arch flags to satisfy these) ---------
FAILS=""
has_flag avx   || FAILS="$FAILS avx-missing(expected-present)"
has_flag avx2  && FAILS="$FAILS avx2-present(expected-absent)"
has_flag fma   && FAILS="$FAILS fma-present(expected-absent)"

if [ -n "$FAILS" ]; then
    say "ASSERTION FAILED:$FAILS"
    say "CPU does not match CONFIG.md assumption (AVX1-only, Ivy Bridge)."
    say "STOP. Do not modify -march/-mtune to work around this."
    exit 1
fi

say "AVX1-only CPU confirmed"
say "=== bootstrap OK ==="
