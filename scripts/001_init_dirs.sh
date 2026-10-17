#!/bin/sh
# 001_init_dirs.sh — Phase A directory init (POSIX sh)
# Spec: ToDo_List.md PHASE A. GATE: tree -L 1 matches spec.

ROOT=/home/mike/sllm
LOG="$ROOT/logs/001_init_dirs.log"

mkdir -p "$ROOT/logs"
: > "$LOG"

say() { printf '%s\n' "$*" | tee -a "$LOG"; }
fail() { printf 'FATAL: %s\n' "$*" | tee -a "$LOG" >&2; exit 1; }

cd "$ROOT" || fail "cannot cd $ROOT"

say "=== init dirs $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

# --- Writable roots (CONFIG.md paths + source/script dirs) -----------------
for d in build models logs bin src scripts; do
    mkdir -p "$d" || fail "cannot create $d"
    say "dir: $d"
done

# --- Symlink CONFIG.md -> ./CONFIG.md if missing ---------------------------
# ROOT is /home/mike/sllm, so ./CONFIG.md and $ROOT/CONFIG.md are the same
# inode; the symlink is a no-op self-reference when it already exists.
if [ ! -e "$ROOT/CONFIG.md" ]; then
    ln -s ./CONFIG.md "$ROOT/CONFIG.md" \
        && say "symlink: created ./CONFIG.md" \
        || fail "cannot symlink ./CONFIG.md"
else
    say "symlink: ./CONFIG.md already present"
fi

# --- Gate check -------------------------------------------------------------
if command -v tree >/dev/null 2>&1; then
    say "--- tree -L 1 ---"
    tree -L 1 "$ROOT" 2>&1 | tee -a "$LOG"
else
    say "--- ls -1 (tree unavailable) ---"
    ls -1 "$ROOT" | tee -a "$LOG"
fi

say "=== init dirs OK ==="
