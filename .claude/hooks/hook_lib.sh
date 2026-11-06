#!/bin/bash
#
# hook_lib.sh — helpers shared by the lab-standard hooks here and the WorkOps hooks. Sourced, never run.
#
# Purpose   Three things every hook here needs: read the event JSON off stdin,
#           pull a field out of it, and fire at most once per file per session.
#           Without the third, a hook that warns about a missing script header
#           warns again on every subsequent edit to the same file, and a warning
#           that always fires stops being read.
#
# Inputs    The hook event JSON on stdin, and $TMPDIR.
# Outputs   Nothing. Sets HOOK_JSON; defines hook_field, hook_fired_once, hook_die.
#
# Depends on  jq (present at /usr/bin/jq on macOS 15+; checked, not assumed).
#
# Last substantive change  2026-09-01 — first version. See WorkOps_log.md.
#
# ---------------------------------------------------------------------------
# The exit-code contract these hooks rely on, stated once here
#
#   PreToolUse   exit 2 blocks the tool call; stderr becomes the reason Claude sees.
#   PostToolUse  exit 2 does not block — the tool already ran — but stderr is shown
#                to Claude, which is what makes a warning land.
#   Stop         exit 2 prevents the turn ending; stderr tells Claude what to do.
#   Any other non-zero exit is a hook bug: non-blocking, reported in the transcript.
#
# Every hook here therefore exits 0 or 2 and nothing else, and a hook that cannot
# do its job (no jq, unreadable file) exits 0 rather than 1 — a broken guard must
# not become a broken session. That is the same departure from Principle 5 that
# session_start.sh documents, for the same reason.
# ---------------------------------------------------------------------------

# --- Read the event ---------------------------------------------------------
# stdin can only be read once, so every hook reads it here and queries the
# variable afterwards.
hook_read_input() {
  HOOK_JSON=$(cat)
  if ! command -v jq >/dev/null 2>&1; then
    # No jq means no hook. Say so on stderr, where exit 0 sends it to the debug
    # log rather than to Claude, and get out of the way.
    echo "workops hook: jq not found; hook skipped" >&2
    exit 0
  fi
}

# --- Pull one field ---------------------------------------------------------
# hook_field '.tool_input.file_path' -> the value, or "" if absent or null.
hook_field() {
  printf '%s' "$HOOK_JSON" | jq -r "$1 // \"\"" 2>/dev/null
}

# --- Fire at most once per session per key ----------------------------------
# Returns 0 (true) the first time it is called with a given key in a given
# session, and 1 (false) every time after. The marker lives under $TMPDIR, so it
# is cleared by the OS rather than needing cleanup, and is keyed on the session
# id so a new session starts fresh.
hook_fired_once() {
  local key="$1"
  local session dir marker
  session=$(hook_field '.session_id')
  [ -z "$session" ] && session="nosession"
  dir="${TMPDIR:-/tmp}/workops-hooks/${session//[^A-Za-z0-9_-]/_}"
  mkdir -p "$dir" 2>/dev/null || return 0   # cannot mark: warn every time rather
                                            # than never
  marker="$dir/$(printf '%s' "$key" | cksum | cut -d' ' -f1)"
  [ -e "$marker" ] && return 1
  : > "$marker"
  return 0
}

# --- The repository, not the working directory ------------------------------
# In a git worktree `--show-toplevel` is the worktree's own path, so its
# basename is the worktree name, not the project slug. That is how session_log
# came to look for `communication-style-adjustment-2c763b_log.md` and report a
# repository that has a decision log as having none.
#
# `--git-common-dir` points at the main checkout's `.git` from inside a worktree
# and at `.git` from a plain checkout, so its parent is the project root in both
# cases. Every fallback returns the old answer rather than failing: these are
# used to build a filename, and a wrong filename is a missed check, not a broken
# session.
hook_repo_root() {
  local top="$1" common root
  common=$(git -C "$top" rev-parse --git-common-dir 2>/dev/null) \
    || { printf '%s' "$top"; return; }
  case "$common" in
    /*) ;;                        # already absolute (the worktree case)
    *)  common="$top/$common" ;;  # relative to the toplevel (plain checkout)
  esac
  root=$(cd "$(dirname "$common")" 2>/dev/null && pwd) \
    || { printf '%s' "$top"; return; }
  printf '%s' "$root"
}

hook_repo_slug() {
  basename "$(hook_repo_root "$1")"
}

# --- Refuse, with a reason --------------------------------------------------
# The message goes to stderr because that is the only channel Claude reads.
hook_die() {
  printf '%s\n' "$@" >&2
  exit 2
}
