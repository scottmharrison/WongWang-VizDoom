#!/bin/bash
#
# junk_guard.sh — refuse to stage regenerable junk.
#
# Purpose   Keep `.DS_Store`, `.RData`, `.Rhistory`, `Rplots.pdf`, Unity's
#           `Library/` and the like out of the repositories. All of these were
#           tracked somewhere in this lab on 2026-08-28; two repos were cleaned
#           on 2026-09-01. The cost is not disk — it is that forty junk files in
#           a diff bury the four source lines that matter, which is git rules §4.
#
#           A `git add -A` is where this happens, and it happens by accident:
#           the files are invisible in the editor and present in the tree.
#
# Inputs    PreToolUse event JSON on stdin. Matcher: Bash.
# Outputs   Nothing unless a staging command would sweep in junk. Otherwise the
#           offending paths on stderr and exit 2, which blocks the command.
#
# Depends on  hooks/hook_lib.sh, jq, git.
#
# Last substantive change  2026-09-21 — confine the path scan to the staging
#           command's own text, so a commit message naming a junk file is no
#           longer read as a list of paths. See WorkOps_log.md.
#
# ---------------------------------------------------------------------------
# Scope, and one thing it deliberately does not do
#
# It inspects `git add` and `git commit -a`. For a broad add it asks git what is
# actually there, rather than guessing from the command text; for an explicit
# `git add <paths>` it checks the paths given. It runs in the `cwd` the event
# reports, so a command that changes directory first is outside its view.
#
# It does NOT touch `.gitignore`. Restructuring what every clone ignores is a
# decision with reach beyond this machine, and git rules §4 says to flag it
# rather than act. This hook blocks the accident and names the file; adding a
# rule stays a deliberate act.
#
# Already-tracked junk is not its business either. A tracked `.DS_Store` keeps
# appearing in `git status` until it is untracked, and blocking every commit in
# such a repo would make the guard the problem. Only paths git reports as
# untracked or newly added are refused.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/hook_lib.sh"
hook_read_input

[ "$(hook_field '.tool_name')" = "Bash" ] || exit 0
cmd=$(hook_field '.tool_input.command')
[ -z "$cmd" ] && exit 0

# Only staging commands are of interest. `git commit -m ...` with nothing staged
# broadly is not: whatever is in the index was put there deliberately.
printf '%s' "$cmd" | grep -Eq '\bgit[[:space:]]+(add|commit)\b' || exit 0

cwd=$(hook_field '.cwd')
[ -n "$cwd" ] && [ -d "$cwd" ] && cd "$cwd" 2>/dev/null
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# One pattern per line, matched against the basename or a path component.
JUNK_RE='(^|/)(\.DS_Store|\.RData|\.RDataTmp[^/]*|\.Rhistory|Rplots\.pdf|Thumbs\.db|\.Rproj\.user|__pycache__|\.ipynb_checkpoints|Library)(/|$)'

offenders=""

# Look only at the staging command's own text, not at the whole Bash call.
#
# `grep -o` works a line at a time, and `[^;&|]*` stops at a shell separator, so
# each of these is the one command and nothing after it. Without that confinement
# the old version stripped everything up to `git add` and then read the entire
# rest of the call — heredoc bodies included — as a list of paths. It refused two
# commits on 2026-09-21 over the words "Library" and ".RData" appearing in a
# commit message, with nothing offending staged, and the second refusal also
# discarded a heredoc that was appending to a log file in the same command. A
# guard that blocks correct work teaches people to reach for --no-verify, which
# costs more than the accident it was written to catch.
add_seg=$(printf '%s\n' "$cmd" | grep -oE '\bgit[[:space:]]+add\b[^;&|]*' | head -1)
commit_seg=$(printf '%s\n' "$cmd" | grep -oE '\bgit[[:space:]]+commit\b[^;&|]*' | head -1)

# `-[a-zA-Z]*a` anchored to a space means a short flag bundle containing `a`.
# Anchoring matters: without it `--amend` matched, since its second dash is
# followed by "a".
if printf '%s' "$add_seg" | grep -Eq '\bgit[[:space:]]+add[[:space:]]+(-[A-Za-z-]+[[:space:]]+)*(-A|--all|\.|:/)([[:space:]]|$)' \
   || printf '%s' "$commit_seg" | grep -Eq '(^|[[:space:]])-[a-zA-Z]*a'; then
  # A broad add or `commit -a`. Ask git what would actually be swept in, rather
  # than trusting the command text. `--porcelain` marks untracked with `??` and
  # added with `A`; both are new to the index and therefore in scope.
  offenders=$(git status --porcelain 2>/dev/null \
    | grep -E '^(\?\?|A |AM| A)' \
    | sed 's/^...//' \
    | grep -E "$JUNK_RE")
else
  # An explicit `git add <paths>`. Check the paths given, ignoring flags and the
  # `--` that ends them. A quoted path containing a space is split here and each
  # half checked separately, which can only over-match, never under-match — the
  # safe direction for a guard.
  paths=$(printf '%s' "$add_seg" \
    | sed -E 's/^[[:space:]]*git[[:space:]]+add[[:space:]]*//' \
    | tr ' ' '\n' \
    | grep -v '^-' \
    | grep -v '^$')
  offenders=$(printf '%s\n' "$paths" | grep -E "$JUNK_RE")
fi

[ -z "$offenders" ] && exit 0

n=$(printf '%s\n' "$offenders" | grep -c .)
{
  echo "Refused: this would stage $n regenerable file(s) that do not belong in the repository."
  echo
  printf '%s\n' "$offenders" | head -20 | sed 's/^/  /'
  [ "$n" -gt 20 ] && echo "  ... and $((n - 20)) more"
  echo
  echo "These are rebuilt by the tools that made them. Tracking them costs"
  echo "reviewability: a diff full of them buries the source changes (git rules §4)."
  echo
  echo "Stage the paths you mean explicitly, or add a .gitignore rule — but say so"
  echo "first. What a repository ignores reaches every clone, so it is the repository owner's call,"
  echo "not one to make while getting a commit through."
} >&2
exit 2
