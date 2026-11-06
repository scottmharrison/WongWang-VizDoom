#!/bin/bash
#
# session_log.sh — at the end of a turn, ask for a log entry if work went
# unrecorded.
#
# Purpose   The failure the whole system exists to catch: work happens, the
#           decision log does not record it, and a month later nobody can say
#           why a choice was made. Principle 9 asks for one entry per
#           substantive change. The moment to write it is while the reasoning
#           is still in the session, not at review.
#
# Inputs    Stop event JSON on stdin.
# Outputs   Nothing unless commits were made today with no matching log entry.
#           Otherwise a prompt on stderr and exit 2, which prevents the turn
#           ending and gives Claude the message.
#
# Depends on  hooks/hook_lib.sh, jq, git.
#
# Last substantive change  2026-10-26 — one-line reply when the session made
#                          none of the commits.
#
# ---------------------------------------------------------------------------
# Why the trigger is commits-today rather than anything looser
#
# A Stop hook that fires on every turn is a Stop hook that gets disabled. The
# condition here is narrow and measurable: this repository has at least one
# commit dated today, and its decision log has no entry dated today. That is
# precisely the state Principle 9 is about, and it is false on the great
# majority of turns — a turn that only read files, or only edited without
# committing, triggers nothing.
#
# Loop protection is a marker file keyed on the session id, so the prompt can
# fire at most once per session. Without it, exit 2 would return control to
# Claude, which would stop again, and the hook would fire again. The marker is
# written BEFORE the exit, so even a hook that is killed mid-message cannot loop.
#
# Uncommitted work is deliberately not a trigger. Editing without committing is
# a normal mid-task state, and prompting for a decision-log entry about work
# that may yet be abandoned would train the prompt to be dismissed.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/hook_lib.sh"
hook_read_input

cwd=$(hook_field '.cwd')
[ -n "$cwd" ] && [ -d "$cwd" ] && cd "$cwd" 2>/dev/null

repo=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[ -z "$repo" ] && exit 0
slug=$(hook_repo_slug "$repo")   # the project name, not the worktree name
today=$(date +%Y-%m-%d)

# Commits made today, by anyone, on the checked-out branch.
commits=$(git -C "$repo" log --since="$today 00:00" --format='%h %s' 2>/dev/null)
[ -z "$commits" ] && exit 0

# Both accepted log filenames — the two standards documents disagree and neither
# has been made authoritative. See the note on the `L` flag in registry.md.
#
# Two directories, not one. Inside a worktree the log is present only if it was
# committed; an edit made in the main checkout and not yet committed is invisible
# from here. Checking the main root as well means a log that exists is found
# wherever it currently is, which is the question being asked.
main_root=$(hook_repo_root "$repo")
logfile=""
for d in "$repo" "$main_root"; do
  for f in "$d/${slug}_log.md" "$d/project_log.md" "$d/PROJECT_LOG.md"; do
    [ -f "$f" ] && { logfile="$f"; break 2; }
  done
done

if [ -n "$logfile" ] && grep -q "$today" "$logfile" 2>/dev/null; then
  exit 0   # today is already recorded
fi

# Mark before prompting, so this cannot fire twice however the turn ends.
hook_fired_once "session_log:$repo" || exit 0

n=$(printf '%s\n' "$commits" | grep -c .)
{
  if [ -n "$logfile" ]; then
    echo "$n commit(s) today in $slug, and $(basename "$logfile") has no entry dated $today."
  else
    echo "$n commit(s) today in $slug, and this repository has no decision log."
    echo "The two accepted names are ${slug}_log.md and project_log.md."
  fi
  echo
  printf '%s\n' "$commits" | head -10 | sed 's/^/  /'
  [ "$n" -gt 10 ] && echo "  ... and $((n - 10)) more"
  echo
  echo "If any of this was a substantive decision — a changed exclusion, a"
  echo "different analysis, a bug found late — append an entry now, while the"
  echo "reasoning is still in this session (Principle 9). Say what changed, why,"
  echo "and who made the call."
  echo
  echo "If it was all minor, say so and stop; that is a valid answer. Minor work"
  echo "is recorded by its commit messages, not the decision log."
  echo
  echo "The list counts every commit made today, including scheduled runs. If this"
  echo "session made none of them, reply with exactly \"Nothing committed.\" and"
  echo "nothing else."
  echo
  echo "This fires at most once per session."
} >&2
exit 2
