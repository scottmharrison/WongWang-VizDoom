#!/bin/bash
#
# raw_is_readonly.sh — block writes to raw data.
#
# Purpose   Lab standard: "Raw data is read-only. No script should ever overwrite
#           or modify a file in /1_raw." A prompt cannot guarantee that; a
#           PreToolUse hook can, because it runs before the write and can refuse it.
#           This is the one rule in the standards where a violation is not
#           recoverable from the repository — the original is gone.
#
# Inputs    PreToolUse event JSON on stdin, matcher Write|Edit|NotebookEdit|Bash;
#           and PostToolUse event JSON, matcher Bash, for the after-check below.
#           WORKOPS_REPO_ROOT (default ~/GitHub) bounds the after-check's scan.
# Outputs   Nothing on success. On a violation, the reason on stderr and exit 2,
#           which blocks the tool call (PreToolUse) or reports the change to
#           Claude after it happened (PostToolUse). PreToolUse Bash also leaves a
#           timestamp marker under $TMPDIR for the after-check.
#
# Depends on  hooks/hook_lib.sh, jq.
#
# Last substantive change  2026-10-01 — the after-check on Bash (MB, "Patch the
#           hole", SP2-RAWREADME). Before that, 2026-09-01 — first version. See
#           WorkOps_log.md.
#
# ---------------------------------------------------------------------------
# What this catches, and what it does not
#
# Exact, on Write / Edit / NotebookEdit: the target path is a single field, so a
# path with a `1_raw*` directory component is refused with no ambiguity.
#
# Best-effort, on Bash: shell text is not a path. Five write-shaped forms are
# recognised — a redirect into a raw path, `rm`, `sed -i`, `tee`, and `mv`/`cp`
# whose LAST argument is a raw path. Reading from a raw path is untouched, which
# is the whole point: `Rscript 01_clean.R < 1_raw/data.csv` must keep working.
#
# The text check cannot see a path assembled from a shell variable, a write
# performed by a script the command invokes, or an editor opened on the file.
# The after-check covers those for Bash (2026-10-01): PreToolUse leaves a
# timestamp marker, and PostToolUse lists every file or directory under a
# `1_raw*` directory that is newer than it, in each repository directly under
# WORKOPS_REPO_ROOT and in the command's own repository (a worktree's raw copy).
# A directory's mtime changes when an entry is added, removed or renamed, so a
# deletion shows as its directory. It reports after the fact and cannot undo;
# the 2026-09-16 StarPower2 extract is the case it exists for. A file changed by
# someone else during the command is reported too, which is the right default.
# `chmod -R a-w 1_raw/` would be a true guarantee, but it would also stop Mark
# adding raw files from Finder, so it is not applied.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/hook_lib.sh"
hook_read_input

# A directory component beginning `1_raw` — matches `1_raw` and `1_raw_data`,
# which are both in use (BFLLatency and StarPower2 respectively).
RAW_RE='(^|/)1_raw[^/]*/'

tool=$(hook_field '.tool_name')
event=$(hook_field '.hook_event_name')

# Marker for one Bash call, shared by its PreToolUse and PostToolUse events.
raw_marker() {
  local session id
  session=$(hook_field '.session_id'); [ -z "$session" ] && session="nosession"
  id=$(hook_field '.tool_use_id');     [ -z "$id" ] && id="noid"
  printf '%s/workops-hooks/%s/raw-%s' "${TMPDIR:-/tmp}" \
    "${session//[^A-Za-z0-9_-]/_}" "${id//[^A-Za-z0-9_-]/_}"
}

# --- after-check: what changed under raw while the command ran ---------------
if [ "$event" = "PostToolUse" ]; then
  [ "$tool" = "Bash" ] || exit 0
  marker=$(raw_marker)
  [ -e "$marker" ] || exit 0
  cwd=$(hook_field '.cwd')
  top=$(git -C "${cwd:-.}" rev-parse --show-toplevel 2>/dev/null)
  changed=$(
    { find "${WORKOPS_REPO_ROOT:-$HOME/GitHub}" -mindepth 2 -maxdepth 2 -type d -name '1_raw*' 2>/dev/null
      [ -n "$top" ] && find "$top" -mindepth 1 -maxdepth 1 -type d -name '1_raw*' 2>/dev/null
    } | sort -u | while IFS= read -r d; do
      find "$d" -newer "$marker" ! -name .DS_Store -print 2>/dev/null
    done)
  rm -f "$marker"
  if [ -n "$changed" ]; then
    hook_die \
      "Raw data changed while this command ran:" \
      "" \
      "$(printf '%s\n' "$changed" | head -20 | sed 's/^/  /')" \
      "" \
      "Raw data is read-only (lab standards, Pipeline section). The change has" \
      "already happened and this check cannot undo it. Do not build on it: say" \
      "so plainly in your reply, and record it in the project's decision log."
  fi
  exit 0
fi

case "$tool" in
  Write|Edit|NotebookEdit)
    path=$(hook_field '.tool_input.file_path')
    [ -z "$path" ] && path=$(hook_field '.tool_input.notebook_path')
    [ -z "$path" ] && exit 0
    if printf '%s' "$path" | grep -Eq "$RAW_RE"; then
      hook_die \
        "Refused: $path is under a raw-data directory." \
        "" \
        "Raw data is read-only (lab standards, Pipeline section). Cleaning," \
        "recoding and derived tables are written to 2_data_processing/ instead." \
        "" \
        "If this file genuinely belongs in raw data, add it outside a Claude" \
        "session so the decision is a deliberate act rather than a side effect."
    fi
    ;;

  Bash)
    cmd=$(hook_field '.tool_input.command')
    [ -z "$cmd" ] && exit 0

    # Each pattern below identifies a raw path in a *destination* position.
    # Anchoring on the destination is what keeps reads working.
    hit=""

    # 1. Redirection into a raw path:  > 1_raw/x   or  >> ../1_raw_data/x
    #    The leading directory part is optional, because `> 1_raw/x` has no
    #    slash before `1_raw`; the trailing slash is required, so that a file
    #    merely named `out_1_raw.csv` is not mistaken for a raw directory.
    printf '%s' "$cmd" | grep -Eq '>>?[[:space:]]*([^[:space:];&|]*/)?1_raw[^[:space:];&|]*/' \
      && hit="a redirect writing into it"

    # 2. rm, with or without flags
    [ -z "$hit" ] && printf '%s' "$cmd" \
      | grep -Eq '(^|[;&|(][[:space:]]*)rm([[:space:]]+-[^[:space:]]+)*[[:space:]]+[^;&|]*(^|/|[[:space:]])[^[:space:]]*1_raw' \
      && hit="an rm"

    # 3. sed -i, which edits in place
    [ -z "$hit" ] && printf '%s' "$cmd" \
      | grep -Eq 'sed[[:space:]]+([^;&|]*[[:space:]])?-i[^;&|]*1_raw' \
      && hit="a sed -i"

    # 4. tee, whose arguments are all destinations
    [ -z "$hit" ] && printf '%s' "$cmd" \
      | grep -Eq '\btee\b([[:space:]]+-[^[:space:]]+)*[[:space:]]+[^;&|]*1_raw' \
      && hit="a tee"

    # 5. mv or cp whose LAST argument is a raw path. Only the last argument is a
    #    destination, so `cp 1_raw/a.csv 2_data_processing/` must pass.
    if [ -z "$hit" ]; then
      last_arg_is_raw=$(printf '%s' "$cmd" | awk '
        BEGIN { RS = "[;&|]+"; found = 0 }
        {
          # Within one simple command, find mv/cp and take its final argument.
          n = split($0, tok, /[[:space:]]+/)
          verb = 0
          for (i = 1; i <= n; i++) {
            if (tok[i] == "mv" || tok[i] == "cp") verb = i
          }
          if (verb == 0) next
          for (j = n; j > verb; j--) {
            if (tok[j] == "") continue
            if (substr(tok[j], 1, 1) == "-") continue
            if (tok[j] ~ /(^|\/)1_raw/) found = 1
            break
          }
        }
        END { print found }')
      [ "$last_arg_is_raw" = "1" ] && hit="an mv or cp writing into it"
    fi

    if [ -n "$hit" ]; then
      hook_die \
        "Refused: this command contains $hit a raw-data path." \
        "" \
        "  $cmd" \
        "" \
        "Raw data is read-only (lab standards, Pipeline section). Derived files" \
        "belong in 2_data_processing/." \
        "" \
        "If the read is being misread as a write, say so — this check reads shell" \
        "text and can be wrong, and the rule it guards is worth the false positive."
    fi

    # Passed the text check: leave the marker the after-check compares against.
    marker=$(raw_marker)
    mkdir -p "$(dirname "$marker")" 2>/dev/null && : > "$marker"
    ;;
esac

exit 0
