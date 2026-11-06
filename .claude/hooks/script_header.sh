#!/bin/bash
#
# script_header.sh — warn when an analysis script has no five-field header.
#
# Purpose   Lab standard, Principle 3: every script begins with a header stating
#           Purpose, Inputs, Outputs, Depends on, and Last substantive change.
#           The point is that a reviewer can orient in seconds. The header is
#           cheap to write while the script is being written and expensive to
#           reconstruct afterwards, which is why the warning belongs at the edit
#           rather than at review.
#
# Inputs    PostToolUse event JSON on stdin. Matcher: Write|Edit.
# Outputs   Nothing when the header is complete or the file is not a script.
#           Otherwise the missing fields on stderr and exit 2, which shows the
#           message to Claude without blocking anything — the write already ran.
#
# Depends on  hooks/hook_lib.sh, jq.
#
# Last substantive change  2026-10-21 — shell scripts are read too, when they
#           are pipeline scripts: directly inside a numbered stage folder, or
#           named <slug>_pipeline.sh. See ResearchOps_log.md.
#           Before that, 2026-10-13 — check the leading comment block (or the
#           module docstring) instead of a flat 40 lines from the top, so a
#           header whose last field sits below line 40 stops reporting as
#           missing; see friction.md 2026-10-07.
#           Before that, 2026-09-18 — read a Python module docstring as the
#           header, so docstring-headed scripts stop reporting as missing all
#           five fields; and forced LC_ALL=C on the header checks;
#           the default UTF-8 locale made BSD grep silently fail to match
#           "Last substantive change" whenever the header line held an em
#           dash. See WorkOps_log.md.
#
# ---------------------------------------------------------------------------
# Two deliberate limits
#
# It fires once per file per session. Warning on every edit to a headerless file
# would produce a dozen identical messages in one sitting, and the second one is
# already noise.
#
# It checks that a field is NAMED, not that it is filled in. "Inputs: " with
# nothing after it passes. Judging whether a header is honest is reading, and a
# hook does not read — it counts. A hook that tried would be wrong often enough
# to be ignored, which costs more than the case it would catch.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/hook_lib.sh"
hook_read_input

path=$(hook_field '.tool_input.file_path')
[ -z "$path" ] && exit 0
[ -f "$path" ] || exit 0

# The languages the lab's pipelines are written in. Anything else is out of
# scope rather than compliant. A shell script's #! line is a comment line, so
# its header is read as the leading comment block like R's.
case "$path" in
  *.R|*.r|*.Rmd|*.rmd|*.py|*.m|*.sh) ;;
  *) exit 0 ;;
esac

# A file that is not a pipeline script does not owe a pipeline header. Test
# files, __init__.py and the like are excluded by name rather than by guessing.
case "$(basename "$path")" in
  __init__.py|conftest.py|setup.py|test_*|*_test.py|*_test.R) exit 0 ;;
esac

# Shell is mostly used for utilities (install scripts, hooks, one-off helpers),
# which owe no pipeline header. A shell script is held to the header only when it
# is a pipeline script: directly inside a numbered stage folder
# (2_data_processing/22_sc2gears_timebase.sh) or the pipeline itself
# (<slug>_pipeline.sh).
case "$path" in
  *.sh)
    [[ "$path" =~ /[0-9]+_[^/]*/[^/]+\.sh$ || "$path" =~ _pipeline\.sh$ ]] || exit 0
    ;;
esac

hook_fired_once "script_header:$path" || exit 0

# The header lives at the top of the file, in a leading comment block or (in
# Python) a module docstring. Read a bounded window and then narrow it to the
# header itself, rather than checking a fixed number of lines from the top.
#
# Why not simply read more lines. The field patterns below are anchored to a
# comment marker, not to the header, so a comment anywhere in the scanned text
# can satisfy one; widening a flat window would start passing chatty scripts
# that have no header at all. Narrowing to the leading comment block keeps the
# anchor meaningful, because the block ends at the first line of code and
# nothing below the header is ever read. The window caps the work at 200 lines
# so this does not scan a 900-line script on every edit; the longest header in
# this repository is 49 lines (tools/project_brief.py, measured 2026-10-13).
window=$(head -200 "$path" 2>/dev/null) || exit 0

# The leading comment block: contiguous lines from the top that are blank or
# begin with a comment marker. A blank line continues the block, because some
# headers separate fields with an empty line rather than a bare "#".
leading_comment_block() {
  printf '%s' "$1" | LC_ALL=C awk '
    /^[ \t]*$/    { print; next }
    /^[ \t]*[#%]/ { print; next }
                   { exit }'
}

missing=()
# Each field accepts the variants actually in use across the lab's repos, so a
# correct header written in a slightly different hand is not reported as absent.
#
# LC_ALL=C on every check, on purpose. Under this machine's default UTF-8
# locale, BSD grep's `\b...\b.*\b...\b` intermittently fails to match even
# when both words are plain ASCII, as soon as the checked text contains a
# multibyte character anywhere (an em dash, for this lab's own header style,
# is nearly always present) — reproduced 2026-09-18 on tools/registry_sync.sh's
# own long-standing compliant header, which this hook had been silently
# misreporting as missing "Last substantive change" the whole time. The check
# is pure ASCII keyword matching and has no reason to be locale-aware at all.
has_field() { printf '%s' "$head" | LC_ALL=C grep -Eqi "$1"; }

# Python puts the header in a module docstring, which carries no comment marker,
# so the marker-anchored patterns below matched nothing and every docstring-headed
# script was reported as missing all five fields — all 24 Python scripts in
# BFLLatency, recorded there as D6. When a .py file opens with a module docstring,
# the docstring's own body becomes the text that is checked and the marker becomes
# optional. Scoping it to the docstring rather than dropping the marker outright
# matters: an unanchored search over 40 lines would match the word "purpose"
# anywhere in the code below the header and pass a file with no header at all.
marker='[#%]+'
doc=''
case "$path" in
  *.py)
    # Only a module docstring is a header: the search gives up at the first line
    # that is neither blank nor a comment, so a docstring further down the file
    # (a function's) is not read as one. That mattered once the window grew from
    # 40 lines to 200.
    doc=$(printf '%s' "$window" | LC_ALL=C awk -v d='"""' -v q="''" '
      { t = $0; sub(/^[ \t]*/, "", t); q3 = q "\047" }
      !inside && (substr(t,1,3) == d || substr(t,1,3) == q3) { inside = 1; next }
      !inside && t != "" && substr(t,1,1) != "#" { exit }
      inside  && (substr(t,1,3) == d || substr(t,1,3) == q3) { exit }
      inside { print }')
    ;;
esac

if [ -n "$doc" ]; then
  head=$doc
  marker='([#%]+)?'
else
  head=$(leading_comment_block "$window")
fi

has_field '^[[:space:]]*'"$marker"'.*\bpurpose\b' \
  || missing+=("Purpose — what this script does in the pipeline")
has_field '^[[:space:]]*'"$marker"'.*\binputs?\b' \
  || missing+=("Inputs — files it reads, and where they live")
has_field '^[[:space:]]*'"$marker"'.*\boutputs?\b' \
  || missing+=("Outputs — files it writes")
has_field '^[[:space:]]*'"$marker"'.*\bdepends\b' \
  || missing+=("Depends on — which upstream script must run first")
has_field '^[[:space:]]*'"$marker"'.*\blast\b.*\bchange\b' \
  || missing+=("Last substantive change — date and a one-line reason")

[ ${#missing[@]} -eq 0 ] && exit 0

{
  echo "$path is missing ${#missing[@]} of the 5 header fields (lab standards, Principle 3):"
  echo
  for m in "${missing[@]}"; do echo "  - $m"; done
  echo
  echo "Add them in a comment block at the top. Reported once per file per session."
} >&2
exit 2
