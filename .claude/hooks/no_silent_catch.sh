#!/bin/bash
#
# no_silent_catch.sh — flag error handling that hides unexpected data.
#
# Purpose   Lab standard, Principle 5: unexpected input is a potential data
#           error to be documented and reviewed, not something to be massaged
#           around. Principle 6 adds that AI agents fall into defensive error
#           handling more readily than students, because broad handling looks
#           production-ready. In this lab it is the opposite of safe: it converts
#           a loud, findable data problem into a silent, wrong number.
#
#           The plan calls this the hook that pays for itself. It is the direct
#           countermeasure to a reflex, and a reflex needs an interrupt.
#
# Inputs    PostToolUse event JSON on stdin. Matcher: Write|Edit.
# Outputs   Nothing when no unjustified construct is found. Otherwise the
#           line numbers and matches on stderr and exit 2, which shows the
#           message to Claude without blocking.
#
# Depends on  hooks/hook_lib.sh, jq.
#
# Last substantive change  2026-10-21 — shell scripts are read: `|| true`,
#           `|| :`, `2>/dev/null`, `&>/dev/null`, `>/dev/null 2>&1` and `set +e`. See
#           ResearchOps_log.md.
#           Before that, 2026-09-01 — first version. See WorkOps_log.md.
#
# ---------------------------------------------------------------------------
# The comment escape, and why it is the right shape
#
# Every construct below has legitimate uses. `na.rm = TRUE` is correct when the
# NAs are known and their meaning is established; `tryCatch` is correct around a
# network call. What is never correct is using one WITHOUT saying why, because
# the next reader cannot tell a considered decision from a silenced crash.
#
# So a construct is flagged unless a comment sits on the same line or the line
# directly above it. That makes the standard enforceable without forbidding
# anything: the cost of using one of these is one sentence explaining it, which
# is exactly the cost the standard intends.
#
# It fires once per file per session, and it counts occurrences rather than
# judging them — whether a given justification is a good one is reading.
# ---------------------------------------------------------------------------

source "$(dirname "${BASH_SOURCE[0]}")/hook_lib.sh"
hook_read_input

path=$(hook_field '.tool_input.file_path')
[ -z "$path" ] && exit 0
[ -f "$path" ] || exit 0

case "$path" in
  *.R|*.r|*.Rmd|*.rmd|*.py|*.m|*.sh) ;;
  *) exit 0 ;;
esac

# Constructs that suppress, coerce, or drop rather than fail. R, Python and
# MATLAB together, because a hook that knew only one of them would pass the
# other two silently.
#
# POSIX ERE only — no \s, no \b. This pattern is evaluated by awk, whose regex
# engine is POSIX and silently matches nothing when handed a Perl escape. That
# is exactly how the first version of this hook passed its own test suite while
# flagging nothing at all.
SUPPRESSING='tryCatch[[:space:]]*\(|[^A-Za-z0-9_.]try[[:space:]]*\(|^[[:space:]]*except[[:space:]]*:|^[[:space:]]*except[[:space:]]+[A-Za-z_.(]|suppressWarnings[[:space:]]*\(|suppressMessages[[:space:]]*\(|na\.rm[[:space:]]*=[[:space:]]*T|na\.omit[[:space:]]*\(|errors[[:space:]]*=[[:space:]]*.?coerce|\.fillna[[:space:]]*\(|\.dropna[[:space:]]*\(|on_bad_lines[[:space:]]*=|warn[[:space:]]*=[[:space:]]*F|options[[:space:]]*\([[:space:]]*warn[[:space:]]*=[[:space:]]*-1'

# Shell has its own ways of carrying on past a failure: `cmd || true` and
# `cmd || :` discard a non-zero exit, `2>/dev/null`, `&>/dev/null` and
# `>/dev/null 2>&1` discard the error message, and `set +e` turns off stopping
# on error for what follows.
# Only shell's patterns are applied to a shell script, and only R, Python and
# MATLAB's to the others: "try (" means nothing in shell, and "|| true" nothing
# in R.
#
# A shell comment starts with '#' at the start of a word. "$#" and "${x#y}" are
# not comments, so in shell a '#' counts as a comment only after a space or at
# the start of the line. The #! line is not a comment: it would otherwise justify
# anything in the script's first three lines. In R, Python and MATLAB any '#' or
# '%' on the line counts, as before.
case "$path" in
  *.sh)
    SUPPRESSING='[|][|][[:space:]]*(true|:)([[:space:];)]|$)|[^&>]2>[[:space:]]*/dev/null|&>[[:space:]]*/dev/null|>[[:space:]]*/dev/null[[:space:]]+2>&1|^[[:space:]]*set[[:space:]]+[+]e'
    COMMENT_ON_LINE='(^|[[:space:]])#'
    COMMENT_LINE='^[[:space:]]*#([^!]|$)'
    ;;
  *)
    COMMENT_ON_LINE='[#%]'
    COMMENT_LINE='^[[:space:]]*[#%]'
    ;;
esac

hook_fired_once "no_silent_catch:$path" || exit 0

# Walk the matches and keep only those with no comment on the line itself and no
# standalone comment in the three lines above. awk rather than grep because the
# decision needs the preceding lines, which grep does not carry.
#
# Three lines rather than one: in the try/except idiom the reason is written
# above the `try`, not above the `except`, and demanding it directly above the
# `except` would flag correctly-justified code.
#
# The pattern travels in the environment rather than through `awk -v`, because
# -v processes backslash escapes and would strip the \( and \. from it.
report=$(SUPPRESSING="$SUPPRESSING" COMMENT_ON_LINE="$COMMENT_ON_LINE" COMMENT_LINE="$COMMENT_LINE" awk '
  BEGIN { re = ENVIRON["SUPPRESSING"]; con = ENVIRON["COMMENT_ON_LINE"]; cl = ENVIRON["COMMENT_LINE"] }
  { line[NR] = $0 }
  END {
    n = 0
    for (i = 1; i <= NR; i++) {
      if (line[i] !~ re) continue
      if (line[i] ~ con) continue                         # commented on the line
      justified = 0
      for (k = i - 3; k < i; k++)
        if (k >= 1 && line[k] ~ cl) justified = 1
      if (justified) continue
      n++
      if (n <= 6) {
        s = line[i]
        sub(/^[[:space:]]+/, "", s)
        if (length(s) > 88) s = substr(s, 1, 85) "..."
        printf "  %d: %s\n", i, s
      }
    }
    if (n > 6) printf "  ... and %d more\n", n - 6
    if (n > 0) printf "COUNT %d\n", n
  }' "$path" 2>/dev/null)

printf '%s' "$report" | grep -q '^COUNT ' || exit 0

count=$(printf '%s' "$report" | sed -n 's/^COUNT //p')
lines=$(printf '%s' "$report" | grep -v '^COUNT ')

{
  echo "$path: $count construct(s) that suppress or massage unexpected data, with no comment saying why (lab standards, Principle 5)."
  echo
  printf '%s\n' "$lines"
  echo
  echo "Each of these can be right. What the standard forbids is using one without"
  echo "stating the reason, because a silenced crash and a considered decision then"
  echo "look identical to the next reader."
  echo
  echo "Either remove it and let the script fail on unexpected input, or add a"
  echo "comment on the line — or directly above it — saying what the case is and"
  echo "why handling it here is correct. A comment satisfies this check."
  echo
  echo "Reported once per file per session."
} >&2
exit 2
