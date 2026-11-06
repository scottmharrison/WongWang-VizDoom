#!/bin/bash
#
# seed_check.sh — warn when a stochastic script sets no seed.
#
# Purpose   Lab standard, Principle 7: any stochastic step — train/test splits,
#           random forests, permutation tests, bootstrapping — must set and log
#           an explicit seed. Without one the result is not reproducible, and
#           the failure is silent: the script runs, produces a number, and
#           produces a different number next week. Nothing about the output says
#           which kind of number it is.
#
# Inputs    PostToolUse event JSON on stdin. Matcher: Write|Edit.
# Outputs   Nothing when a seed is set or no stochastic call is present.
#           Otherwise the calls found on stderr and exit 2, which shows the
#           message to Claude without blocking.
#
# Depends on  hooks/hook_lib.sh, jq.
#
# Last substantive change  2026-10-21 — read shell scripts; skip Python
#                          docstrings; an argument-less random.Random() or
#                          default_rng() is an unseeded draw, not a seed; a
#                          seed after the first draw, a None seed and calls on
#                          a passed-in generator are reported; a header Seed
#                          field declares a seed set elsewhere.
#                          See ResearchOps_log.md.
#
# ---------------------------------------------------------------------------
# What it can and cannot tell
#
# It reports a file that draws random numbers and
#   - sets no seed anywhere in it and declares none in its header,
#   - sets its first seed on a later line than its first draw, or
#   - passes None as a seed (seed=None, random_state=None), which asks the
#     library for an unseeded generator.
# The order test compares line numbers, not the order the code runs in: a draw
# inside a function defined above the seed is reported even when the function is
# only called after it. Moving the seed above the definitions satisfies it.
#
# It cannot see a seed set in a caller. StarPower2's pipeline sets one seed for
# the run; a stage script invoked from it may legitimately set none. Such a
# script declares it with a "Seed" field in its header (in a leading comment or
# a Python module docstring, e.g. "# Seed  set by StarPower2_pipeline.R"), and
# the check accepts the declaration; whether it is true is for review. A random
# generator passed in from another file is the same case: its calls
# (rng.choice(...)) count as draws, and the header declares where it is seeded.
# It fires once per file per session.
#
# Full-line comments are ignored, for both the stochastic calls and the seeds:
# a comment that says "the seed is applied before nsga2()" neither draws a
# random number nor sets a seed. A comment trailing code on the same line is
# still read as code, because telling where it starts means knowing whether a
# '#' sits inside a quoted string, and a pattern cannot tell that reliably.
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

# The comment marker for full-line comments: '%' in MATLAB, '#' in R, Python and
# shell.
# In an .Rmd, '#' lines outside code chunks are markdown headings, which are not
# code either, so blanking them is also correct.
case "$path" in
  *.m) COMMENT='^[[:space:]]*%' ;;
  *)   COMMENT='^[[:space:]]*#' ;;
esac

# The file with each full-line comment replaced by an empty line. Blanking
# rather than deleting keeps the line numbers the same as the file's, so the
# numbers in the message below point at the right lines. Every pattern below
# reads this, not the file itself.
code=$(sed -E "s/${COMMENT}.*//" "$path" 2>/dev/null)

# A Python docstring is prose, like a comment: "uses random.choice to pick a
# trial" in a docstring draws nothing. Python's own tokenizer finds each
# triple-quoted string, and its characters are replaced by spaces, so line
# numbers stay the same and any code before or after it on the line is still
# read. Pairing triple quotes by counting them would misread a string that holds
# a literal triple quote. A file the tokenizer cannot read (one half-written
# during an edit) is checked with its strings left in, which can only report
# more, never less.
PY_BLANK_DOCSTRINGS='
import io, sys, tokenize
text = sys.stdin.read()
lines = text.split("\n")
try:
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
except (tokenize.TokenError, SyntaxError, IndentationError):
    sys.stdout.write(text)
    sys.exit(0)
for tok in tokens:
    body = tok.string.lstrip("rRbBuUfF")
    if tok.type != tokenize.STRING or not body.startswith(("\"\"\"", "\x27\x27\x27")):
        continue
    (r1, c1), (r2, c2) = tok.start, tok.end
    for r in range(r1, r2 + 1):
        a = c1 if r == r1 else 0
        b = c2 if r == r2 else len(lines[r - 1])
        lines[r - 1] = lines[r - 1][:a] + " " * (b - a) + lines[r - 1][b:]
sys.stdout.write("\n".join(lines))
'
case "$path" in
  *.py) code=$(printf '%s\n' "$code" | python3 -c "$PY_BLANK_DOCSTRINGS") ;;
esac

# Calls that consume randomness. R, Python, MATLAB and shell together, because a
# hook that only knew one of them would pass the others silently. A random
# generator object built with no seed (random.Random(), default_rng()) counts as
# a draw. So does a method call on an object with one of the names generators
# are conventionally given (rng, gen, generator, prng, random_state, rs); a
# generator under any other name is seen only where it is built.
STOCHASTIC='\b(rng|gen|generator|prng|random_state|rs)\.(choice|choices|shuffle|sample|integers|random|normal|uniform|permutation|permuted|binomial|poisson|exponential|randint|rand|randn)\s*\(|random\.Random\s*\(\s*\)|default_rng\s*\(\s*\)|\$RANDOM|\bshuf\b|sort -R|--random-sort|\b(sample|rnorm|runif|rbinom|rpois|rexp|replicate|slice_sample|sample_n|randomForest|ranger|boot|nsga2|kmeans|train_test_split|shuffle|permutation|bootstrap)\s*\(|np\.random\.|numpy\.random\.|random\.(random|randint|choice|shuffle|sample)|\brand\s*\(|\brandn\s*\(|\brandi\s*\('

# Ways of fixing it, including the ones this lab actually uses (%dorng% for
# parallel loops in StarPower2, random_state= throughout scikit-learn). A
# generator constructor is a seed only when it is given an argument. In shell,
# assigning RANDOM seeds $RANDOM, and --random-source fixes shuf and sort -R.
SEEDED='set\.seed\s*\(|set_seed\s*\(|np\.random\.seed\s*\(|numpy\.random\.seed\s*\(|random\.seed\s*\(|default_rng\s*\(\s*[^) ]|random_state\s*=|random\.Random\s*\(\s*[^) ]|\bRANDOM=|--random-source|torch\.manual_seed\s*\(|%dorng%|\brng\s*\(|registerDoRNG'

# Passing None where a seed goes asks for an unseeded generator, so it is
# reported even though random_state= and default_rng(...) otherwise count as seeds.
NONE_SEED='\b(seed|random_state)\s*=\s*None\b|default_rng\s*\(\s*None\s*\)|random\.Random\s*\(\s*None\s*\)'

# A "Seed" field in the header: a comment line, or a line of a Python module
# docstring, that begins with the word Seed. Read from the file itself, since
# comments are blanked in $code, and only from the top 60 lines, where the header is.
SEED_FIELD='^[[:space:]]*([#%]+[[:space:]]*)?Seed\b'

first_line() { printf '%s\n' "$code" | grep -nE "$1" | head -1 | cut -d: -f1; }

first_draw=$(first_line "$STOCHASTIC")
[ -z "$first_draw" ] && exit 0
first_seed=$(first_line "$SEEDED")
first_none=$(first_line "$NONE_SEED")

if [ -n "$first_none" ]; then
  problem="passes None as a seed on line $first_none, which gives an unseeded generator"
elif [ -n "$first_seed" ]; then
  [ "$first_draw" -ge "$first_seed" ] && exit 0
  problem="draws on line $first_draw, before its first seed on line $first_seed"
elif head -60 "$path" | LC_ALL=C grep -Eq "$SEED_FIELD"; then
  exit 0
else
  problem="sets no seed and declares none in its header"
fi

hook_fired_once "seed_check:$path" || exit 0

found=$(printf '%s\n' "$code" | grep -nEo "$STOCHASTIC" | head -5)

{
  echo "$path draws random numbers and $problem (lab standards, Principle 7)."
  echo
  echo "Stochastic calls found (first 5, line-numbered):"
  printf '%s\n' "$found" | sed 's/^/  /'
  echo
  echo "Set an explicit seed before the first draw and log it — set.seed() in R,"
  echo "np.random.seed(), default_rng(<seed>) or random_state=<seed> in Python,"
  echo "rng() in MATLAB, RANDOM= in shell. If the seed is set by the pipeline"
  echo "script that calls this one, or a generator is passed in, add a Seed field"
  echo "to the header saying where it is set, rather than adding a second seed."
  echo
  echo "Reported once per file per session."
} >&2
exit 2
