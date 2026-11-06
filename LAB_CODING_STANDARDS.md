Synced from ResearchOps `lab-coding-standards.md` at 77fd022 on 2026-11-06.

# Lab Coding Standards

Canonical source. Lab repositories carry a synced copy as `LAB_CODING_STANDARDS.md`;
when this file changes, update those copies and bump their sync date.

## Context

This is a research and teaching lab studying human learning, attention, categorization
and expertise. Its code is written and reviewed by a rotating team of students, often new
to the protocol, the methods and the programming. These standards therefore depart from
industry practice where that serves teaching, transparency and long-term review. They
apply equally to students and AI coding agents; agents have additional rules (Principle 6).

---

## Pipeline

**Raw data → exclusions and re-coding → data tables at several levels of analysis (e.g.
raw, fixation, trial, participant) → analysis → publication-ready figures and
statistics.**

By the end of a project a single script, `<slug>_pipeline` (e.g. `StratCraft_pipeline.R`),
takes the raw data through every step and recreates every finding and figure in the
manuscript.

### Standard project structure

```
README.md                 project description
<slug>.md                 project slug file
<slug>_log.md             decision log (Principle 9)

/1_raw                    raw data; read-only, except that 00_fetch_raw adds files
/2_data_processing        cleaning and derived tables; reads only /1_raw
/3_analyses               analysis code; reads only tables in /2_data_processing
/4_output                 publication-ready figures, tables and statistics
/process                  process code (below); never run by the pipeline

<slug>_pipeline.R (or .sh) runs the full pipeline end to end
```

Scripts are numbered by pipeline stage (`01_clean_raw.R`, `02_extract_trial_level.py`,
`03_fit_rf_models.R`) so the directory listing shows the order.

No script writes to `/1_raw`, with one exception: `00_fetch_raw` (`.R` or `.py`), the
first stage of the pipeline, which downloads raw data from where it is published or
archived. Fetching lets the pipeline run from a clean clone without the raw data being
stored in git, and lets anyone confirm the files are the ones the analysis used. It:

- only adds files that are absent (a downloaded file, or the contents of an archive it
  downloaded), and never overwrites, modifies or deletes a file already in `/1_raw`;
- records each file's source, size and checksum in a manifest in `/2_data_processing`;
- stops and names the file when a download fails or does not match a published
  checksum, rather than continuing without it.

No other script, including later pipeline stages, writes to `/1_raw`.

### Product code and process code

**Product code** is the pipeline: every script `<slug>_pipeline` runs to produce the
manuscript's numbers and figures. It is what is reviewed, verified and maintained.

**Process code** is everything written along the way to check or explore: confirming a
script's input and output look right, checking ANOVA assumptions, looking for streakiness,
inspecting the cases an exclusion removes. It goes in `/process`, is committed so it can
be rerun, and is not part of review, verification or the pipeline. What it showed, when it
bears on a decision, goes in `<slug>_log.md` with its numbers and the script that produced
them. If a process check changes the analysis, the change enters product code through
Principle 11.

The model is a manuscript, which reports the resulting statistics, not the assumption
checks behind them. Product code is read by reviewers whose time is limited. It holds what
the analysis does and why, and nothing about how it came to be that way. That record is
in `<slug>_log.md` and the commit history.

---

## Principles

### 1. Clarity over elegance

Code should be as clear as possible, forgoing elegance for understandability by students
who join partway through.

### 2. Comments should teach

- **Methodology**: why a step is taken, with the relevant literature.
- **Math/statistics**: the underlying concept, or a source for it.
- **Programming**: the idea behind non-obvious code.

Comments in product code do not carry minor notes, remarks about procedure ("consistent
with Principle 5"), or history of who decided what and when. Those belong in
`<slug>_log.md`. The exception is the header's last-change field (Principle 3), kept
because shared code does not always travel with its git history.

### 3. Small, reviewable files

Code is split into small files, so that a reviewer can understand a file and re-verify it
quickly after a change. Every script begins with a header:

- **Purpose**: what it does in the pipeline
- **Inputs**: files or tables read, and their location
- **Outputs**: files or tables written
- **Depends on**: upstream scripts
- **Last substantive change**: date and one-line reason

### 4. Minimize external dependencies, but not at the cost of clarity

Prefer language basics to packages, unless a well-established package makes the code
substantially clearer to a student (e.g. `dplyr` verbs over nested base-R indexing).
Principle 1 takes precedence. Record and pin every package used (Principle 7).

### 5. Do not work around the data

Unexpected input is a possible data error to be reviewed, not something to be patched.
Product code never coerces, imputes, drops or reformats values quietly to keep running,
and never wraps steps in `try`/`except`/`tryCatch` that suppress a problem. A fix that is
warranted belongs in the exclusions and re-coding stage, explicitly, with a comment saying
why.

Checking that a script's input and output are what they should be is done when the script
is written, as process code. Product code does not carry validation blocks.

### 6. AI coding agents

Agents follow every rule above, and also:

- **Flag assumptions.** Where the protocol, a variable or an edge case is ambiguous, state
  the ambiguity and the interpretation chosen, in a comment or in `<slug>_log.md`.
- **Never add exception handling that suppresses or works around unexpected data**
  (Principle 5).
- **Never add tests, adjustments or analysis variants on their own initiative**
  (Principle 11).
- **Verify before committing.** Every new or changed product script is checked by an
  agent before it is committed: it meets these standards, and its input and output were
  inspected and are sensible. A committed script has passed that check; human review then
  confirms it behaves as intended.

### 7. Reproducibility

- Every stochastic step (splits, random forests, permutation tests, bootstrapping) sets
  and logs an explicit seed.
- `<slug>_pipeline` saves a record of the environment beside its outputs (`sessionInfo()`
  in R, `pip freeze` in Python).
- Package versions are pinned where practical (`renv` lockfile, `requirements.txt`).

### 8. Data dictionaries

Each table the pipeline produces has a data dictionary giving each column's name,
description, units and valid values, so the table can be understood without the code.

### 9. Decision log

`<slug>_log.md` at the project root holds one entry per substantive change: what changed,
why, and who made the call.

### 10. Claims about the data

A finding about the data (what a column means, why a pattern exists, whether a check
detects faults) is only as good as the test it has passed.

1. **Label every claim by status**: *measured* (a number from the data, with its source),
   *inferred* (an explanation of a measurement) or *assumed* (not yet tested). A log entry
   gives the label, the evidence, and what observation would show the claim false.
   Repetition does not promote a claim.
2. **Name an alternative before adopting a mechanism.** For a reported finding or its
   explanation, write down the one or two most plausible alternatives and check each
   against the data. If the data cannot separate them, the mechanism stays *inferred*.
3. **State the sample a finding came from** (years, regions, versions, participants), and
   check it on at least one slice outside that sample before applying it there.
4. **Foundational claims rest on raw data.** A claim other work builds on (a unit, a
   timebase, what a timestamp means) is checked against the raw tables, not a derived
   measure.
5. **Validate a check before it excludes data.** Inspect a sample of the cases it rejects
   and show they are faulty, not a subgroup that behaves differently. Until then it labels
   data; it does not exclude it.
6. **Name what was measured, not what it is believed to be**: "off-grid games", not
   "camera-stream fault". This applies to columns, variables, log entries and reports as
   well as figures (see Code output).
7. **Replicate before telling the team.** A finding goes beyond the people working on it
   only after a second, independent route to it: a different method, different data, or a
   reviewer (human, or a separate agent) asked to disprove it. Until then it is called
   provisional.

The checks these rules call for are process code.

### 11. Proportion: the simplest analysis that answers the question

Every added test, adjustment or variant costs people: code review, Methods and Results
text, and the effort reviewers and readers spend following the paper. The difference
between 92 and 95 of 100 participants doing x does not change the lesson that most do.

The default is the simplest analysis that answers each project question at the level of
detail it is asked. The possible answers are the categories in the `## Questions` section
of `<slug>.md`, written before the analysis and without saying which is expected (e.g.
"in no players / a few / most").

When a concern arises (an alternative explanation, a complicating factor), check the data
first: before adding AR(1) to control for streakiness, see whether there is streakiness
and how much. Do not run an analysis several ways to see what happens. Then:

1. **Stop.** Build nothing yet.
2. **Report** what the check showed, with its numbers and their source.
3. **Suggest** one adjustment, and only if the check showed the concern could move an
   answer to a different category.
4. **Implement** only on approval, as its own commit.

---

## Code output

Output carries what is needed to interpret it: parameter settings, what data is shown.
Axes are labelled informatively. Plots meant to be compared share axis limits. Markers
(colours, shapes, line types, fills) mean the same condition in every plot.

A process plot has no caption or text around it. It carries everything a reader needs on
its face: parameter values, settings, n, restrictions on the data. A product figure moves
most of this into its caption and the manuscript text, and keeps on the figure only what
the rules below require.

### Labels describe the data, not the conclusion

A label names what was measured; interpretation goes in the caption or text, where it can
be qualified and revised without regenerating the figure. This applies to panel titles,
axis labels, legend entries and series names.

- Panel titles state contents: "RT distribution by league", not "Skill is a pure time
  scaling".
- Axis and legend labels use the quantity and its units: "% of trials at or below 271 ms",
  not "fast component weight".
- A subset is described, not named for what it is believed to be: "RT > 271 ms", not
  "decision component".
- No evaluative words: "too narrow", "poor fit", "excellent agreement".
- A threshold or exclusion that defines a series appears in its label.

### Make the sampling visible

- Plot the computed points, not only the line joining them.
- Use minor tick marks on both axes wherever a value has to be read off, including log
  axes.
- State the grid of any curve computed on one (quantiles, bins, parameter sweeps).
- State n, and any restriction on the observations, on the figure or in the caption.

### APA format for manuscript figures and tables

- Figure text in Helvetica, 8–14 pt; table text in Times New Roman.
- Maximum width 6.5 inches (1-inch margins).
- At least 300 dpi.
- Tables use horizontal rules only: above and below the column headers and at the bottom.
  Figures have no external border.

---

## Checklist for a new product script

- Numbered filename matching its pipeline stage
- Header: purpose, inputs, outputs, depends on, last change
- Comments teach methodology, statistics and non-obvious programming
- No silent coercion, imputation or dropping; no exception handling around data
- Few packages, each justified and pinned
- Seeds set and logged for stochastic steps
- Data dictionary for each output table
- Ambiguous calls noted in `<slug>_log.md`
- Any excluding check validated against the cases it rejects (Principle 10.5)
- Nothing beyond the simplest analysis that answers the question, unless approved
  (Principle 11)
- Input and output checked as process code; agent-verified before commit (Principle 6)
