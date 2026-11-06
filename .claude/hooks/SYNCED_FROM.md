# Synced hooks — do not edit here

These are copies. The source of truth is `ResearchOps/hooks/`, and an edit made
here is lost the next time `ResearchOps/tools/sync_hooks.sh` runs. Fix the source,
then re-sync.

| | |
|---|---|
| Source | `ResearchOps/hooks/` |
| Source commit | `77fd022` |
| Synced | 2026-11-06 |

They are registered in `.claude/settings.json`, which is committed, so anyone
who clones this repository and runs Claude Code gets them (D8, option b).

Every hook in `ResearchOps/hooks/` enforces a lab standard, and all of them are
here.

Check for drift with:

```bash
ResearchOps/tools/sync_hooks.sh <this repo> --check
```
