# Findings

Working notes and analysis of experiment results. Updated manually as we go.

The folder layout mirrors `results/` — both organise content by ISO week so a
finding lines up with the result files it references.

## Layout

```
findings/
  week<NN>_<YYYY>/
    <topic>.md
```

Today's week folder is `week22_2026`.

## Conventions

Each `.md` should:

- State which results JSON it references (e.g. `results/week22_2026/benchmark_20260526_v1.json`)
- Quote the exact numbers it relies on (so the file stays useful even if results get reorganised later)
- Distinguish what the data shows from what we conclude
- Note any caveats, controls we wish we had, or follow-ups

These are working documents, not published findings. They feed into the paper draft.
