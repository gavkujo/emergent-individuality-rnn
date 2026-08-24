# Emergent Individuality in Neural Networks via Persistent Internal State Dynamics

> *Preprint: work in progress.*

Two recurrent networks, initialised identically, exposed to different sequences of experiences, develop measurably different internal dynamics. No explicit individuality objective. No different architectures. Just history.

This repo contains the code, experiments, results, and working notes for an ongoing investigation of that claim.

---

## Repo layout

```
src/
  model.py        : LeakyRNN with self-feedback (the proposed architecture)
  baselines.py    : VanillaRNN, GatedRNN (GRU), LSTM, FrozenESN
  train.py        : wake/sleep training loop, idle measurement, divergence metrics
  results_io.py   : save_result + load_latest helpers (week-organised)
  experiments/    : one folder per experiment, auto-discovered by run.py

run.py            : single entry point; dispatches to src/experiments/<name>
results/          : <name>_<YYYYMMDD>_<version>.json files in week<NN>_<YYYY>/
findings/         : working notes referencing results, in week<NN>_<YYYY>/
paper.pdf         : full draft paper
requirements.txt
```

## Running experiments

```bash
pip install -r requirements.txt

python run.py --list           # show available experiments
python run.py benchmark        # run the full B1-B6 suite
python run.py baselines        # run the architecture comparison
```

## Adding an experiment

Each experiment lives in its own folder at `src/experiments/<your_name>/`.
The folder must expose a package interface in `__init__.py` with:

```python
NAME        = "your_name"
DESCRIPTION = "one line"
DETAILS     = "what changed from previous versions"
VERSION     = "v1"

def run(device):
    ...
    return results_dict   # or None if nothing to save
```

`run.py` auto-discovers it. Results land in
`results/week<NN>_<YYYY>/your_name_<date>_v1.json`.

Any figures the experiment generates live in the same folder (e.g.
`src/experiments/<your_name>/figures.py`) and write PNGs to a sibling
`figures/` subdirectory or directly into the experiment folder. Do not
add a top-level cross-experiment figure script — figures are owned by
the experiment that produces the data they depend on.

## Conventions

- Every experiment writes a JSON via `src.results_io.save_result` with mandatory `name`, `description`, `details`, `time` (ISO-8601 second precision), and `results` keys.
- Results and findings are both organised by ISO week so they line up.
- `load_latest("benchmark")` returns the most recently dated result across all weeks — consumer scripts (per-experiment figures, downstream experiments) keep working when the week rolls over.

---

## Status

- [x] Stage 1: Baseline architecture (LeakyRNN, attractor characterisation)
- [x] Stage 2: Sleep/wake cycle (Hebbian consolidation, SR safety)
- [x] Stage 3: Experiential divergence (core novelty experiment)
- [x] Baseline comparison (Vanilla RNN, GRU, LSTM, ESN) — see findings/week22_2026/
- [ ] Stage 4: Neuromodulatory gates (DA/5HT/ACh/NE) — in progress
- [ ] Multi-architecture Stage 3 replication
- [ ] Adversarial decoder benchmark

## Citation

```
@misc{sachdev2026emergent,
  author = {Sachdev, Garv},
  title  = {Emergent Individuality in Neural Networks via 
             Persistent Internal State Dynamics},
  year   = {2026},
  note   = {Preprint. Work in progress.}
}
```
