# Contributing to jspace

Thanks for your interest. jspace is open research on when LLM internal signals
actually detect errors — and most ways to contribute **require no GPU**: all
24.5k graded traces, feature tables, and fitted lenses are public on Hugging
Face, so analysis ideas run on a laptop.

## Ways to contribute

**No GPU needed:**
- **Analysis ideas** — new slices, controls, or baselines over the published
  traces and features
  ([solarkyle/jspace-hallucination-campaign](https://huggingface.co/datasets/solarkyle/jspace-hallucination-campaign)).
  Open an issue with the `analysis idea` template first so we can sanity-check
  the framing before you spend time.
- **Reproduction reports** — run `python campaign/reproduce_mini.py` (90
  seconds, CPU) and report PASS/FAIL with your environment. Failures are
  especially valuable.
- **Confound hunting** — the campaign's own most-cited result is a confound we
  found in our favorable numbers (the answer-readout confound). If you think a
  published number here has one we missed, that's the most useful issue you
  can open.
- **Docs** — anything unclear in the reports or preregistrations.

**GPU needed:**
- Extending the trace collection to other open models (the
  [16GB-consumer-GPU recipe](docs/GPU_NOTES.md) covers fitting lenses).

## Ground rules for claims

This repo's whole premise is that reliability claims need receipts. PRs that
add or change a numerical claim must:

1. Include the code that produces the number, runnable from the repo root.
2. State what data it ran on (dataset, N, split) and how labels were derived.
3. **Decompose before claiming**: if your evaluation slice has constant or
   near-constant ground truth, report the varying-truth number too — see the
   answer-readout confound in the README.
4. Report the honest number even when it's worse. Negative and null results
   are first-class here; the repo's headline result is a preregistered miss.

## Dev setup

```bash
git clone https://github.com/solarkyle/jspace && cd jspace
pip install -r requirements.txt
python campaign/reproduce_mini.py   # smoke test: should print OVERALL: PASS
```

CI runs the same reproduction on every PR (`.github/workflows/reproduce.yml`).

## Pull requests

- Small and focused beats large and sweeping.
- Don't modify anything under `campaign/frozen/` or the preregistration files
  (`campaign/PREREG_*.md`) — those are hashed, frozen artifacts; changing them
  breaks the point of the repo. If you believe one is wrong, open an issue.
- Analysis scripts go in `analysis/`, campaign infrastructure in `campaign/`,
  library code in `jspace/`.

## Questions

Open a [Discussion](https://github.com/solarkyle/jspace/discussions) or an
issue. Maintainer: [@solarkyle](https://github.com/solarkyle).
