## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Type

- [ ] Analysis (new numbers over the published traces/features)
- [ ] Bug fix
- [ ] Docs
- [ ] Infrastructure (CI, packaging, tooling)

## Receipts (required if this adds or changes any number)

- Command to reproduce, from repo root:
- Data it ran on (dataset, N, split, label source):
- If the evaluation slice has constant/near-constant ground truth, the
  varying-truth number alongside it:

## Checklist

- [ ] `python campaign/reproduce_mini.py` still passes locally
- [ ] Nothing under `campaign/frozen/` or `campaign/PREREG_*.md` was modified
- [ ] Negative/null results reported as-is (no folding, no silent drops)
