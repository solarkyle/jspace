# jspace

Local j-space snapshots for fitted Jacobian lenses.

## Install

```bash
pip install -e .
pip install git+https://github.com/anthropics/jacobian-lens
```

For 4-bit loading, install a CUDA build of PyTorch plus bitsandbytes. Use
`quant=None` for bf16 loading.

## Three-line usage

```python
from jspace import Workspace
ws = Workspace("google/gemma-4-E4B-it", quant="4bit")
ws.snapshot("Which singer had a 1962 hit with Roses Are Red?").show()
```

## CLI

```bash
jspace info
jspace snap "Which singer had a 1962 hit with Roses Are Red?"
jspace chat --model google/gemma-4-E4B-it
```
