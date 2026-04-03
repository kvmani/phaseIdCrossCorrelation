# Sphinx Site Source

This directory contains the canonical user-facing documentation source for the repository.

## Build

```powershell
python -m pip install -r .\docs\requirements.txt
python .\scripts\build_docs.py --clean
```

## Open

```powershell
python .\scripts\build_docs.py --open
```

## Output

The built HTML site is written to:

- `docs/_build/html/index.html`
