# Breathing Memory Development Guide

This guide collects contributor-oriented setup details that do not belong in the top-level README.

## Local Setup

Inside this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
python -m breathing_memory
```

`python -m breathing_memory` and `python -m breathing_memory serve` both start the server.

## Repository Layout

- `src/breathing_memory/`: package source
- `tests/`: unit tests
- `docs/spec.md`: implementation-facing specification
- `docs/design-rationale.md`: adopted design choices and their rationale
- `docs/user-guide.md`: user-facing operational guide between the README and the specification
