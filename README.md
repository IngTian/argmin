# argmin

An archive of optimization algorithms written by hand: each one implemented, run and debugged to learn how it actually behaves.

## Algorithms

| Algorithm | Worked problems | Source |
|---|---|---|
| PDHG (Primal-Dual Hybrid Gradient) | [1-D signal denoising (TV)](pdhg/1d-signal-denoising/) | Chambolle & Pock (2011) |

## Layout

One folder per algorithm. Inside it, one subfolder per worked problem, each with its own `README.md` (the derivation) and `main.py` (the code).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
