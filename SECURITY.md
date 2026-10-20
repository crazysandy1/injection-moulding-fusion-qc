# Security policy

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use [GitHub private vulnerability reporting](https://github.com/crazysandy1/injection-moulding-fusion-qc/security/advisories/new). Include what you found, how to reproduce it, and the impact. We aim to reply within 7 days.

## Scope notes

- This repository holds research code, not a hardened service. Do not expose a model-serving endpoint built from it to untrusted users without authentication and input validation.
- Never commit datasets, model weights, credentials or `.env` files. If you find one in the history, report it.
- Model checkpoints are loaded with PyTorch. Only load `.pt` files you trust.
