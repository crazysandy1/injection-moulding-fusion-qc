# Contributing to FusionQC

Thanks for helping. Bug reports, docs, tests, experiments and code are all welcome.

## Ways to contribute

- **Report a bug or ask a question:** open an [issue](https://github.com/crazysandy1/injection-moulding-fusion-qc/issues) using the template.
- **Propose a feature:** open an issue first for anything large, so we can agree on the approach.
- **Good first areas:** tests for `training/splits.py` (no group leakage), dataset-loading checks, type hints, docs, notebook clean-up, and the roadmap items in the README.

## Development setup

```bash
git clone https://github.com/crazysandy1/injection-moulding-fusion-qc.git
cd injection-moulding-fusion-qc
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r fusion_system/requirements.txt pytest ruff
pytest -q tests                                      # runs on CPU, no dataset or network needed
```

The ProBayes dataset is not included. Code that needs it should be testable with synthetic data, as in `tests/test_model_smoke.py`.

## Pull request checklist

1. Fork, then create a branch: `git checkout -b feat/short-description`.
2. Keep the change focused. One topic per pull request.
3. Add or update tests. `pytest -q tests` must pass.
4. Run `ruff check tests --select E9,F63,F7,F82` and fix what it reports.
5. Update `README.md` and `CHANGELOG.md` (under *Unreleased*) when behaviour changes.
6. Use [Conventional Commits](https://www.conventionalcommits.org/) for messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
7. Fill in the pull-request template.

## Ground rules

- **Evaluation integrity:** never report random-split results as the headline metric. Group k-fold by `MET_ExperimentNumber` is the number to trust.
- **No secrets or data:** do not commit datasets, model weights, `.env` files, credentials or personal documents.
- **Config in one place:** hyperparameters and column names live in `fusion_system/config.py`.
- **Thermal images:** no colour augmentation, since pixel values are temperatures.

## Licensing of contributions

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE), and that you have the right to submit it. Add yourself to [AUTHORS.md](AUTHORS.md) in your first pull request if you wish.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
