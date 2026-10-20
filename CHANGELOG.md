# Changelog

All notable changes are recorded here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- MIT license, AUTHORS, CITATION, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY.
- Smoke tests for the fusion model and gradient reversal (`tests/`).
- GitHub Actions workflow, issue and pull-request templates.

## [0.1.0] - 2026-05-07

### Added
- Four-modality fusion model (thermal, visual, TCN sequence, tabular) with cross-modal transformer fusion and missing-modality masking.
- Domain-adversarial (DANN) training, focal loss, group k-fold, time-aware and random splits.
- Explainability, counterfactual recommender, drift monitor and consistency checker.
- Colab and Jupyter notebooks.
