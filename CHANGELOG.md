# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed a type-safety bug in `predict_fn_gender` where the batch-loop
  probability variable shadowed an earlier, differently-typed local variable
  in the same function scope, and a missing typed return in the char-BiLSTM
  model's `forward()`.

### Changed

- Adopted [py-canon](https://github.com/gojiplus/py-canon) fleet conventions:
  reusable CI/docs/release workflows, `pyright` in place of `mypy`, expanded
  `ruff` rule set, dependency groups for docs tooling, and PEP 639 license
  metadata.

## [0.8.0]

### Changed

- Reworked native-name transliteration and moved the model-training pipeline
  to PyTorch (from Keras/TensorFlow).
- Migrated docs to Sphinx + Furo, packaging to `uv`/`uv_build`, and linting to
  `ruff`.

## [0.6.0] - 2023-02-19

### Added

- Type hints and docstrings across the public API.
- Cross-platform GitHub Actions test matrix (Ubuntu, macOS, Windows).

## [0.5.0] - 2022-01-27

### Added

- Native-language electoral-roll dataset support (`v2_native`, `v2_en`).
- Neural-network fallback model for names not found in the electoral rolls.

## Earlier releases

Versions prior to 0.5.0 predate this changelog; see the
[commit history](https://github.com/appeler/naampy/commits/master) for
details.

[Unreleased]: https://github.com/appeler/naampy/compare/v0.6.0...master
[0.6.0]: https://github.com/appeler/naampy/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/appeler/naampy/compare/v0.3.0...v0.5.0
