# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-08-02

### Fixed

- `in_rolls_fn_gender` no longer reuses a state- or year-filtered lookup table
  for a subsequent unfiltered query. The cache-invalidation key read `__state`
  and `__year`, which were never assigned, so a national query issued after a
  state query silently returned that state's counts.
- Rows with a missing, empty, or whitespace-only name are no longer passed to
  the model, which scored the literal string `"nan"` and returned a
  confident-looking gender for a row that had no name at all.
- Names with no `a-z` characters (Devanagari, Gujarati, Kannada, …) now return
  `None`/`NaN` instead of being labelled `"male"` at 0.5 confidence, which is
  what the previous neutral-probability default produced.
- Re-running `in_rolls_fn_gender` on its own output no longer raises. Existing
  naampy columns are dropped before the merge (previously they became
  `_x`/`_y` suffixes and the subsequent `prop_female` lookup raised `KeyError`),
  and the ML columns are assigned with `.loc` rather than the scalar-only `.at`.
- `in_rolls_fn_gender` no longer mutates the caller's DataFrame with a
  `__first_name` column.
- Interrupted or truncated downloads no longer leave a partial file that later
  runs report as a valid cache. `download_file` now streams to a temporary file
  and renames it into place only after verifying the transfer completed.
- Fixed a type-safety bug in `predict_fn_gender` where the batch-loop
  probability variable shadowed an earlier, differently-typed local variable
  in the same function scope, and a missing typed return in the char-BiLSTM
  model's `forward()`.

### Changed

- `pred_gender` and `pred_prob` are now always present on the returned frame
  (except for `v2_native`, which opts out of the model), holding `None`/`NaN`
  where unused. Previously they appeared only when some name missed the lookup,
  so callers had to probe with `row.get(...)`.
- The CLI validates `--state` after parsing, against the dataset named by
  `--dataset`. It was an argparse `choices=`, which downloaded and parsed the
  full v2_1k dataset just to build the parser — so even `--help` paid for it,
  and states were checked against v2_1k regardless of `--dataset`.
- The CLI now exits `1` rather than `-1` (which the shell reports as 255).
- Removed the dead `find_ngrams` helper, left over from the pre-LSTM n-gram
  model, and the stale TensorFlow mocks in the Sphinx config and the unused
  `tensorflow` pin in the Streamlit demo's requirements. Nothing in the package
  has imported TensorFlow since the PyTorch migration in 0.8.0.
- Corrected the documented state counts (v2/v2_1k carry 31 states and union
  territories, not 30) and the `maharastra` state key, which the docs and the
  example notebook spelled `maharashtra` — a value the data never contained.
- Dataverse-downloading tests are now opt-in via `NAAMPY_NETWORK_TESTS=1`; the
  default suite runs offline against fabricated fixtures.
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
