# naampy

[![PyPI](https://img.shields.io/pypi/v/naampy)](https://pypi.org/project/naampy/)
[![CI](https://github.com/appeler/naampy/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/naampy/actions/workflows/ci.yml)
[![Docs](https://github.com/appeler/naampy/actions/workflows/docs.yml/badge.svg)](https://appeler.github.io/naampy/)
[![Python](https://img.shields.io/pypi/pyversions/naampy)](https://pypi.org/project/naampy/)

Naampy estimates population-level patterns in Indian first names. It returns a
calibrated score or an exact aggregate lookup, together with abstention status,
script support, and immutable artifact provenance.

Naampy does not observe or assign a person's gender. Do not use its output for
individual classification or consequential decisions.

## Install

```bash
pip install naampy
```

Naampy downloads its model and lookup artifacts from Hugging Face at revisions
pinned by the package. The wheel contains code, not model weights or source
tables.

## Choose the operation that matches the question

| Question | Function | Result |
| --- | --- | --- |
| What female source-label share is associated with this first-name pattern? | `estimate_first_name_pattern()` | Calibrated nullable score from 0 to 1 |
| What female and male source-label counts were released for this exact first name? | `lookup_first_name_composition()` | Exact nullable counts and shares |

The lookup never falls back to the model. The model never returns a hard gender
label.

## Estimate a name pattern

```python
from naampy import estimate_first_name_pattern

estimates = estimate_first_name_pattern(["Priya", "Rahul", "देव", None])
print(
    estimates[
        [
            "input_name",
            "female_label_score",
            "abstained",
            "abstention_reason",
            "script_supported",
            "model_revision",
            "model_bundle_sha256",
        ]
    ]
)
```

`female_label_score` estimates the female share among the female and male
electoral-roll source labels associated with a first-name pattern. It is not a
confidence score and does not measure a person's identity. An ineligible input
has a missing score and an explicit `abstention_reason`.

## Look up exact aggregate composition

```python
from naampy import lookup_first_name_composition

composition = lookup_first_name_composition(["Priya", "Rahul", "unknown"])
print(
    composition[
        [
            "input_name",
            "female_label_record_count",
            "male_label_record_count",
            "female_label_share_among_binary_labels",
            "lookup_status",
            "lookup_reason",
            "lookup_artifact_revision",
        ]
    ]
)
```

The public lookup contains one global row per released name. It combines states
and birth years, excludes the sparse third-gender source-label field, and keeps
only names with at least 1,000 represented female plus male label records. A
`not-released` result means only that the normalized name is absent from this
released table.

## Input contract

Both functions accept one string, a sequence containing strings or missing
values, or a pandas Series. Both return a new pandas DataFrame in input order.

Naampy applies Unicode NFC normalization, removes surrounding whitespace, and
case-folds the input. It does not delete punctuation, remove diacritics,
transliterate, reorder tokens, or truncate names. A name outside an artifact's
documented domain produces an abstention instead of a guess.

The learned model supports one Latin first name containing 3 to 19 ASCII letters
`a` through `z`, with no character repeated three times in a row. The exact
lookup supports the lowercase ASCII keys published in its manifest.

## Held-out evidence

On 12,445 test names representing 46,517,339 female plus male source-label
records, the calibrated v0.11 model has record-weighted expected binary log loss
0.3788, Brier score 0.1223, expected accuracy 0.8117, and expected female F1
0.8041. The 95% name-cluster bootstrap interval for log loss is 0.3486 to
0.4116. See the model card for name-weighted results, intervals, calibration
diagnostics, and the exact evidence hashes.

## Responsible use

The source labels come from historical Indian electoral-roll records. They are
binary because of the retained source target, not because identity is binary.
Coverage, spelling, romanization, geography, time, and administrative practices
can all change the score or lookup composition.

Use Naampy only for aggregate research where a name-pattern estimate is an
appropriate and validated measurement. Do not use it to classify a person or to
make decisions about employment, credit, housing, health care, policing,
immigration, voting, advertising, access to services, or any other consequential
outcome.

## Artifact controls

Set `NAAMPY_MODEL_DIR` to a directory containing the complete model bundle for
an offline or controlled deployment. Set `NAAMPY_LOOKUP_TABLE_DIR` to a directory
containing the complete lookup bundle. Naampy verifies the artifact schemas and
SHA-256 hashes before use.

The model and lookup manifests record their targets, reference populations,
label sources, supported inputs, versions, and content hashes. The package pins
the Hugging Face repository revision separately. Every model result also includes
`model_bundle_sha256`, a stable fingerprint of the validated weights,
architecture, target, and calibration contract. For a local override, the
repository and revision fields both report
`local-artifact-directory`; the fingerprint identifies the validated model bundle.
See the
[model card](https://github.com/appeler/naampy/blob/master/MODEL_CARD.md) and
[data contract](https://github.com/appeler/naampy/blob/master/model_training/DATA.md)
for the evidence and data contracts.

## Development

```bash
git clone https://github.com/appeler/naampy.git
cd naampy
uv sync --all-groups
make test
make lint
make docs
make build
```

Naampy is released under the MIT license. The data and artifact manifests state
their separate source terms and provenance.
