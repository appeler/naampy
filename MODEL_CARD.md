---
license: other
library_name: pytorch
tags:
  - names
  - india
  - pytorch
  - safetensors
---

# Naampy first-name pattern artifacts

Naampy estimates population-level patterns associated with Indian first names.
Version 0.11 provides two separate operations:

- a calibrated model score for the female share among female and male
  electoral-roll source labels associated with a first-name pattern; and
- an exact lookup of released aggregate female and male source-label counts.

Neither operation observes or assigns a person's gender. Do not use these
artifacts for individual classification or consequential decisions.

## Model output

The learned model returns `female_label_score`, a calibrated value from 0 to 1.
It estimates the female share among the female and male electoral-roll source
labels associated with an eligible first-name pattern. It is not a confidence
score, identity probability, or hard gender label.

The runtime also returns abstention status, the reason for an abstention, script
support, the target and reference population, the calibration population, and
immutable model provenance. `model_bundle_sha256` fingerprints the validated
weight hashes, architecture, target, and calibration contract, including for a
local artifact directory without a repository revision. Ineligible inputs
receive a missing score rather than a transformed or truncated guess.

## Architecture and artifact format

The model is an equal-probability ensemble of two character-level bidirectional
LSTMs trained with seeds 0 and 1. Each member uses:

- the lowercase ASCII characters `a` through `z`;
- a 64-dimensional character embedding;
- a two-layer bidirectional LSTM with hidden dimension 256; and
- dropout probability 0.2.

The two raw probabilities are averaged, then calibrated with a positive-slope
logit-affine transformation fitted on the calibration partition. Model weights
are stored as SafeTensors files. The manifest records and the runtime verifies
each filename, SHA-256 digest, architecture parameter, target, calibration
method, and model version. The package pins the Hugging Face repository revision
separately.

The frozen final-fit evidence hashes the model-definition source preserved at Git
commit `a048c6adc536ed927969acbc6a95d716f76bfd54`. A later release correction changes
only whether the default loader reports a local directory or the pinned repository;
it does not change the architecture, weights, ensemble, score, or calibration.

## Training data and partitions

The retained training artifact contains 124,447 unique eligible normalized
first names representing 465,179,772 female plus male electoral-roll label
records. The target is the female share among those two retained source labels.

Names, rather than source rows, were assigned to disjoint training, validation,
calibration, and test partitions in a 70/10/10/10 split. The assignment was
stratified by source-label share and represented-record support. Architecture,
training budget, seeds, and selected epochs were chosen with the training and
validation partitions. Final model fitting uses the combined training and
validation partitions. Calibration uses only the calibration partition. The
test partition remains sealed until the model and calibrator are frozen.

The private typed training artifact is not a public runtime dependency. Its
manifest records its Parquet schema, source and output hashes, partition
membership hashes, transformations, reference population, source revisions,
and privacy classification. See `model_training/DATA.md` in the source
distribution for the complete data contract.

## Held-out evaluation

The sealed test partition contains 12,445 names representing 46,517,339 female
plus male source-label records. The table reports the calibrated release model.
Intervals are 95% name-cluster percentile bootstrap intervals from 1,000 draws.

| Test metric | Name weighted | Record weighted | Record-weighted 95% interval |
| --- | ---: | ---: | ---: |
| Expected binary log loss | 0.4428 | 0.3788 | 0.3486 to 0.4116 |
| Expected binary Brier score | 0.1465 | 0.1223 | 0.1106 to 0.1354 |
| Expected record accuracy | 0.7732 | 0.8117 | 0.7855 to 0.8367 |
| Expected female precision | 0.7604 | 0.8051 | 0.7715 to 0.8377 |
| Expected female recall | 0.7677 | 0.8031 | 0.7444 to 0.8481 |
| Expected female F1 | 0.7640 | 0.8041 | 0.7676 to 0.8342 |
| 10-bin calibration error | 0.0171 | 0.0159 | 0.0132 to 0.0331 |

The calibration transform improved record-weighted test log loss from 0.3801
to 0.3788 and Brier score from 0.12239 to 0.12233. It did not improve every
metric: 10-bin calibration error increased from 0.0138 to 0.0159, and expected
record accuracy decreased from 0.8122 to 0.8117. The transform was selected on
the separate calibration partition by weighted log loss; test results did not
change the model or calibrator.

## Supported input

The model accepts one normalized first name containing 3 to 19 lowercase ASCII
letters, with no character repeated three or more times consecutively. The
runtime applies Unicode NFC normalization, trims surrounding whitespace, and
case-folds. It does not delete punctuation, remove diacritics, transliterate,
reorder tokens, or truncate names.

Inputs with missing values, multiple tokens, unsupported scripts or characters,
or values outside the frozen training domain produce explicit abstentions.

## Exact composition lookup

The repository also contains a separate public Parquet lookup artifact. It has
one globally aggregated row per released normalized first name and contains
female and male source-label record counts. It does not contain state or birth
year fields. The sparse third-gender source-label field is validated in the
source but is not aggregated, zero-imputed, or published.

The lookup covers 40,581 names and 475,541,789 represented female plus male
label records from Naampy v2_1k. Every released name has at least 1,000 included
source-label records. An absent lookup result means only that the name was not
released in this table. The lookup never falls back to the learned model.

## Intended use

Use Naampy for aggregate research only when a name-pattern estimate or released
aggregate composition is an appropriate, validated measurement. Report the
artifact version, target, abstention rate, and relevant performance slices.

Do not use Naampy to infer or assert an individual's gender. Do not use it for
employment, credit, housing, health care, policing, immigration, voting,
advertising, access to services, or any other consequential decision.

## Limitations

The source labels come from historical Indian electoral-roll records. They are
binary because of the retained modeling target, not because gender identity is
binary. Names may be shared across source-label categories. Spelling,
romanization, geography, time, coverage, and administrative practices may
change both model scores and lookup composition.

The learned model supports only a narrow lowercase ASCII first-name domain. It
does not support native-script names, accented Latin characters, punctuation,
or full names. Its score is a population-level association within the stated
reference data and should not be generalized to a different population without
direct validation and recalibration.

## Use with Naampy

```python
from naampy import estimate_first_name_pattern, lookup_first_name_composition

estimates = estimate_first_name_pattern(["Priya", "Rahul", "\u0926\u0947\u0935", None])
composition = lookup_first_name_composition(["Priya", "Rahul", "unknown"])
```

The package downloads every artifact at an immutable Hugging Face commit and
verifies the hashes and schemas in its manifests. For controlled or offline
deployments, set `NAAMPY_MODEL_DIR` and `NAAMPY_LOOKUP_TABLE_DIR` to directories
containing complete verified bundles.

## Licensing and citation

Naampy source code and learned weights are MIT licensed. The exact lookup is
derived from the source dataset and remains CC0 1.0. The source dataset is
published at DOI `10.7910/DVN/WZGJBM`. Cite the software version, the immutable
artifact revision, and the source dataset when reporting results.
