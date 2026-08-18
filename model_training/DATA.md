# Naampy model data contract

Naampy's neural model estimates the female share associated with a Latin-script
first-name pattern in aggregated Indian electoral-roll records. It does not
estimate an individual's gender identity. The v0.11 model target uses only
female and male source counts because the available third-gender count is too
small to train or evaluate a third class.

## Source tables

The figures below come from a full streaming profile of the locally retained
Dataverse transport files. A represented record is a count in an aggregated
cell. It is not necessarily a unique individual across every underlying roll.

| Table | State, year, name rows | States | Years | Distinct names | Represented records | Third-gender count | SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `naampy_v2.csv.gz` | 23,824,378 | 31 | 1887 to 2017 | 197,344 | 522,139,152 | 65 | `e47c917ba396aaa87f14db9e0fdc7dbbc83749987074802e93d2c228ec7ce9f8` |
| `naampy_v2_1k.csv.gz` | 10,226,249 | 31 | 1887 to 2017 | 40,581 | 475,541,837 | 48 | `2f72d8555ee6da837f94adb93fad6661a80fa141abc1eda7fa4e17f565fe4417` |
| `naampy_v2_native.csv.gz` | 3,719,116 | 16 | 1887 to 2017 | 110,267 | 410,965,158 | 0 | `899305b41330f944cc6e90a69e51147dd20bd680a23e9616f8a162841f45e6bd` |
| retained v3 model-development source | 6,670,064 | 31 | 1887 to 2017 | 125,635 | 466,713,646 | 65 | `a548226d9fe4c1dd7193d79487f51e55d77ad58e327b1b58e966b910e484ccf4` |

The v2, v2_1k, and native files are published under CC0 in Dataverse at
DOI `10.7910/DVN/WZGJBM`. The local
v3 construction retransliterates names from the native table with electoral
roll word maps, then combines those states with v2 states that did not require
retransliteration. The v3 file is currently gitignored and is not a published
runtime table. It contains 1,291,788 singleton state, birth-year, and first-name
cells, so the raw v3 table must not be published. The typed name-level export is
stored privately in `gojiberries/naampy-data` at immutable revision
`164a5e54e03254165a068cedd580ac2b42ae6bc5`; this is not a public data release.
The historical construction provenance pins Naampy commit
`2b15840cf0c63ddf6b5b81bf9ecf068d65d7722d` and `eroll_transliteration`
commit `262844fdaec6ee707a87160306e139e141a52bcd`. The hash in the table
identifies the legacy gzip bytes used to train and audit the shipped checkpoint.
Those bytes predate the canonical
writer below and are not the expected hash of a newly constructed artifact.

Build the full-coverage artifact from the repository root with both published
transport inputs:

```console
python model_training/retransliterate_native.py \
  --native-table /tmp/naampy_v2_native.csv.gz \
  --published-v2-table /tmp/naampy_v2.csv.gz \
  --output model_training/data/naampy_v3.csv.gz
```

The construction program is included in source distributions. It uses a stable
row order, fixed CSV formatting and line endings, and a gzip stream with no
embedded filename and timestamp zero, so identical input tables, word maps,
software, and code produce the same bytes and SHA-256 digest. The required word
maps are not yet published at immutable revisions, so this is a deterministic
construction procedure, not yet an independently reproducible data release or
a source of a published canonical v3 hash.

## Typed model-development export

Export the retained v3 input to one row per usable normalized first name with
typed label counts and a fixed development partition:

```console
python -m model_training.export_training_data \
  --data model_training/data/naampy_v3.csv.gz \
  --output /tmp/naampy_v3_training.parquet \
  --manifest /tmp/naampy_v3_training.json \
  --privacy-classification private \
  --publication-intent private_model_development
```

The Parquet schema is explicit: normalized name and partition are non-null
strings; female, male, and represented-record counts are non-null signed
64-bit integers. The exporter assigns the seed-zero 70/10/10/10 split to every
usable name before applying `--minimum-name-support`, so filtering cannot move
a retained name between partitions. Its manifest records source, artifact,
split-membership, and source-code hashes. Privacy classification and publication
intent are required declarations. Privacy classification is one of `private`,
`restricted`, or `public`; publication intent is either
`private_model_development` or `public_release_candidate`. The manifest stores
portable artifact filenames rather than machine-specific absolute paths.
Creating this aggregate artifact does not make the raw v3 cells suitable for
public release.

## Data dictionary

One source row represents an aggregated state, birth-year, and first-name
cell. The key is unique in all four profiled tables.

| Column | Type | Unit and universe | Missing values | Provenance and checks |
| --- | --- | --- | --- | --- |
| `state` | string | State or union-territory key for every aggregated cell | None observed | Electoral-roll processing pipeline |
| `birth_year` | whole year | Birth year attached to every aggregated cell | None observed | v2 transports store whole years as floating-point text; the runtime cache validates and casts them to `int16` |
| `first_name` | string | Processed first-name text for every cell | 111 blank rows in local v3; none in published tables | Native input contains script-specific text; v2 and v2_1k contain Latin letters only |
| `n_female` | nonnegative integer | Source records labeled female in the cell | None observed | Counts are nonnegative in every profiled table |
| `n_male` | nonnegative integer | Source records labeled male in the cell | None observed | Counts are nonnegative in every profiled table |
| `n_third_gender` | nonnegative integer | Source records labeled third gender in the cell | None observed | Only 65 represented records in v2 and local v3; zero in the native table |
| `prop_female` | proportion | `n_female / (n_female + n_male + n_third_gender)` | None observed | Recomputed values matched the source within floating-point precision |

No profiled table contains a zero-total cell, negative count, or duplicate key.
The local v3 table contains 20,336 rows whose first-name value is not purely
alphabetic, including the 111 blank values. Most other failures occur because
retransliteration produced spaces or multiple words.

## Recode ledger

The model-development exporter aggregates every source row by `first_name`,
sums the female and male counts, and computes the target as:

```text
female_proportion = female_count / (female_count + male_count)
```

It then applies these filters:

| Source value | Model value | Reason and consequence |
| --- | --- | --- |
| Names shorter than 3 or longer than 19 characters | Excluded | Matches the current model's training scope |
| Names containing nonalphabetic characters | Excluded | Removes spaces, punctuation, and malformed values |
| Names containing the same character three times in sequence | Excluded | Removes a documented noisy-name pattern |
| Characters outside Latin `a` to `z` | Excluded | The exporter never creates a valid name by dropping characters |
| Female and male label-record counts | `represented_binary_label_record_count` | Used as the training and record-weighted evaluation weight |
| Female share among female and male counts | Soft binary target | The model does not use the third-gender count |

These steps produce 124,447 usable unique Latin-script names from the local v3
construction. The exporter requires lowercase ASCII before encoding, so it never
obtains a valid name by silently dropping characters. Training reads the typed
Parquet produced by this exporter.

## Split and evaluation contract

The typed v3 artifact assigns each usable name to one fixed partition before
any support filtering. The seed-zero split is disjoint and exhaustive.

| Partition | Names | Represented binary-label records | Membership SHA-256 |
| --- | ---: | ---: | --- |
| Training | 87,113 | 325,628,001 | `95c1707c543283a69e2076fd6d85dae56bfc6cea8f671ab7100dd336efa2c27c` |
| Validation | 12,445 | 46,518,213 | `6e3c9c7439b0cad8dfe248b9cf12f9277aa7a467f46bf7bf1f6b8136e02dac81` |
| Calibration | 12,444 | 46,516,219 | `458cb4b32d9d3ba0158301b5b69f48f675665ec017b80f782901b587e10154f8` |
| Test | 12,445 | 46,517,339 | `59b957a3ed613368ed01c97470d8fdd204b760ebe7f2c884aeff9b2abaa92c06` |

Model and architecture selection use only the training and validation
partitions. Final ensemble constituents fit on training plus validation for
precommitted epoch counts. Calibration fits one positive-slope affine transform
to the ensemble logit on the calibration partition. The test command refuses to
load test rows until the data, ensemble, calibration, and runtime manifests pass
SHA-256 and schema checks.

The release workflow exposes four commands: `development`, `fit-final`,
`calibrate`, and `score-test`. Each command loads only its permitted partitions
and writes new files without replacing frozen artifacts. The two final model
constituents use SafeTensors. The runtime verifies both checkpoints, averages
their raw probabilities, and then applies the fitted calibration transform.

Thresholded precision, recall, and F1 use fractional female shares to form
expected source-label confusion counts across represented registration records.
Majority-name accuracy remains a separate metric. Aggregate-composition mean
squared error compares the score with the observed name-level share. Expected
binary Brier score also includes the irreducible Bernoulli variance
`female_share * (1 - female_share)`.

The final test report gives point estimates and 95 percent name-cluster
percentile bootstrap intervals. Bootstrap randomness has its own seed and does
not change model fitting, calibration, partition membership, or point estimates.

## Public lookup contract

The public lookup is a separate typed artifact derived from the published
v2_1k table. It aggregates all states and birth years to one row per normalized
name, then retains names with at least 1,000 female plus male source-label
records. The artifact contains 40,581 names and 475,541,789 represented binary
label records.

The lookup does not publish state, birth year, or third-gender source-label
counts. It never calls the learned model when a name is absent. The reason
`not-released` means only that a normalized name is absent from the released
global table.

## Artifact publication

The row-level v3 table remains private because its state, birth-year, and name
cells can have very low support. The typed name-level training artifact is
stored in private Hugging Face dataset repository `gojiberries/naampy-data` at
revision `164a5e54e03254165a068cedd580ac2b42ae6bc5`. Public model and global
lookup artifacts belong in `gojiberries/naampy` at immutable commit revisions.
Runtime outputs report those revisions and the artifact hashes.

## Open provenance questions

- The published source documentation does not establish whether the same person
  can appear across roll editions. Counts are represented registration records,
  not confirmed unique people.
- Birth years before plausible modern electoral cohorts need a source-level
  explanation or an explicit support rule.
- The native string `[লে` and similar malformed values need review against the
  extraction source before any native-script model uses them.
- The v3 transliteration word maps and construction environment need immutable
  releases before another team can reproduce v3 from the published inputs.
