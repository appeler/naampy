# Naampy model data contract

Naampy's neural model estimates the female share associated with a Latin-script
first-name pattern in aggregated Indian electoral-roll records. It does not
estimate a person's gender identity. The current checkpoint uses only female
and male source counts because the available third-gender count is too small
to train or evaluate a third class.

## Source tables

The figures below come from a full streaming profile of the locally retained
Dataverse transport files. A represented record is a count in an aggregated
cell. It is not necessarily a unique person across every underlying roll.

| Table | State, year, name rows | States | Years | Distinct names | Represented records | Third-gender count | SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `naampy_v2.csv.gz` | 23,824,378 | 31 | 1887 to 2017 | 197,344 | 522,139,152 | 65 | `e47c917ba396aaa87f14db9e0fdc7dbbc83749987074802e93d2c228ec7ce9f8` |
| `naampy_v2_1k.csv.gz` | 10,226,249 | 31 | 1887 to 2017 | 40,581 | 475,541,837 | 48 | `2f72d8555ee6da837f94adb93fad6661a80fa141abc1eda7fa4e17f565fe4417` |
| `naampy_v2_native.csv.gz` | 3,719,116 | 16 | 1887 to 2017 | 110,267 | 410,965,158 | 0 | `899305b41330f944cc6e90a69e51147dd20bd680a23e9616f8a162841f45e6bd` |
| legacy local v3 checkpoint input | 6,670,064 | 31 | 1887 to 2017 | 125,636 | 466,713,646 | 65 | `a548226d9fe4c1dd7193d79487f51e55d77ad58e327b1b58e966b910e484ccf4` |

The v2, v2_1k, and native files are published Dataverse transports. The local
v3 construction retransliterates names from the native table with electoral
roll word maps, then combines those states with v2 states that did not require
retransliteration. The v3 file is currently gitignored and is not a published
runtime table. The hash in the table identifies the legacy gzip bytes used to
train and audit the shipped checkpoint. Those bytes predate the canonical
writer below and are not the expected hash of a newly constructed artifact.

Build the full-coverage artifact from the repository root with both published
transport inputs:

```console
python model_training/retransliterate_native.py \
  --native /tmp/naampy_v2_native.csv.gz \
  --v2 /tmp/naampy_v2.csv.gz \
  --out model_training/data/naampy_v3.csv.gz
```

The construction program is included in source distributions. It uses a stable
row order, fixed CSV formatting and line endings, and a gzip stream with no
embedded filename and timestamp zero, so identical input tables, word maps,
software, and code produce the same bytes and SHA-256 digest. The required word
maps are not yet published at immutable revisions, so this is a deterministic
construction procedure, not yet an independently reproducible data release or
a source of a published canonical v3 hash.

## Data dictionary

One source row represents an aggregated state, birth-year, and first-name
cell. The key is unique in all four profiled tables.

| Column | Type | Unit and universe | Missing values | Provenance and checks |
| --- | --- | --- | --- | --- |
| `state` | string | State or union-territory key for every aggregated cell | None observed | Electoral-roll processing pipeline |
| `birth_year` | whole year | Birth year attached to every aggregated cell | None observed | v2 transports store whole years as floating-point text; the runtime cache validates and casts them to `int16` |
| `first_name` | string | Processed first-name text for every cell | One blank row in local v3; none in published tables | Native input contains script-specific text; v2 and v2_1k contain Latin letters only |
| `n_female` | nonnegative integer | Source records labeled female in the cell | None observed | Counts are nonnegative in every profiled table |
| `n_male` | nonnegative integer | Source records labeled male in the cell | None observed | Counts are nonnegative in every profiled table |
| `n_third_gender` | nonnegative integer | Source records labeled third gender in the cell | None observed | Only 65 represented records in v2 and local v3; zero in the native table |
| `prop_female` | proportion | `n_female / (n_female + n_male + n_third_gender)` | None observed | Recomputed values matched the source within floating-point precision |

No profiled table contains a zero-total cell, negative count, or duplicate key.
The local v3 table contains 20,226 rows whose first-name value is not purely
alphabetic, mostly because retransliteration produced spaces or multiple words.

## Recode ledger

The model-training loader aggregates every source row by `first_name`, sums the
female and male counts, and computes the target as:

```text
female_proportion = female_count / (female_count + male_count)
```

It then applies these filters:

| Source value | Model value | Reason and consequence |
| --- | --- | --- |
| Names shorter than 3 or longer than 19 characters | Excluded | Matches the current model's training scope |
| Names containing nonalphabetic characters | Excluded | Removes spaces, punctuation, and malformed values |
| Names containing the same character three times in sequence | Excluded | Removes a documented noisy-name pattern |
| Characters outside Latin `a` to `z` | Dropped by the encoder | A name with no remaining characters is excluded |
| Female and male counts | `person_count` | Used as the training and person-weighted evaluation weight |
| Female share among female and male counts | Soft binary target | The model does not use the third-gender count |

These steps produce 124,447 usable unique Latin-script names from the local v3
construction. The evaluation code tests the recodes through the same loader
used by training.

## Split contract

The shipped checkpoint used a seeded 80 percent training and 20 percent
development split over unique aggregated names. Exact names do not cross that
boundary. Its membership is fixed by the original recipe's seed of zero; the
evaluator does not expose that seed as an option. Bootstrap randomness has a
separate option and cannot change split membership. The report records canonical
SHA-256 hashes of the names in each partition. The training loop reported the
complete development result after every epoch, so the result is developmental
rather than confirmatory evidence.

Evaluating a custom `--model` also requires `--training-manifest`. The evaluator
verifies the manifest's data hash, checkpoint hash, split recipe, and all four
name-membership hashes before using its calibration and test partitions. It
refuses a custom checkpoint without that evidence instead of applying the
shipped checkpoint's split to unrelated weights.

The current-checkpoint audit reproduces that boundary, then partitions the
development names into balanced calibration and test halves. The balancing
algorithm considers name count, represented records, female count, and male
count. Calibration fits a positive-scale logistic transform. Test metrics are
computed once after calibration and receive 95 percent name-cluster percentile
bootstrap intervals.

Future training must use the new stratified 70 percent training, 10 percent
validation, 10 percent calibration, and 10 percent test contract in
`model_training.evaluation`. Model selection may inspect only validation
metrics. Calibration may use only the calibration partition. The final test
partition may be scored only after the model and calibration method are frozen.

Thresholded precision, recall, and F1 use fractional female shares to form
expected individual-label confusion counts. Majority-name accuracy is named
separately. Aggregate-composition mean squared error measures squared error
against the observed name-level share; expected binary Brier score additionally
includes the irreducible Bernoulli variance `female_share * (1 - female_share)`.

The raw `.pt` checkpoint emits uncalibrated probabilities. Calibrated serving
requires the accompanying JSON manifest and its fitted scale and intercept.
Naampy's current runtime loads only the raw checkpoint and does not apply the
offline calibration reported by the evaluation and training tools.

## Join contract for the hybrid product

The neural baseline does not join tables. The future hybrid evaluation will
join a test-name table on the left to a lookup table aggregated to one row per
normalized first name on the right.

| Contract item | Requirement |
| --- | --- |
| Left table | Every evaluated name, preserved once and in its original order |
| Right table | One row per normalized first name after the declared state and birth-year filters |
| Key | Normalized first name |
| Cardinality | Many evaluation rows to one lookup row |
| Expected result rows | Exactly the number of left-table rows |
| Miss behavior | Route to the binary neural model only when the script is supported; otherwise abstain |
| Required diagnostics | Match rate overall and by state, year, script, support, and target class; unmatched examples; row-count identity |

The lookup and neural paths do not share a label space. Lookup rows retain the
published female, male, and third-gender composition. Neural fallback rows
estimate only the binary female and male source target. The product must expose
that difference and must not fill an unsupported third-gender probability with
zero.

## Open provenance questions

- The published source documentation must establish whether represented
  records can repeat the same person across roll editions.
- Birth years before plausible modern electoral cohorts need a source-level
  explanation or an explicit support rule.
- The native string `[লে` and similar malformed-looking values need review
  against the extraction source before any native-script model uses them.
- The v3 transliteration word maps and construction environment need immutable
  hashes before v3 can support a release claim.
- The v3 training table needs publication in a versioned data repository or a
  fully reproducible build from immutable inputs.
