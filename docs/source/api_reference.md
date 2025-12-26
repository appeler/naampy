# API Reference

This page contains the complete API documentation for naampy.

## Main Functions

These are the two primary functions you'll use with naampy:

```{eval-rst}
.. autofunction:: naampy.in_rolls_fn_gender
```

```{eval-rst}
.. autofunction:: naampy.predict_fn_gender
```

## Core Classes

```{eval-rst}
.. autoclass:: naampy.InRollsFnData
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utility Functions

```{eval-rst}
.. automodule:: naampy.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Module Constants

### Available Datasets

```{eval-rst}
.. autodata:: naampy.in_rolls_fn.IN_ROLLS_DATA
   :annotation: = Dictionary mapping dataset versions to Harvard Dataverse URLs
```

The following dataset versions are available:

- **v1**: 12 states dataset (legacy)
- **v2**: Full 30 states dataset
- **v2_1k**: 30 states with 1000+ name occurrences (recommended default)
- **v2_native**: Native language dataset (16 states, no ML fallback)
- **v2_en**: English transliteration of v2_native

### Output Columns

```{eval-rst}
.. autodata:: naampy.in_rolls_fn.IN_ROLLS_COLS
   :annotation: = List of columns added by in_rolls_fn_gender()
```

The electoral roll functions add these columns to your DataFrame:

- `n_male`, `n_female`, `n_third_gender`: Count statistics
- `prop_male`, `prop_female`, `prop_third_gender`: Proportion statistics

## Command Line Interface

The package includes a command-line interface:

```bash
in_rolls_fn_gender input.csv -f first_name -o output.csv
```

```{eval-rst}
.. autofunction:: naampy.in_rolls_fn.main
```

For usage examples, see the [User Guide](user_guide.md).
