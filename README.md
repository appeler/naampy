# naampy: Infer Sociodemographic Characteristics from Indian Names

[![CI](https://github.com/appeler/naampy/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/naampy/actions/workflows/ci.yml)
[![Documentation](https://github.com/appeler/naampy/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/naampy/actions/workflows/docs.yml)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/gojiberries/naampy)
[![image](https://img.shields.io/pypi/v/naampy.svg)](https://pypi.python.org/pypi/naampy)
[![image](https://static.pepy.tech/badge/naampy)](https://pepy.tech/project/naampy)

<!-- START:description -->

The ability to programmatically and reliably infer the social attributes
of a person from their name can be useful for a broad set of tasks, from
estimating bias in coverage of women in the media to estimating bias in
lending against certain social groups. But unlike the American Census
Bureau, which produces a list of last names and first names, which can
(and are) used to infer the gender, race, ethnicity, etc., from names,
the Indian government produces no such commensurate datasets. Hence
inferring the relationship between gender, ethnicity, language group,
etc., and names has generally been done with small datasets constructed
in an ad-hoc manner.

We fill this yawning gap. Using data from the [Indian Electoral
Rolls](https://github.com/in-rolls/electoral_rolls) (parsed data
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MUEGDT)),
we estimate the proportion female, male, and [third sex]{.title-ref}
(see [here](https://en.wikipedia.org/wiki/Third_gender)) for a
particular [first name, year, and state.]{.title-ref}

Please also check out [pranaam](https://github.com/appeler/pranaam) that
uses land record data from Bihar to infer religion based on the name.
The package uses [indicate](https://github.com/in-rolls/indicate) to
transliterate Hindi to English.
<!-- END:description -->

<!-- START:streamlit -->
## Try it Online

Check out our interactive [Streamlit App](https://naampy.streamlit.app/) to test naampy with your own names!
<!-- END:streamlit -->

<!-- START:features -->
## Features

- 🚀 **Easy to use**: Simple API with just two main functions
- 📊 **Data-driven**: Based on millions of names from Indian Electoral Rolls
- 🎯 **Accurate**: Provides confidence scores with predictions
- 🗺️ **State-specific**: Get region-specific predictions for better accuracy
- 🤖 **ML-powered**: Neural network fallback for names not in database
- 📈 **Comprehensive**: Covers 31 states and union territories
<!-- END:features -->

<!-- START:installation -->
## Installation

### Requirements

- Python 3.11
- pip or uv package manager

### Install from PyPI

We strongly recommend installing naampy inside a Python virtual environment (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments)):

```bash
pip install naampy
```

Or if you're using uv:

```bash
uv pip install naampy
```

### Install from Source

To install the latest development version:

```bash
git clone https://github.com/appeler/naampy.git
cd naampy
pip install -e .
```
<!-- END:installation -->

<!-- START:quick_start -->
## Quick Start

### Basic Usage

```python
import pandas as pd
from naampy import in_rolls_fn_gender, predict_fn_gender

# Create a DataFrame with names
names_df = pd.DataFrame({"name": ["Priyanka", "Rahul", "Anjali"]})

# Get gender predictions from electoral roll data
result = in_rolls_fn_gender(names_df, "name")
print(result[["name", "prop_female", "prop_male"]])
```

### Using the ML Model

For names not in the electoral roll database:

```python
# Use the neural network model for predictions
names = ["Aadhya", "Reyansh", "Kiara"]
predictions = predict_fn_gender(names)
print(predictions)
```
<!-- END:quick_start -->

<!-- START:detailed_usage -->
## Detailed Usage Examples

### Electoral Roll Data

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Sample data
names = [{"name": "gaurav"}, {"name": "yasmin"}, {"name": "deepti"}]
df = pd.DataFrame(names)

result = in_rolls_fn_gender(df, "name")
print(result[["name", "n_male", "n_female", "prop_female", "prop_male"]])
```

### Machine Learning Predictions

```python
from naampy import predict_fn_gender

# Names not in electoral roll database
names = ["nabha", "hrithik", "kiara", "reyansh"]
predictions = predict_fn_gender(names)
print(predictions)
```

<!-- END:detailed_usage -->

<!-- START:functionality -->
## How it Works

When you first run `in_rolls_fn_gender`, it downloads the selected raw table from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WZGJBM), validates its schema, and stores a typed Parquet cache under `~/.naampy`. The downloaded CSV.gz is only a temporary transport file and is removed after conversion. Subsequent runs use the Parquet cache.

The package provides two complementary approaches:

1. **Electoral Roll Data**: Statistical data from millions of Indian voters
2. **Machine Learning Model**: Neural network trained on name patterns

For names not found in the electoral roll database, the package automatically falls back to the ML model.
<!-- END:functionality -->

<!-- START:links -->
## Documentation

For comprehensive documentation, examples, and API reference, visit:
**[https://appeler.github.io/naampy/](https://appeler.github.io/naampy/)**

## Authors

Suriyan Laohaprapanon, Gaurav Sood, and Rajashekar Chintalapati

## Related Projects

- [appeler/pranaam](https://github.com/appeler/pranaam) — Predict religion based on names
- [appeler/outkast](https://github.com/appeler/outkast) — Map last names to caste categories
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI-powered name parsing
<!-- END:links -->
