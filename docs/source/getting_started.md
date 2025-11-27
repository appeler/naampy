# Getting Started

This guide will help you install naampy and make your first predictions.

## Installation

### Requirements

- Python 3.11 or 3.12
- pip or uv package manager

### Install from PyPI

The easiest way to install naampy is from PyPI:

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

## Quick Start

### Your First Prediction

Here's a simple example to get you started:

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Create a DataFrame with Indian first names
names = pd.DataFrame({
    'first_name': ['Priyanka', 'Rahul', 'Anjali', 'Vikram']
})

# Get gender predictions based on electoral roll data
results = in_rolls_fn_gender(names, 'first_name')

# View the results
print(results[['first_name', 'prop_female', 'prop_male', 'n_male', 'n_female']])
```

Output:
```
  first_name  prop_female  prop_male  n_male  n_female
0   Priyanka        0.99       0.01     245     23567
1      Rahul        0.01       0.99   89234       892
2     Anjali        0.98       0.02     567     34521
3     Vikram        0.02       0.98   45678       923
```

### Using the Machine Learning Model

For names not in the electoral rolls database, use the ML model:

```python
from naampy import predict_fn_gender

# List of first names
names = ['Aadhya', 'Reyansh', 'Kiara']

# Get predictions using the neural network model
predictions = predict_fn_gender(names)

print(predictions)
```

Output:
```
      name pred_gender  pred_prob
0   Aadhya      female      0.897
1  Reyansh        male      0.923
2    Kiara      female      0.945
```

## Understanding the Output

### Electoral Roll Data (`in_rolls_fn_gender`)

The function returns a DataFrame with the original data plus these columns:

- **prop_female**: Proportion of people with this name who are female (0-1)
- **prop_male**: Proportion of people with this name who are male (0-1)
- **prop_third_gender**: Proportion of people with this name who are third gender (0-1)
- **n_female**: Total count of females with this name in the dataset
- **n_male**: Total count of males with this name in the dataset
- **n_third_gender**: Total count of third gender individuals with this name

### ML Model Predictions (`predict_fn_gender`)

The function returns a DataFrame with:

- **name**: The input name
- **pred_gender**: Predicted gender ('male' or 'female')
- **pred_prob**: Confidence score for the prediction (0-1)

## Data Download

When you first use `in_rolls_fn_gender`, it will automatically download the electoral roll dataset (about 30MB). This data is cached locally for future use. You'll see a message like:

```
Downloading naampy data from the server (naampy_v2_1k.csv.gz)...
```

Subsequent runs will use the cached data:

```
Using cached naampy data from local (/path/to/cache/naampy_v2_1k.csv.gz)...
```

## Next Steps

- Read the [User Guide](user_guide.md) for more detailed examples
- Check the [API Reference](api_reference.md) for all available options
- Learn about the [methodology and data sources](about.md)