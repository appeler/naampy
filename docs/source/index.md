# naampy Documentation

Welcome to **naampy** - a Python package for inferring sociodemographic characteristics from Indian names.

## What is naampy?

naampy helps you infer gender and other demographic information from Indian first names using data from the Indian Electoral Rolls. This can be useful for:

- Analyzing gender representation in datasets
- Estimating demographic biases in various contexts
- Research on Indian names and demographics
- Data enrichment and analysis

## Features

- 🚀 **Easy to use**: Simple API with just two main functions
- 📊 **Data-driven**: Based on millions of names from Indian Electoral Rolls
- 🎯 **Accurate**: Provides confidence scores with predictions
- 🗺️ **State-specific**: Get region-specific predictions for better accuracy
- 🤖 **ML-powered**: Includes a neural network model for name-based predictions

## Quick Links

### 🚀 [Getting Started](getting_started.md)
Installation instructions and your first prediction

### 📖 [User Guide](user_guide.md)
Detailed usage examples and best practices

### 📚 [API Reference](api_reference.md)
Complete API documentation

### ℹ️ [About](about.md)
Background and methodology

## Quick Example

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Create a DataFrame with names
names_df = pd.DataFrame({'name': ['Priyanka', 'Rahul', 'Kavita']})

# Get gender predictions
result = in_rolls_fn_gender(names_df, 'name')
print(result[['name', 'prop_female', 'prop_male']])
```

## Try it Online

Check out our [Streamlit App](https://naampy.streamlit.app/) for an interactive demo!

## Contents

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide
api_reference
about
```
