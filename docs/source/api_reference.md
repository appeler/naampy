# API Reference

Complete reference for all naampy functions and parameters.

## Main Functions

### `in_rolls_fn_gender`

Infer gender based on Indian Electoral Roll data.

```python
in_rolls_fn_gender(
    df: pd.DataFrame,
    namecol: str,
    state: Optional[str] = None,
    year: Optional[int] = None,
    dataset: str = "v2_1k"
) -> pd.DataFrame
```

#### Parameters

- **df** (`pd.DataFrame`): 
  - Input DataFrame containing the names to analyze
  - Required parameter

- **namecol** (`str`): 
  - Name of the column containing first names
  - This column should contain cleaned first names for best results
  - Required parameter

- **state** (`Optional[str]`): 
  - State for region-specific predictions
  - Default: `None` (uses aggregate data across all states)
  - Available states:
    - North: `delhi`, `haryana`, `himachal_pradesh`, `jammu_kashmir`, `punjab`, `uttarakhand`
    - South: `andhra_pradesh`, `karnataka`, `kerala`, `tamil_nadu`, `telangana`
    - East: `bihar`, `jharkhand`, `odisha`, `west_bengal`
    - West: `goa`, `gujarat`, `maharashtra`, `rajasthan`
    - Central: `chhattisgarh`, `madhya_pradesh`, `uttar_pradesh`
    - Northeast: `arunachal_pradesh`, `assam`, `manipur`, `meghalaya`, `mizoram`, `nagaland`, `sikkim`, `tripura`
    - Union Territories: `andaman`, `chandigarh`, `dadra`, `daman`, `lakshadweep`, `puducherry`

- **year** (`Optional[int]`): 
  - Year for temporal analysis
  - Default: `None` (uses most recent data)
  - Available years depend on the dataset

- **dataset** (`str`): 
  - Version of the electoral roll dataset to use
  - Default: `"v2_1k"`
  - Options:
    - `"v1"`: Original dataset
    - `"v2"`: Full updated dataset (largest, most comprehensive)
    - `"v2_1k"`: Names with 1000+ occurrences (recommended, good balance)
    - `"v2_native"`: Native script data
    - `"v2_en"`: English transliterated data

#### Returns

`pd.DataFrame`: Original DataFrame with additional columns:

- **prop_female**: Proportion of females with this name (0-1)
- **prop_male**: Proportion of males with this name (0-1)
- **prop_third_gender**: Proportion of third gender individuals with this name (0-1)
- **n_female**: Count of females with this name
- **n_male**: Count of males with this name
- **n_third_gender**: Count of third gender individuals with this name

#### Example

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Create sample data
df = pd.DataFrame({
    'employee_id': [101, 102, 103],
    'first_name': ['Amit', 'Priya', 'Rahul']
})

# Get gender statistics
result = in_rolls_fn_gender(df, 'first_name')

# State-specific analysis
delhi_result = in_rolls_fn_gender(df, 'first_name', state='delhi')

# Use comprehensive dataset
full_result = in_rolls_fn_gender(df, 'first_name', dataset='v2')
```

---

### `predict_fn_gender`

Predict gender using a machine learning model trained on Indian names.

```python
predict_fn_gender(first_names: list[str]) -> pd.DataFrame
```

#### Parameters

- **first_names** (`list[str]`): 
  - List of first names to predict gender for
  - Names are automatically converted to lowercase
  - Required parameter

#### Returns

`pd.DataFrame`: DataFrame with columns:

- **name**: The input name (lowercase)
- **pred_gender**: Predicted gender (`"male"` or `"female"`)
- **pred_prob**: Confidence score for the prediction (0-1)
  - Higher values indicate higher confidence
  - Values > 0.5 indicate the probability for the predicted gender

#### Model Details

- Uses a character-level neural network
- Trained on Indian electoral roll data
- Maximum name length: 24 characters
- Unknown characters are handled with an "UNK" token

#### Example

```python
from naampy import predict_fn_gender

# Single name
result = predict_fn_gender(['Deepika'])
print(result)
# Output:
#       name pred_gender  pred_prob
# 0  deepika      female      0.967

# Multiple names
names = ['Arjun', 'Priya', 'Vikram', 'Anjali']
results = predict_fn_gender(names)
print(results)
# Output:
#      name pred_gender  pred_prob
# 0   arjun        male      0.983
# 1   priya      female      0.991
# 2  vikram        male      0.978
# 3  anjali      female      0.995

# Filter high confidence predictions
high_conf = results[results['pred_prob'] > 0.9]
```

---

## Data Management

### Data Caching

The electoral roll data is automatically downloaded and cached on first use:

- **Cache Location**: `~/.naampy/` (or OS-specific app data directory)
- **Data Files**: 
  - `naampy_v2_1k.csv.gz` (~30MB, default)
  - `naampy_v2.csv.gz` (~150MB, full dataset)
  - Other versions as specified

### Manual Data Management

```python
from naampy.utils import get_app_file_path

# Get cache directory path
cache_dir = get_app_file_path("naampy", "")
print(f"Cache directory: {cache_dir}")

# Clear cache (if needed)
import shutil
shutil.rmtree(cache_dir, ignore_errors=True)
```

---

## Command Line Interface

naampy provides a CLI for batch processing:

```bash
in_rolls_fn_gender [OPTIONS]
```

### Options

- `--name TEXT`: Single name to analyze
- `--input PATH`: Input CSV file path
- `--name-column TEXT`: Column name containing first names
- `--output PATH`: Output CSV file path
- `--state TEXT`: State for region-specific analysis
- `--year INTEGER`: Year for temporal analysis
- `--dataset TEXT`: Dataset version to use

### Examples

```bash
# Analyze a single name
in_rolls_fn_gender --name "Priya"

# Process a CSV file
in_rolls_fn_gender \
  --input employees.csv \
  --name-column "first_name" \
  --output results.csv \
  --state "maharashtra"
```

---

## Error Handling

### Common Issues and Solutions

#### FileNotFoundError
```python
try:
    result = in_rolls_fn_gender(df, 'name')
except FileNotFoundError:
    print("Data file not found. It will be downloaded automatically.")
    # Retry or handle accordingly
```

#### Network Issues
```python
try:
    result = in_rolls_fn_gender(df, 'name')
except Exception as e:
    if "Cannot download" in str(e):
        print("Network issue. Please check your connection.")
    raise
```

#### Invalid State Names
```python
# Validate state name
valid_states = ['delhi', 'maharashtra', 'karnataka', ...]  # etc
if state and state not in valid_states:
    print(f"Invalid state: {state}")
```

---

## Performance Considerations

### Memory Usage

- **v2_1k dataset**: ~30MB download, ~100MB in memory
- **v2 full dataset**: ~150MB download, ~500MB in memory
- **Model**: ~10MB loaded once, reused for all predictions

### Speed Optimization

```python
# For large datasets, process in batches
def process_large_dataset(df, batch_size=10000):
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        result = in_rolls_fn_gender(batch, 'name')
        results.append(result)
    return pd.concat(results, ignore_index=True)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def process_batch(batch):
    return in_rolls_fn_gender(batch, 'name')

# Split data and process in parallel
chunks = np.array_split(large_df, 4)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, chunks))
final_result = pd.concat(results, ignore_index=True)
```