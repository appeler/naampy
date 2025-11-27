# User Guide

This guide provides detailed examples and best practices for using naampy effectively.

## Choosing the Right Function

naampy provides two main functions:

1. **`in_rolls_fn_gender`**: Use this when you want data-driven predictions based on actual electoral roll statistics
2. **`predict_fn_gender`**: Use this for names not in the database or when you need quick predictions without statistical data

## Working with Electoral Roll Data

### Basic Usage

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Your data
df = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'full_name': ['Priya Sharma', 'Rajesh Kumar', 'Anita Patel', 'Amit Singh'],
    'first_name': ['Priya', 'Rajesh', 'Anita', 'Amit']
})

# Get gender statistics
result = in_rolls_fn_gender(df, 'first_name')
```

### State-Specific Predictions

Different states in India may have different naming patterns. You can get more accurate predictions by specifying the state:

```python
# Get predictions specific to a state
result_delhi = in_rolls_fn_gender(df, 'first_name', state='delhi')
result_kerala = in_rolls_fn_gender(df, 'first_name', state='kerala')

# Compare the results
print("Delhi results:")
print(result_delhi[['first_name', 'prop_female']])

print("\nKerala results:")
print(result_kerala[['first_name', 'prop_female']])
```

### Available States

The following states are available in the dataset:

- andaman, andhra_pradesh, arunachal_pradesh, assam, bihar
- chandigarh, chhattisgarh, dadra, daman, delhi
- goa, gujarat, haryana, himachal_pradesh, jammu_kashmir
- jharkhand, karnataka, kerala, lakshadweep, madhya_pradesh
- maharashtra, manipur, meghalaya, mizoram, nagaland
- odisha, puducherry, punjab, rajasthan, sikkim
- tamil_nadu, tripura, uttar_pradesh, uttarakhand, west_bengal

### Year-Specific Data

You can also specify a year for temporal analysis:

```python
# Get predictions for a specific year
result_2017 = in_rolls_fn_gender(df, 'first_name', year=2017)
```

### Different Dataset Versions

naampy offers different dataset versions with varying levels of detail:

```python
# Use the full dataset (larger download, more comprehensive)
result_full = in_rolls_fn_gender(df, 'first_name', dataset='v2')

# Use the 1000+ frequency dataset (default, good balance)
result_1k = in_rolls_fn_gender(df, 'first_name', dataset='v2_1k')

# Use native script data
result_native = in_rolls_fn_gender(df, 'first_name', dataset='v2_native')
```

## Using the Machine Learning Model

### Basic Predictions

```python
from naampy import predict_fn_gender

# Single name
single_result = predict_fn_gender(['Deepika'])
print(single_result)

# Multiple names
names = ['Arjun', 'Kavya', 'Rohan', 'Shreya', 'Nikhil']
results = predict_fn_gender(names)
print(results)
```

### Filtering by Confidence

You can filter results based on prediction confidence:

```python
# Get predictions
results = predict_fn_gender(names)

# Only keep high-confidence predictions (>80%)
high_confidence = results[results['pred_prob'] > 0.8]
print(f"High confidence predictions: {len(high_confidence)}/{len(results)}")
print(high_confidence)
```

## Practical Examples

### Example 1: Analyzing Gender Distribution in a Dataset

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Load your data (e.g., employee list, student roster, etc.)
data = pd.read_csv('employees.csv')

# Assuming the data has a 'name' column with first names
enriched_data = in_rolls_fn_gender(data, 'name')

# Analyze gender distribution
gender_summary = {
    'likely_female': (enriched_data['prop_female'] > 0.8).sum(),
    'likely_male': (enriched_data['prop_male'] > 0.8).sum(),
    'ambiguous': ((enriched_data['prop_female'] > 0.2) & 
                  (enriched_data['prop_female'] < 0.8)).sum()
}

print("Gender Distribution Analysis:")
for category, count in gender_summary.items():
    percentage = (count / len(enriched_data)) * 100
    print(f"{category}: {count} ({percentage:.1f}%)")
```

### Example 2: Handling Missing Data

```python
# For names not in the electoral roll database, use the ML model as fallback
def get_gender_with_fallback(df, name_col):
    # Try electoral roll data first
    result = in_rolls_fn_gender(df, name_col)
    
    # Find rows with missing data (NaN in prop_female)
    missing_mask = result['prop_female'].isna()
    
    if missing_mask.any():
        # Get names that need ML prediction
        missing_names = result.loc[missing_mask, name_col].tolist()
        
        # Get ML predictions
        ml_predictions = predict_fn_gender(missing_names)
        
        # Fill in the missing data
        for idx, pred in zip(result[missing_mask].index, ml_predictions.itertuples()):
            if pred.pred_gender == 'female':
                result.loc[idx, 'prop_female'] = pred.pred_prob
                result.loc[idx, 'prop_male'] = 1 - pred.pred_prob
            else:
                result.loc[idx, 'prop_male'] = pred.pred_prob
                result.loc[idx, 'prop_female'] = 1 - pred.pred_prob
    
    return result
```

### Example 3: Batch Processing Large Datasets

```python
import pandas as pd
from naampy import in_rolls_fn_gender

def process_large_dataset(file_path, name_column, batch_size=10000):
    """Process a large dataset in batches to manage memory"""
    
    # Read the data in chunks
    chunks = pd.read_csv(file_path, chunksize=batch_size)
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing batch {i+1}...")
        
        # Process this batch
        result = in_rolls_fn_gender(chunk, name_column)
        processed_chunks.append(result)
    
    # Combine all results
    final_result = pd.concat(processed_chunks, ignore_index=True)
    return final_result

# Process a large file
result = process_large_dataset('large_dataset.csv', 'first_name')
```

## Best Practices

### 1. Data Preparation

- Ensure names are properly extracted (first names work best)
- Clean the data: remove titles (Mr., Ms., Dr.), numbers, and special characters
- Handle case consistently (the functions handle lowercase conversion internally)

### 2. Interpreting Results

- Use threshold values appropriate for your use case
- Consider names with `prop_female` between 0.3 and 0.7 as ambiguous
- Always validate results against known data when possible

### 3. Performance Tips

- For large datasets, process in batches
- The electoral roll data is cached after first download
- The ML model is loaded once and reused for multiple predictions

### 4. Handling Edge Cases

```python
# Handle empty or invalid names
def safe_predict(names):
    # Filter out empty/invalid names
    valid_names = [n for n in names if n and isinstance(n, str) and n.strip()]
    
    if not valid_names:
        return pd.DataFrame()
    
    return predict_fn_gender(valid_names)
```

## Limitations and Considerations

1. **Cultural Sensitivity**: Gender inference from names should be used thoughtfully and ethically
2. **Accuracy**: No name-based method is 100% accurate
3. **Unisex Names**: Some names are used across genders
4. **Regional Variations**: Names may have different gender associations in different regions
5. **Transliteration**: English transliterations of Indian names may vary

## Command Line Usage

naampy also provides a command-line interface:

```bash
# Get gender predictions for a name
in_rolls_fn_gender --name "Priya"

# Process a CSV file
in_rolls_fn_gender --input names.csv --name-column "first_name" --output results.csv
```

## Next Steps

- Check the [API Reference](api_reference.md) for detailed parameter documentation
- Learn about the [data and methodology](about.md)