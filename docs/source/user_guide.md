# User Guide

This guide provides advanced usage examples and best practices for using naampy effectively in production environments.

## Advanced Configuration

### Choosing the Right Dataset

naampy offers several dataset versions optimized for different use cases:

1. **`v2_1k`** (default): Names with 1000+ occurrences - best balance of coverage and quality
2. **`v2`**: Full dataset - most comprehensive but larger download
3. **`v2_native`**: Native script data - no ML fallback, useful for linguistic analysis
4. **`v2_en`**: English transliterations - standardized transliterations

### Function Selection Strategy

- **Use `in_rolls_fn_gender`** for production systems requiring statistical confidence
- **Use `predict_fn_gender`** for exploratory analysis or when electoral data is unavailable

## Advanced Electoral Roll Usage

### Regional Analysis Patterns

Different regions show distinct naming patterns. Leverage this for better accuracy:

```python
import pandas as pd
from naampy import in_rolls_fn_gender

# Multi-region analysis
def analyze_by_region(df, name_col, regions):
    results = {}
    for region in regions:
        results[region] = in_rolls_fn_gender(
            df, name_col, state=region, dataset='v2'
        )
    return results

# Compare regional differences
regions = ['delhi', 'kerala', 'maharashtra']
regional_data = analyze_by_region(df, 'first_name', regions)
```

### Confidence Scoring and Thresholds

Implement robust confidence scoring for production systems:

```python
def classify_with_confidence(df, name_col, confidence_threshold=0.8):
    """
    Classify names with configurable confidence thresholds.
    """
    result = in_rolls_fn_gender(df, name_col)
    
    # Create confidence-based classifications
    conditions = [
        (result['prop_female'] >= confidence_threshold),
        (result['prop_male'] >= confidence_threshold),
        ((result['prop_female'] >= 0.3) & (result['prop_female'] <= 0.7))
    ]
    
    choices = ['high_confidence_female', 'high_confidence_male', 'ambiguous']
    result['classification'] = pd.np.select(conditions, choices, 'unknown')
    
    return result
```

### Temporal Analysis

Analyze naming trends over time for demographic insights:

```python
def temporal_gender_analysis(df, name_col, birth_years):
    """
    Analyze gender proportions across different birth years.
    """
    temporal_results = {}
    for year in birth_years:
        result = in_rolls_fn_gender(df, name_col, year=year)
        temporal_results[year] = result.groupby(name_col).agg({
            'prop_female': 'mean',
            'prop_male': 'mean',
            'n_female': 'sum',
            'n_male': 'sum'
        })
    return temporal_results

# Analyze trends from 1980-2000
years = range(1980, 2001, 5)
trends = temporal_gender_analysis(df, 'first_name', years)
```

### Data Quality Assessment

Validate data quality before processing:

```python
def assess_data_quality(df, name_col):
    """
    Assess the quality and coverage of names in the dataset.
    """
    result = in_rolls_fn_gender(df, name_col)
    
    quality_metrics = {
        'total_names': len(df),
        'found_in_electoral_data': result['prop_female'].notna().sum(),
        'ml_predictions': result['pred_gender'].notna().sum(),
        'high_confidence_electoral': (
            (result['prop_female'] > 0.8) | (result['prop_male'] > 0.8)
        ).sum(),
        'coverage_rate': result['prop_female'].notna().mean()
    }
    
    return quality_metrics

# Assess your data
quality = assess_data_quality(df, 'first_name')
print(f"Electoral data coverage: {quality['coverage_rate']:.2%}")
```

### Production Deployment Patterns

#### Caching Strategy

```python
import functools
from typing import Dict, Tuple

class NaampyCache:
    def __init__(self):
        self._cache: Dict[Tuple, pd.DataFrame] = {}
    
    def get_cached_result(self, df_hash, namecol, state, year, dataset):
        cache_key = (df_hash, namecol, state, year, dataset)
        return self._cache.get(cache_key)
    
    def cache_result(self, result, df_hash, namecol, state, year, dataset):
        cache_key = (df_hash, namecol, state, year, dataset)
        self._cache[cache_key] = result
        return result

# Global cache instance
naampy_cache = NaampyCache()
```

## Advanced Machine Learning Usage

### Ensemble Methods

Combine electoral roll data with ML predictions for optimal accuracy:

```python
def ensemble_prediction(df, name_col, ml_weight=0.3):
    """
    Create ensemble predictions combining electoral data with ML.
    """
    electoral_result = in_rolls_fn_gender(df, name_col)
    
    # Get ML predictions for all names
    ml_result = predict_fn_gender(df[name_col].tolist())
    
    # Convert ML predictions to probabilities
    ml_prob_female = np.where(
        ml_result['pred_gender'] == 'female',
        ml_result['pred_prob'],
        1 - ml_result['pred_prob']
    )
    
    # Ensemble only for names with electoral data
    has_electoral = electoral_result['prop_female'].notna()
    
    ensemble_prob = electoral_result['prop_female'].copy()
    ensemble_prob.loc[has_electoral] = (
        (1 - ml_weight) * electoral_result.loc[has_electoral, 'prop_female'] +
        ml_weight * ml_prob_female[has_electoral]
    )
    
    electoral_result['ensemble_prob_female'] = ensemble_prob
    return electoral_result
```

### Model Performance Monitoring

```python
def monitor_ml_performance(predictions, known_labels=None):
    """
    Monitor ML model performance and prediction quality.
    """
    performance_metrics = {
        'total_predictions': len(predictions),
        'avg_confidence': predictions['pred_prob'].mean(),
        'high_confidence_rate': (predictions['pred_prob'] > 0.8).mean(),
        'gender_distribution': predictions['pred_gender'].value_counts(normalize=True)
    }
    
    # If ground truth is available
    if known_labels is not None:
        accuracy = (predictions['pred_gender'] == known_labels).mean()
        performance_metrics['accuracy'] = accuracy
    
    return performance_metrics
```

## Production Use Cases

### Enterprise Data Pipeline

```python
import logging
from typing import Optional

class NaampyPipeline:
    """Production-ready naampy pipeline with error handling and monitoring."""
    
    def __init__(self, default_dataset='v2_1k', log_level=logging.INFO):
        self.default_dataset = default_dataset
        self.logger = self._setup_logging(log_level)
        self.stats = {'processed': 0, 'errors': 0, 'cache_hits': 0}
    
    def process_dataframe(self, df: pd.DataFrame, name_col: str, 
                         state: Optional[str] = None, 
                         year: Optional[int] = None,
                         fallback_to_ml: bool = True) -> pd.DataFrame:
        """Process DataFrame with comprehensive error handling."""
        try:
            self.logger.info(f"Processing {len(df)} records")
            
            # Validate input
            self._validate_input(df, name_col)
            
            # Get electoral roll data
            result = in_rolls_fn_gender(
                df, name_col, state=state, year=year, dataset=self.default_dataset
            )
            
            # Add ML fallback if enabled
            if fallback_to_ml:
                result = self._add_ml_fallback(result, name_col)
            
            self.stats['processed'] += len(df)
            self.logger.info(f"Successfully processed {len(df)} records")
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    def _validate_input(self, df: pd.DataFrame, name_col: str):
        """Validate input data quality."""
        if name_col not in df.columns:
            raise ValueError(f"Column '{name_col}' not found in DataFrame")
        
        if df[name_col].isna().sum() > len(df) * 0.1:  # 10% threshold
            self.logger.warning(f"High missing values in {name_col}: {df[name_col].isna().sum()}")
    
    def get_statistics(self) -> dict:
        """Return pipeline processing statistics."""
        return self.stats.copy()
```

### Bias Detection and Analysis

```python
def analyze_gender_bias(df, name_col, group_col, confidence_threshold=0.8):
    """
    Analyze potential gender bias across different groups.
    """
    result = in_rolls_fn_gender(df, name_col)
    
    # Filter high-confidence predictions
    high_conf_female = result['prop_female'] > confidence_threshold
    high_conf_male = result['prop_male'] > confidence_threshold
    
    bias_analysis = df.groupby(group_col).agg({
        name_col: 'count'
    }).rename(columns={name_col: 'total_count'})
    
    bias_analysis['female_count'] = df[high_conf_female].groupby(group_col)[name_col].count()
    bias_analysis['male_count'] = df[high_conf_male].groupby(group_col)[name_col].count()
    
    bias_analysis['female_ratio'] = bias_analysis['female_count'] / bias_analysis['total_count']
    bias_analysis['male_ratio'] = bias_analysis['male_count'] / bias_analysis['total_count']
    
    # Calculate bias metrics
    overall_female_ratio = bias_analysis['female_count'].sum() / bias_analysis['total_count'].sum()
    bias_analysis['bias_score'] = abs(bias_analysis['female_ratio'] - overall_female_ratio)
    
    return bias_analysis.fillna(0)
```

### Scalable Batch Processing

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Iterator

class ScalableNaampyProcessor:
    """Scalable processor for large datasets using multiprocessing."""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or mp.cpu_count()
    
    def process_file_parallel(self, file_path: str, name_col: str, 
                            batch_size: int = 10000,
                            output_path: str = None) -> pd.DataFrame:
        """Process large files using parallel processing."""
        
        # Read file in chunks
        chunks = pd.read_csv(file_path, chunksize=batch_size)
        chunk_list = list(chunks)
        
        print(f"Processing {len(chunk_list)} batches with {self.max_workers} workers")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, name_col): i 
                for i, chunk in enumerate(chunk_list)
            }
            
            results = []
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append((chunk_idx, result))
                    print(f"Completed batch {chunk_idx + 1}")
                except Exception as e:
                    print(f"Batch {chunk_idx + 1} failed: {e}")
            
            # Sort by original order and combine
            results.sort(key=lambda x: x[0])
            final_result = pd.concat([r[1] for r in results], ignore_index=True)
            
            if output_path:
                final_result.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
            
            return final_result
    
    @staticmethod
    def _process_chunk(chunk: pd.DataFrame, name_col: str) -> pd.DataFrame:
        """Process a single chunk - static method for multiprocessing."""
        return in_rolls_fn_gender(chunk, name_col)

# Usage
processor = ScalableNaampyProcessor(max_workers=4)
result = processor.process_file_parallel(
    'large_dataset.csv', 
    'first_name',
    batch_size=5000,
    output_path='processed_results.csv'
)
```

## Advanced Best Practices

### Production Data Validation

```python
def advanced_data_validation(df, name_col):
    """Comprehensive data validation for production use."""
    validation_report = {
        'total_records': len(df),
        'missing_names': df[name_col].isna().sum(),
        'empty_names': (df[name_col] == '').sum(),
        'numeric_names': df[name_col].str.isnumeric().sum(),
        'non_alpha_names': (~df[name_col].str.isalpha()).sum(),
        'avg_name_length': df[name_col].str.len().mean(),
        'name_length_outliers': (
            (df[name_col].str.len() < 2) | (df[name_col].str.len() > 20)
        ).sum()
    }
    
    # Flag potential data quality issues
    quality_flags = []
    if validation_report['missing_names'] / len(df) > 0.05:
        quality_flags.append('High missing rate')
    if validation_report['numeric_names'] > 0:
        quality_flags.append('Contains numeric names')
    if validation_report['name_length_outliers'] / len(df) > 0.1:
        quality_flags.append('High outlier rate')
    
    validation_report['quality_flags'] = quality_flags
    return validation_report
```

### Error Recovery Strategies

```python
def robust_processing_with_retry(df, name_col, max_retries=3, fallback_dataset='v2_1k'):
    """Robust processing with automatic retry and fallback mechanisms."""
    for attempt in range(max_retries):
        try:
            # Try with default settings first
            if attempt == 0:
                return in_rolls_fn_gender(df, name_col)
            # Use fallback dataset on retry
            elif attempt == 1:
                return in_rolls_fn_gender(df, name_col, dataset=fallback_dataset)
            # Use ML-only approach as last resort
            else:
                names = df[name_col].dropna().tolist()
                ml_result = predict_fn_gender(names)
                # Merge back with original df
                return df.merge(ml_result, left_on=name_col, right_on='name', how='left')
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return df  # Fallback return
```

## Ethical Guidelines and Limitations

### Responsible AI Usage

```python
def ethical_gender_inference(df, name_col, purpose, confidence_threshold=0.8):
    """
    Implement ethical guidelines for gender inference.
    """
    # Document the purpose and methodology
    metadata = {
        'purpose': purpose,
        'timestamp': datetime.now().isoformat(),
        'confidence_threshold': confidence_threshold,
        'methodology': 'Indian Electoral Rolls + ML fallback',
        'limitations': [
            'Accuracy not guaranteed for individuals',
            'Based on statistical patterns, not individual identity',
            'May not reflect personal gender identity',
            'Regional and temporal variations exist'
        ]
    }
    
    result = in_rolls_fn_gender(df, name_col)
    
    # Add confidence indicators
    result['confidence_level'] = pd.cut(
        result['prop_female'].fillna(result['pred_prob']),
        bins=[0, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Add metadata
    result.attrs['processing_metadata'] = metadata
    
    return result
```

### Bias Monitoring

```python
class BiasMonitor:
    """Monitor and report potential biases in predictions."""
    
    def __init__(self):
        self.monitoring_data = []
    
    def log_prediction_batch(self, predictions, context):
        """Log prediction batch for bias monitoring."""
        batch_stats = {
            'timestamp': datetime.now(),
            'context': context,
            'total_predictions': len(predictions),
            'female_ratio': (predictions['pred_gender'] == 'female').mean(),
            'avg_confidence': predictions['pred_prob'].mean(),
            'low_confidence_rate': (predictions['pred_prob'] < 0.6).mean()
        }
        self.monitoring_data.append(batch_stats)
    
    def generate_bias_report(self):
        """Generate bias monitoring report."""
        df = pd.DataFrame(self.monitoring_data)
        
        report = {
            'time_range': (df['timestamp'].min(), df['timestamp'].max()),
            'total_predictions': df['total_predictions'].sum(),
            'overall_female_ratio': df['female_ratio'].mean(),
            'confidence_trends': df.groupby('context')['avg_confidence'].mean(),
            'bias_indicators': self._detect_bias_indicators(df)
        }
        
        return report
```

## Integration Patterns

### API Service Integration

```python
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)
pipeline = NaampyPipeline()  # From earlier example

@app.route('/predict', methods=['POST'])
def predict_gender():
    """REST API endpoint for gender prediction."""
    try:
        data = request.get_json()
        names = data.get('names', [])
        
        if not names:
            return jsonify({'error': 'No names provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame({'name': names})
        
        # Process
        result = pipeline.process_dataframe(df, 'name')
        
        # Format response
        predictions = []
        for _, row in result.iterrows():
            predictions.append({
                'name': row['name'],
                'gender': row.get('pred_gender', 'unknown'),
                'confidence': row.get('pred_prob', 0.0),
                'data_source': 'electoral' if pd.notna(row.get('prop_female')) else 'ml'
            })
        
        return jsonify({
            'predictions': predictions,
            'metadata': {
                'processed_count': len(names),
                'processing_time': '< 1s'  # Add actual timing
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Database Integration

```python
import sqlalchemy as sa

def process_database_table(connection, table_name, name_column, batch_size=10000):
    """Process names directly from database table."""
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM {table_name}"
    total_count = connection.execute(sa.text(count_query)).scalar()
    
    processed = 0
    results = []
    
    while processed < total_count:
        # Read batch
        query = f"""
        SELECT * FROM {table_name} 
        ORDER BY id 
        LIMIT {batch_size} OFFSET {processed}
        """
        
        batch_df = pd.read_sql(query, connection)
        
        # Process batch
        result = in_rolls_fn_gender(batch_df, name_column)
        
        # Prepare for database update
        update_data = result[[
            'id', 'prop_female', 'prop_male', 'pred_gender', 'pred_prob'
        ]].to_dict('records')
        
        results.extend(update_data)
        processed += len(batch_df)
        
        print(f"Processed {processed}/{total_count} records")
    
    return results
```

## Performance Optimization

### Memory Management

```python
import psutil
import gc

def memory_efficient_processing(file_path, name_col, memory_limit_gb=4):
    """Process large files with memory management."""
    memory_limit_bytes = memory_limit_gb * 1024**3
    
    def get_memory_usage():
        return psutil.Process().memory_info().rss
    
    chunk_size = 10000
    results = []
    
    for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
        # Check memory usage
        if get_memory_usage() > memory_limit_bytes:
            # Force garbage collection
            gc.collect()
            
            # Reduce chunk size if still over limit
            if get_memory_usage() > memory_limit_bytes:
                chunk_size = max(1000, chunk_size // 2)
                print(f"Reducing chunk size to {chunk_size}")
        
        # Process chunk
        result = in_rolls_fn_gender(chunk_df, name_col)
        results.append(result)
        
        # Clear intermediate variables
        del chunk_df, result
    
    return pd.concat(results, ignore_index=True)
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib

class PersistentNaampyCache:
    """Persistent caching for naampy results."""
    
    def __init__(self, cache_dir="./naampy_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, names_list, **kwargs):
        """Generate cache key from names and parameters."""
        content = str(sorted(names_list)) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, names_list, **kwargs):
        """Retrieve cached result if available."""
        cache_key = self._get_cache_key(names_list, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            return pd.read_pickle(cache_file)
        return None
    
    def cache_result(self, result, names_list, **kwargs):
        """Cache processing result."""
        cache_key = self._get_cache_key(names_list, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        result.to_pickle(cache_file)
        return result

# Usage
cache = PersistentNaampyCache()

def cached_processing(df, name_col, **kwargs):
    """Process with persistent caching."""
    names_list = df[name_col].tolist()
    
    # Try cache first
    cached_result = cache.get_cached_result(names_list, **kwargs)
    if cached_result is not None:
        return cached_result
    
    # Process and cache
    result = in_rolls_fn_gender(df, name_col, **kwargs)
    return cache.cache_result(result, names_list, **kwargs)
```

## Next Steps

- Review the [API Reference](api_reference.md) for complete documentation
- Understand the [methodology and data sources](about.md)
- Implement monitoring and bias detection for production use
- Consider ensemble methods for maximum accuracy