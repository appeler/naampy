# Getting Started

This guide will help you install naampy and make your first predictions.

```{include} ../../README.md
:start-after: <!-- START:installation -->
:end-before: <!-- END:installation -->
```

```{include} ../../README.md
:start-after: <!-- START:quick_start -->
:end-before: <!-- END:quick_start -->
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

```{include} ../../README.md
:start-after: <!-- START:functionality -->
:end-before: <!-- END:functionality -->
```

## Next Steps

- Read the [User Guide](user_guide.md) for more detailed examples
- Check the [API Reference](api_reference.md) for all available options
- Learn about the [methodology and data sources](about.md)