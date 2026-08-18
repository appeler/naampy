# Usage examples

## Electoral-roll lookup

Pass the input column explicitly. Naampy preserves the input frame and its index,
replaces any colliding naampy output columns, and appends counts and proportions.

```python
import pandas as pd

from naampy import in_rolls_fn_gender

people = pd.DataFrame(
    {"first_name": ["Priya", "Rahul", "Anjali"]},
    index=pd.Index([101, 102, 103], name="person_id"),
)
result = in_rolls_fn_gender(people, "first_name")
print(result[["first_name", "prop_female", "prop_male"]])
```

The first call downloads the selected Dataverse table, validates it, and creates a
typed Parquet cache under `~/.naampy`. Later calls reuse that cache.

## Neural fallback

Names missing from an English electoral-roll table are scored by the pinned neural
model from [Hugging Face](https://huggingface.co/gojiberries/naampy).

```python
from naampy import predict_fn_gender

predictions = predict_fn_gender(["Aadhya", "Vivaan", "Kiara"])
print(predictions)
```

Set `NAAMPY_MODEL_DIR` to a directory containing `gender_lstm.pt` when an offline
deployment supplies the checkpoint itself. Standard Hugging Face authentication,
including `HF_TOKEN`, works automatically for Hub downloads.

These runtime probabilities are raw and uncalibrated. The calibration parameters
in the developmental evaluation report are used only by the offline evaluation
tool; the current runtime does not load or apply that JSON. A separately trained
checkpoint can be served as calibrated only when its accompanying training JSON
is also loaded and its scale and intercept are applied to the raw model logits.

## State and year filters

```python
punjab = in_rolls_fn_gender(people, "first_name", state="punjab")
cohort = in_rolls_fn_gender(people, "first_name", state="andhra", year=1985)
```

Use `InRollsFnData.list_states()` to inspect the states available in a dataset.
