---
tags:
  - names
  - india
  - pytorch
---

# naampy gender checkpoint

`gender_lstm.pt` powers `naampy.predict_fn_gender`. The package downloads this
repository at an immutable commit, then caches the file locally.

The artifact is a PyTorch state dictionary for the character-level
bidirectional LSTM defined in `naampy.nnets`. It was trained from first-name and
gender counts derived from Indian electoral rolls. The training program and
architecture constants are maintained in the
[`naampy` repository](https://github.com/appeler/naampy) under
`model_training/` and `naampy/nnets.py`.

## Usage

```python
from naampy import predict_fn_gender

predictions = predict_fn_gender(["Priya", "Rahul"])
```

Set `NAAMPY_MODEL_DIR` to a directory containing `gender_lstm.pt` to bypass the
Hub download in controlled or offline deployments.

## Limitations

The output is a statistical estimate from historical administrative records,
not a statement about a person's gender identity. Binary labels reflect the
available training target and do not represent the full range of identities.
Romanization, spelling, regional coverage, shared names, and changes over time
can produce systematic errors. Names without Latin `a` to `z` characters are
left unscored. Do not use these predictions to make decisions about a person or
their access to services.

## Licensing

The `naampy` source code is MIT licensed. Consult the source dataset terms and
your intended use before redistributing or deploying the learned weights.
