"""Train the naampy gender char-BiLSTM (first name -> P(female)) on the v3 data.

Replaces the legacy TensorFlow char-CNN. Aggregates the (state,year,first_name) table to a
global ``first_name -> female_prop`` target (weighted by count), trains a torch ``CharBiLSTM``
with a single sigmoid output. CPU-trainable.

    python train_gender_lstm.py --data model_training/data/naampy_v3.csv.gz \
        --out naampy/model/gender_lstm.pt --epochs 12
"""

import argparse
import gzip
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAAMPY_ROOT = Path(__file__).resolve().parents[1]
# Import nnets.py directly (not the naampy package) to avoid the package __init__'s imports.
sys.path.insert(0, str(NAAMPY_ROOT / "naampy"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from nnets import (  # noqa: E402
    LSTM_DROPOUT,
    LSTM_EMB,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    VOCAB_SIZE,
    CharBiLSTM,
    encode_name,
    pad_encoded,
)

_REPEAT3 = re.compile(r"(.)\1\1")


def load_names(path, max_rows=None):
    """Aggregate to global first_name -> (female_prop, count); apply naampy's name filters."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, nrows=max_rows, dtype={"first_name": str})
    g = df.groupby("first_name")[["n_female", "n_male"]].sum()
    g = g[(g.n_female + g.n_male) > 0]
    names, enc, prop, wt = [], [], [], []
    for name, (nf, nm) in zip(g.index, g.to_numpy(), strict=True):
        if not (2 < len(name) < 20) or not name.isalpha() or _REPEAT3.search(name):
            continue
        e = encode_name(name)
        if not e:
            continue
        names.append(name)
        enc.append(e)
        prop.append(nf / (nf + nm))
        wt.append(float(nf + nm))
    return names, enc, prop, wt


@torch.no_grad()
def evaluate(model, enc, prop, device):
    """Held-out RMSE on P(female) + accuracy@0.5 (vs the >0.5 female label)."""
    model.eval()
    se = correct = 0.0
    for i in range(0, len(enc), 512):
        x, lengths = pad_encoded(enc[i : i + 512])
        p = torch.sigmoid(model(x.to(device), lengths)).squeeze(1).cpu().numpy()
        t = np.array(prop[i : i + 512])
        se += float(((p - t) ** 2).sum())
        correct += float(((p > 0.5) == (t > 0.5)).sum())
    return (se / len(enc)) ** 0.5, correct / len(enc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-rows", type=int, default=None, help="cap (smoke test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    names, enc, prop, wt = load_names(args.data, args.max_rows)
    idx = list(range(len(enc)))
    random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    train_w = [wt[i] for i in train_idx]
    te = [enc[i] for i in test_idx]
    tp = [prop[i] for i in test_idx]
    print(
        f"names {len(names):,} (train {len(train_idx):,}/test {len(test_idx):,}) | device {device}",
        flush=True,
    )

    model = CharBiLSTM(
        VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs = args.batch_size

    for epoch in range(1, args.epochs + 1):
        model.train()
        sample = random.choices(train_idx, weights=train_w, k=args.samples_per_epoch)
        running = 0.0
        for i in range(0, len(sample), bs):
            chunk = sample[i : i + bs]
            x, lengths = pad_encoded([enc[j] for j in chunk])
            target = torch.tensor([[prop[j]] for j in chunk], dtype=torch.float32)
            logits = model(x.to(device), lengths)
            loss = F.binary_cross_entropy_with_logits(logits, target.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(chunk)
        rmse, acc = evaluate(model, te, tp, device)
        print(
            f"epoch {epoch:2d}  loss {running / len(sample):.4f}  "
            f"rmse {rmse:.4f}  acc {acc:.4f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
