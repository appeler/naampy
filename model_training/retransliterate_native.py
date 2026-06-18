"""Re-transliterate naampy's native-script first names with the eroll corpora.

Reads naampy's ``v2_native`` (state, first_name[native], birth_year, gender counts) and romanizes
each native first name via the per-language word-map corpora harvested in eroll_transliteration,
then re-aggregates ``(state, birth_year, first_name_en)`` -> gender counts. Produces a cleaner
English first-name->gender dataset (no roll re-harvest).

Run in eroll's 3.13 venv (has ``eroll`` + the corpora under ``eroll_transliteration/data/``):

    .../eroll_transliteration/.venv/bin/python retransliterate_native.py \
        --native /tmp/naampy_v2_native.csv.gz --out model_training/data/naampy_v3.csv.gz
"""

import argparse
import csv
import gzip
import re
import unicodedata
from pathlib import Path

import pandas as pd
from eroll.states import STATES

# naampy state slug -> eroll corpus language.
NAAMPY_STATE2LANG = {
    "assam": "bengali",
    "tripura": "bengali",
    "bihar": "hindi",
    "chandigarh": "hindi",
    "haryana": "hindi",
    "himachal": "hindi",
    "jharkhand": "hindi",
    "mp": "hindi",
    "rajasthan": "hindi",
    "up": "hindi",
    "uttarakhand": "hindi",
    "gujarat": "gujarati",
    "karnataka": "kannada",
    "maharastra": "marathi",
    "odisha": "odia",
    "punjab": "punjabi",
}


def to_ascii(s: str) -> str:
    """Lowercase ASCII letters + spaces only (mirror of instate name_tables.to_ascii)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(c if (c.isascii() and c.isalpha()) or c == " " else " " for c in s)
    return " ".join(s.split()).strip().lower()


def _lang_config():
    """language -> (corpus_csv Path, native_run regex), taken from the eroll STATES registry."""
    out: dict[str, tuple[Path, re.Pattern[str]]] = {}
    for cfg in STATES.values():
        out.setdefault(cfg.language, (cfg.corpus_csv, cfg.native_run))
    return out


def _load_word_map(corpus_csv) -> dict[str, str]:
    word_map: dict[str, str] = {}
    with gzip.open(corpus_csv, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                word_map[row[0]] = row[1]
    return word_map


def romanize(name: str, word_map: dict[str, str], native_run: re.Pattern[str]) -> str:
    """Romanize a native first name; '' if any native run remains (not in the corpus)."""
    sub = native_run.sub(lambda m: word_map.get(m.group(0), m.group(0)), name)
    if native_run.search(sub):
        return ""
    return to_ascii(sub)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="naampy v2_native csv.gz")
    ap.add_argument("--out", required=True, help="output v3 csv.gz")
    ap.add_argument(
        "--v2",
        default=None,
        help="naampy v2 csv.gz; if given, merge v2's non-native states in (full coverage).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.native, dtype={"first_name": str})
    df = df[df.state.isin(NAAMPY_STATE2LANG)].copy()
    langcfg = _lang_config()
    df["fn_en"] = ""

    # Process one language at a time so only one (large) word-map is resident.
    for lang in sorted(set(NAAMPY_STATE2LANG.values())):
        states = [s for s, lng in NAAMPY_STATE2LANG.items() if lng == lang]
        mask = df.state.isin(states)
        corpus_csv, native_run = langcfg[lang]
        print(
            f"[{lang}] loading {corpus_csv.name} for {len(states)} states ...",
            flush=True,
        )
        word_map = _load_word_map(corpus_csv)
        uniq = df.loc[mask, "first_name"].dropna().unique()
        romap = {nm: romanize(nm, word_map, native_run) for nm in uniq}
        df.loc[mask, "fn_en"] = df.loc[mask, "first_name"].map(romap)
        del word_map, romap
        kept = df.loc[mask & (df.fn_en != "")]
        wt = df.loc[mask, ["n_female", "n_male", "n_third_gender"]].to_numpy().sum()
        wtk = kept[["n_female", "n_male", "n_third_gender"]].to_numpy().sum()
        print(
            f"[{lang}] {len(uniq):,} uniq names; weight kept {wtk / max(1, wt):.1%}",
            flush=True,
        )

    # Keep rows with a valid English name (len>2), re-aggregate, recompute prop_female.
    out = df[(df.fn_en.str.len() > 2)].copy()
    out["first_name"] = out["fn_en"]
    agg = out.groupby(["state", "birth_year", "first_name"], as_index=False)[
        ["n_female", "n_male", "n_third_gender"]
    ].sum()
    tot = (agg.n_female + agg.n_male + agg.n_third_gender).clip(lower=1)
    agg["prop_female"] = agg.n_female / tot

    if args.v2:  # keep v2's non-native states for full coverage; native states use v3
        v2 = pd.read_csv(args.v2, dtype={"first_name": str})
        others = v2[~v2.state.isin(NAAMPY_STATE2LANG)]
        print(
            f"[merge] v3 native {agg.state.nunique()} states + v2 "
            f"{others.state.nunique()} non-native states",
            flush=True,
        )
        agg = pd.concat([agg, others[agg.columns]], ignore_index=True)

    agg = agg.sort_values(["state", "birth_year", "first_name"])
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(outp, index=False, compression="gzip")
    print(
        f"[v3] {len(agg):,} (state,year,first_name_en) rows "
        f"({agg.first_name.nunique():,} unique english names) -> {outp}",
        flush=True,
    )


if __name__ == "__main__":
    main()
