"""Retransliterate Naampy's native-script first names with the eroll corpora.

The program romanizes each v2_native first name with its language corpus, then
reaggregates state, birth year, and romanized first name to source-label counts.

Run in eroll's 3.13 venv (has ``eroll`` + the corpora under ``eroll_transliteration/data/``):

    .../eroll_transliteration/.venv/bin/python retransliterate_native.py \
        --native-table /tmp/naampy_v2_native.csv.gz \
        --published-v2-table /tmp/naampy_v2.csv.gz \
        --output model_training/data/naampy_v3.csv.gz
"""

import argparse
import csv
import gzip
import io
import logging
import re
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

STATE_TO_LANGUAGE = {
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


def normalize_romanized_name(name: str) -> str:
    """Return lowercase ASCII letters and normalized spaces."""
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    letters_and_spaces = "".join(
        character
        if (character.isascii() and character.isalpha()) or character == " "
        else " "
        for character in ascii_name
    )
    return " ".join(letters_and_spaces.split()).strip().lower()


def _load_language_corpus_configuration():
    """Return language-specific corpus paths and native-script patterns."""
    from eroll.states import STATES

    language_configuration: dict[str, tuple[Path, re.Pattern[str]]] = {}
    for state_configuration in STATES.values():
        language_configuration.setdefault(
            state_configuration.language,
            (state_configuration.corpus_csv, state_configuration.native_run),
        )
    return language_configuration


def _load_transliteration_map(corpus_path: Path) -> dict[str, str]:
    transliteration_by_native_text: dict[str, str] = {}
    with gzip.open(corpus_path, "rt", encoding="utf-8", newline="") as corpus_file:
        reader = csv.reader(corpus_file)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                transliteration_by_native_text[row[0]] = row[1]
    return transliteration_by_native_text


def write_deterministic_gzip_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Write a canonical gzip-compressed CSV atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with (
            temporary_path.open("wb") as raw_output,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output,
            io.TextIOWrapper(
                compressed_output, encoding="utf-8", newline=""
            ) as text_output,
        ):
            table.to_csv(
                text_output,
                index=False,
                float_format="%.17g",
                lineterminator="\n",
            )
        temporary_path.replace(output_path)
        output_path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def romanize_first_name(
    name: str,
    transliteration_by_native_text: dict[str, str],
    native_text_pattern: re.Pattern[str],
) -> str:
    """Romanize a native first name; '' if any native run remains (not in the corpus)."""
    romanized_name = native_text_pattern.sub(
        lambda match: transliteration_by_native_text.get(
            match.group(0), match.group(0)
        ),
        name,
    )
    if native_text_pattern.search(romanized_name):
        return ""
    return normalize_romanized_name(romanized_name)


def main() -> None:
    """Retransliterate native names and write the aggregated v3 table."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--native-table", required=True, help="naampy v2_native csv.gz"
    )
    argument_parser.add_argument("--output", required=True, help="output v3 csv.gz")
    argument_parser.add_argument(
        "--published-v2-table",
        default=None,
        help="naampy v2 csv.gz; if given, merge v2's non-native states in (full coverage).",
    )
    arguments = argument_parser.parse_args()

    native_roll_table = pd.read_csv(arguments.native_table, dtype={"first_name": str})
    native_roll_table = native_roll_table[
        native_roll_table.state.isin(STATE_TO_LANGUAGE)
    ].copy()
    language_configuration = _load_language_corpus_configuration()
    native_roll_table["romanized_first_name"] = ""

    # Process one language at a time so only one (large) word-map is resident.
    for language in sorted(set(STATE_TO_LANGUAGE.values())):
        states = [
            state
            for state, state_language in STATE_TO_LANGUAGE.items()
            if state_language == language
        ]
        language_state_rows = native_roll_table.state.isin(states)
        corpus_path, native_text_pattern = language_configuration[language]
        LOGGER.info(
            "Loading %s for %s %s states",
            corpus_path.name,
            len(states),
            language,
        )
        transliteration_by_native_text = _load_transliteration_map(corpus_path)
        unique_native_names = (
            native_roll_table.loc[language_state_rows, "first_name"].dropna().unique()
        )
        romanized_name_by_native_name = {
            native_name: romanize_first_name(
                native_name,
                transliteration_by_native_text,
                native_text_pattern,
            )
            for native_name in unique_native_names
        }
        native_roll_table.loc[language_state_rows, "romanized_first_name"] = (
            native_roll_table.loc[language_state_rows, "first_name"].map(
                romanized_name_by_native_name
            )
        )
        del transliteration_by_native_text, romanized_name_by_native_name
        retained_rows = native_roll_table.loc[
            language_state_rows & (native_roll_table.romanized_first_name != "")
        ]
        label_count_columns = ["n_female", "n_male", "n_third_gender"]
        source_label_record_count = (
            native_roll_table.loc[language_state_rows, label_count_columns]
            .to_numpy()
            .sum()
        )
        retained_source_label_record_count = (
            retained_rows[label_count_columns].to_numpy().sum()
        )
        LOGGER.info(
            "%s: %s unique names; retained %.1f%% of represented records",
            language,
            f"{len(unique_native_names):,}",
            100
            * retained_source_label_record_count
            / max(1, source_label_record_count),
        )

    romanized_rows = native_roll_table[
        native_roll_table.romanized_first_name.str.len() > 2
    ].copy()
    romanized_rows["first_name"] = romanized_rows["romanized_first_name"]
    aggregated_roll_table = romanized_rows.groupby(
        ["state", "birth_year", "first_name"], as_index=False
    )[["n_female", "n_male", "n_third_gender"]].sum()
    represented_source_label_record_count = (
        aggregated_roll_table.n_female
        + aggregated_roll_table.n_male
        + aggregated_roll_table.n_third_gender
    ).clip(lower=1)
    aggregated_roll_table["prop_female"] = (
        aggregated_roll_table.n_female / represented_source_label_record_count
    )

    if arguments.published_v2_table:
        published_v2_table = pd.read_csv(
            arguments.published_v2_table, dtype={"first_name": str}
        )
        non_retransliterated_rows = published_v2_table[
            ~published_v2_table.state.isin(STATE_TO_LANGUAGE)
        ]
        LOGGER.info(
            "Combining %s retransliterated states with %s v2 states",
            aggregated_roll_table.state.nunique(),
            non_retransliterated_rows.state.nunique(),
        )
        aggregated_roll_table = pd.concat(
            [
                aggregated_roll_table,
                non_retransliterated_rows[aggregated_roll_table.columns],
            ],
            ignore_index=True,
        )

    aggregated_roll_table = aggregated_roll_table.sort_values(
        ["state", "birth_year", "first_name"], kind="mergesort"
    ).reset_index(drop=True)
    output_path = Path(arguments.output)
    write_deterministic_gzip_csv(aggregated_roll_table, output_path)
    LOGGER.info(
        "Wrote %s state-year-name rows and %s unique English names to %s",
        f"{len(aggregated_roll_table):,}",
        f"{aggregated_roll_table.first_name.nunique():,}",
        output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
