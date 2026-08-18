"""Gender prediction from Indian first names using Electoral Roll data."""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ._resources import resolve_model
from ._tables import csv_to_parquet, validate_parquet
from .utils import download_file, get_app_file_path

LOGGER = logging.getLogger(__name__)

#: Harvard Dataverse URLs for Indian Electoral Roll datasets.
#:
#: Contains download URLs for different versions of the naampy gender prediction
#: datasets hosted on Harvard Dataverse. Each version contains electoral roll
#: statistics from different numbers of Indian states and territories.
#:
#: Dataset versions:
#:     - v1: 12 states dataset
#:     - v2: Full 31 states dataset
#:     - v2_1k: 31 states with 1000+ name occurrences (recommended)
#:     - v2_native: Native language scripts dataset (16 states)
#:     - v2_en: English transliteration of v2_native
IN_ROLLS_DATA = {
    "v1": "https://dataverse.harvard.edu/api/v1/access/datafile/4967581",
    "v2": "https://dataverse.harvard.edu/api/v1/access/datafile/4965696",
    "v2_1k": "https://dataverse.harvard.edu/api/v1/access/datafile/4965695",
    "v2_native": "https://dataverse.harvard.edu/api/v1/access/datafile/6292042",
    "v2_en": "https://dataverse.harvard.edu/api/v1/access/datafile/6457224",
    # v3: native first names re-romanized via the eroll corpora + v2's non-native states
    # (31 states, full coverage). Build: model_training/retransliterate_native.py. After
    # publishing naampy_v3.csv.gz, add its immutable URL here and default to it.
}

IN_ROLLS_COLS = [
    "n_male",
    "n_female",
    "n_third_gender",
    "prop_female",
    "prop_male",
    "prop_third_gender",
]

#: Every column :func:`in_rolls_fn_gender` may append. Dropped from the input before
#: merging so that re-running naampy on its own output stays idempotent.
OUTPUT_COLS = [*IN_ROLLS_COLS, "pred_gender", "pred_prob"]


def _require_dataset(dataset: str) -> None:
    """Reject unknown dataset keys before any input-dependent fast path.

    Args:
        dataset: Published electoral-roll dataset key.

    Raises:
        ValueError: If dataset is not a published electoral-roll table.
    """
    if dataset not in IN_ROLLS_DATA:
        choices = ", ".join(sorted(IN_ROLLS_DATA))
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}")


class InRollsFnData:
    """Main class for handling Indian Electoral Roll data and gender prediction.

    This class provides methods to predict gender based on Indian first names using
    two approaches:

    1. Statistical data from Indian Electoral Rolls (up to 31 states and UTs)
    2. Machine learning model for names not found in the electoral data

    The class maintains cached data and models for efficient repeated predictions.
    """

    __df: pd.DataFrame | None = None
    __cache_key: tuple[str, str | None, int | None] | None = None
    __model = None

    @staticmethod
    def load_naampy_data(dataset: str) -> str:
        """Download and cache the naampy dataset from Harvard Dataverse.

        This method downloads the specified dataset version if not already cached locally.
        Subsequent calls will use the cached version for faster performance.

        Args:
            dataset: Version of the dataset to load. Options are:
                - 'v1': 12 states dataset
                - 'v2': Full 31 states dataset
                - 'v2_1k': 31 states with 1000+ name occurrences (default)
                - 'v2_native': Native language dataset (16 states)
                - 'v2_en': English transliteration of v2_native

        Returns:
            str: Local path to the typed Parquet cache.

        Raises:
            ValueError: If dataset is unknown or its data violates the schema.
            RuntimeError: If the dataset download fails.

        Example:
            .. code-block:: python

                path = InRollsFnData.load_naampy_data("v2_1k")
                print(f"Data cached at: {path}")
        """  # noqa: DOC503
        _require_dataset(dataset)

        data_path = Path(get_app_file_path("naampy", f"naampy_{dataset}.parquet"))
        if data_path.exists():
            validate_parquet(data_path)
            LOGGER.info("Using cached naampy data from %s", data_path)
            return str(data_path)

        LOGGER.info("Downloading naampy dataset %s", dataset)
        with tempfile.TemporaryDirectory(dir=data_path.parent) as scratch:
            raw_path = Path(scratch) / f"naampy_{dataset}.csv.gz"
            if not download_file(IN_ROLLS_DATA[dataset], str(raw_path)):
                raise RuntimeError(f"Cannot download naampy dataset {dataset!r}")
            csv_to_parquet(raw_path, data_path)
        return str(data_path)

    @classmethod
    def predict_fn_gender(cls, first_names: list[str]) -> pd.DataFrame:
        """Predict gender using a neural network model based on character patterns in names.

        This method uses a character-level neural network trained on Indian names to predict
        gender when names are not found in the electoral roll database. The model learns
        patterns in character sequences to make predictions.

        Args:
            first_names: List of first names to predict gender for.
                Names are automatically converted to lowercase.

        Returns:
            pd.DataFrame: DataFrame containing:
                - name: Input first name (lowercased)
                - pred_gender: Predicted gender ('male' or 'female'), or None
                - pred_prob: Confidence score for the prediction (0.0 to 1.0), or NaN

        Note:
            - Names are classified as 'female' if predicted probability > 0.5
            - Names are classified as 'male' if predicted probability <= 0.5
            - The model reads a-z only. A name with no a-z characters (for example
              one written in Devanagari) has nothing to score, so it comes back as
              None / NaN rather than being assigned a gender at 0.5 confidence.

        Example:
            .. code-block:: python

                names = ["Priya", "Rahul", "हेमा"]
                result = InRollsFnData.predict_fn_gender(names)
                print(result)
        """
        import torch

        from .nnets import (
            LSTM_DROPOUT,
            LSTM_EMB,
            LSTM_HIDDEN,
            LSTM_LAYERS,
            VOCAB_SIZE,
            CharBiLSTM,
            encode_name,
            pad_encoded,
        )

        if len(first_names) == 0:
            return pd.DataFrame(
                {
                    "name": pd.Series(dtype="object"),
                    "pred_gender": pd.Series(dtype="object"),
                    "pred_prob": pd.Series(dtype="float64"),
                }
            )

        # Load the pinned char-BiLSTM once (lazy; cached on the class).
        if cls.__model is None:
            model = CharBiLSTM(
                VOCAB_SIZE, 1, LSTM_EMB, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT
            )
            model_path = resolve_model("gender_lstm.pt")
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            cls.__model = model

        names = [str(n).lower() for n in first_names]
        # Names with no in-vocab characters (e.g. pure Devanagari) are left as NaN:
        # the model has nothing to score, and a 0.5 default would otherwise be
        # rendered as a confident-looking "male".
        probs = np.full(len(names), np.nan)
        valid_rows: list[int] = []
        valid_enc: list[list[int]] = []
        for i, nm in enumerate(names):
            enc = encode_name(nm)
            if enc:
                valid_rows.append(i)
                valid_enc.append(enc)

        for s in range(0, len(valid_enc), 1024):
            rows = valid_rows[s : s + 1024]
            x, lengths = pad_encoded(valid_enc[s : s + 1024])
            with torch.no_grad():
                p = torch.sigmoid(cls.__model(x, lengths)).squeeze(1).tolist()
            for r, pv in zip(rows, p, strict=True):
                probs[r] = pv

        scored = ~np.isnan(probs)
        gender = np.full(len(names), None, dtype=object)
        gender[scored] = np.where(probs[scored] > 0.5, "female", "male")
        score = np.where(probs > 0.5, probs, 1 - probs)
        return pd.DataFrame(
            data={"name": names, "pred_gender": gender, "pred_prob": score}
        )

    @classmethod
    def in_rolls_fn_gender(
        cls,
        df: pd.DataFrame,
        namecol: str,
        state: str | None = None,
        year: int | None = None,
        dataset: str = "v2_1k",
    ) -> pd.DataFrame:
        """Predict gender from Indian first names using Electoral Roll statistics.

        This function enriches the input DataFrame with gender statistics from the Indian
        Electoral Rolls database. For names not found in the database, it automatically
        falls back to machine learning predictions (except for v2_native dataset).

        Args:
            df: Input DataFrame containing the first name column.
            namecol: Name of the column containing first names to analyze.
            state: Specific Indian state to use for analysis.
                Available states: andaman, andhra, arunachal, assam, bihar, chandigarh,
                dadra, daman, delhi, goa, gujarat, haryana, himachal, jharkhand, jk,
                karnataka, kerala, maharastra, manipur, meghalaya, mizoram, mp,
                nagaland, odisha, puducherry, punjab, rajasthan, sikkim, tripura,
                up, uttarakhand. Defaults to None (all states).
            year: Specific birth year to filter data by.
                Defaults to None (all years).
            dataset: Dataset version to use. Options:
                - 'v1': 12 states dataset
                - 'v2': Full 31 states dataset
                - 'v2_1k': 1000+ occurrences dataset (default, good balance)
                - 'v2_native': Native language dataset (no ML fallback)
                - 'v2_en': English transliteration dataset

        Returns:
            pd.DataFrame: A copy of the input with these columns appended:
                - n_female: Count of females with this name
                - n_male: Count of males with this name
                - n_third_gender: Count of third gender individuals
                - prop_female: Proportion female (0.0 to 1.0)
                - prop_male: Proportion male (0.0 to 1.0)
                - prop_third_gender: Proportion third gender (0.0 to 1.0)
                - pred_gender: ML prediction for names not in the database
                - pred_prob: ML prediction confidence score

            The two ML columns are always present unless dataset is 'v2_native',
            and hold None / NaN for rows that were resolved from the rolls.

        Raises:
            ValueError: If dataset, state, or year is not available.

        Note:
            - Names are matched after stripping and lowercasing; the input column
              itself is returned unchanged
            - The input DataFrame is not modified; a copy is returned
            - Rows with a missing or empty name stay NaN in every appended column
            - Data is cached after first download for faster subsequent use
            - Third gender category reflects Indian electoral roll classifications

        Example:
            .. code-block:: python

                import pandas as pd

                df = pd.DataFrame({"name": ["Priya", "Rahul", "Anjali"]})
                result = in_rolls_fn_gender(df, "name")
                print(result[["name", "prop_female", "prop_male"]].head())
        """
        _require_dataset(dataset)
        if not namecol or namecol not in df.columns:
            print(f"No column `{namecol}` in the DataFrame")
            return df.copy()

        # Work on a copy so the caller's frame is never mutated, and drop any
        # naampy columns left over from a previous run so re-running on our own
        # output is idempotent instead of producing `_x`/`_y` merge suffixes.
        output_cols = IN_ROLLS_COLS if dataset == "v2_native" else OUTPUT_COLS
        rdf = df.drop(columns=[c for c in output_cols if c in df.columns])
        first_name = rdf[namecol].astype("string").str.strip().str.lower()
        # Whitespace-only names strip to "", which is not a name; treat it as missing.
        rdf["__first_name"] = first_name.mask(first_name == "")

        if rdf.empty:
            for column in IN_ROLLS_COLS:
                dtype = "Int64" if column.startswith("n_") else "float64"
                rdf[column] = pd.Series(index=rdf.index, dtype=dtype)
            if dataset != "v2_native":
                rdf["pred_gender"] = pd.Series(index=rdf.index, dtype="object")
                rdf["pred_prob"] = pd.Series(index=rdf.index, dtype="float64")
            return rdf.drop(columns="__first_name")

        cache_key = (dataset, state, year)
        if cls.__df is None or cls.__cache_key != cache_key:
            data_path = InRollsFnData.load_naampy_data(dataset)
            adf = cast(
                "pd.DataFrame",
                pd.read_parquet(
                    data_path,
                    columns=[
                        "state",
                        "birth_year",
                        "first_name",
                        "n_female",
                        "n_male",
                        "n_third_gender",
                    ],
                ),
            )
            if state is not None and state not in set(adf["state"]):
                choices = ", ".join(sorted(adf["state"].unique()))
                raise ValueError(
                    f"State {state!r} is not in dataset {dataset!r}; "
                    f"choose one of: {choices}"
                )
            if state is not None:
                adf = adf[adf.state == state]
            if year is not None and year not in set(adf["birth_year"]):
                scope = f"state {state!r}" if state is not None else "all states"
                raise ValueError(
                    f"Birth year {year!r} is not available for {scope} "
                    f"in dataset {dataset!r}"
                )
            if year is not None:
                adf = adf[adf.birth_year == year]
            # Always collapse to one row per name: guarantees a unique merge key
            # regardless of how many (state, birth_year) rows survived the filter.
            table_for_aggregation: Any = adf
            adf = cast(
                "pd.DataFrame",
                table_for_aggregation.groupby("first_name", as_index=False).agg(
                    {
                        "n_female": "sum",
                        "n_male": "sum",
                        "n_third_gender": "sum",
                    }
                ),
            )
            total_count = cast(
                "Any",
                adf["n_female"] + adf["n_male"] + adf["n_third_gender"],
            ).replace(0, np.nan)
            adf["prop_female"] = adf["n_female"] / total_count
            adf["prop_male"] = adf["n_male"] / total_count
            adf["prop_third_gender"] = adf["n_third_gender"] / total_count
            selected_columns: Any = adf[["first_name", *IN_ROLLS_COLS]]
            adf = cast(
                "pd.DataFrame",
                selected_columns.rename(columns={"first_name": "__first_name"}),
            )
            adf["__first_name"] = adf["__first_name"].astype("string")
            cls.__df = adf
            cls.__cache_key = cache_key
        lookup_frame = cast("pd.DataFrame", cls.__df)
        lookup = lookup_frame.set_index("__first_name")
        for column in IN_ROLLS_COLS:
            lookup_values: Any = lookup[column]
            rdf[column] = rdf["__first_name"].map(lookup_values)
        for column in ("n_male", "n_female", "n_third_gender"):
            rdf[column] = rdf[column].astype("Int64")

        if dataset != "v2_native":
            # Declare the ML columns unconditionally so the output schema does not
            # depend on whether any name happened to miss the lookup.
            rdf["pred_gender"] = pd.Series(None, index=rdf.index, dtype="object")
            rdf["pred_prob"] = pd.Series(np.nan, index=rdf.index, dtype="float64")
            # Rows with no usable name are left as NaN rather than handed to the
            # model, which would otherwise score the literal string "nan".
            missing = rdf["prop_female"].isna() & rdf["__first_name"].notna()
            if missing.any():
                mdf = predict_fn_gender(rdf.loc[missing, "__first_name"].tolist())
                rdf.loc[missing, "pred_gender"] = mdf["pred_gender"].to_numpy()
                rdf.loc[missing, "pred_prob"] = mdf["pred_prob"].to_numpy()

        del rdf["__first_name"]

        return rdf

    @staticmethod
    def list_states(dataset: str = "v2_1k") -> np.ndarray:
        """Get list of available states in the specified dataset.

        This method returns all unique states/union territories available in the
        chosen dataset version for filtering and analysis.

        Args:
            dataset: Dataset version to query. Defaults to 'v2_1k'.
                See load_naampy_data() for available dataset options.

        Returns:
            np.ndarray: Array of state names available in the dataset.

        Example:
            .. code-block:: python

                states = InRollsFnData.list_states("v2_1k")
                print(f"Available states: {', '.join(states[:5])}...")
        """
        data_path = InRollsFnData.load_naampy_data(dataset)
        adf = pd.read_parquet(data_path, columns=["state"])
        return adf.state.unique()


in_rolls_fn_gender = InRollsFnData.in_rolls_fn_gender
predict_fn_gender = InRollsFnData.predict_fn_gender


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for naampy gender prediction.

    This function provides a command-line interface to process CSV files and
    add gender predictions based on first names using Indian Electoral Roll data.

    Args:
        argv: Command line arguments. Defaults to sys.argv[1:].

    Returns:
        int: Exit code (0 for success, 1 for error)

    Example:
        $ in_rolls_fn_gender input.csv -f first_name -o output.csv
        $ in_rolls_fn_gender input.csv -f name -s kerala -y 1990
    """
    if argv is None:
        argv = sys.argv[1:]

    title = (
        "Appends Electoral roll columns prop_female, n_female, n_male n_third_gender"
    )
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    parser.add_argument(
        "-f",
        "--first-name",
        required=True,
        help="Name of column containing the first name",
    )
    # Deliberately not an argparse `choices=`: that would download and parse the
    # dataset just to build the parser, so even `--help` would pay for it. The
    # state is validated after parsing, against the dataset the user actually asked for.
    parser.add_argument(
        "-s",
        "--state",
        default=None,
        help="State name of Indian electoral rolls data (default=all)",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=None,
        help="Birth year in Indian electoral rolls data (default=all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="in-rolls-output.csv",
        help="Output file with Indian electoral rolls data columns",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="v2_1k",
        choices=["v1", "v2", "v2_1k", "v2_native", "v2_en"],
        help="Select the dataset. v1 is 12 states,"
        + " v2 and v2_1k for 31 states with 100 and 1,000"
        + " first name occurrences respectively"
        + " v2_native is the native language dataset of"
        + " 16 states with 10 first name occurrences per state,"
        + " and v2_en is Hindi transliteration of v2_native dataset"
        + " (default=v2_1k)",
    )

    args = parser.parse_args(argv)

    print(args)

    if args.state is not None:
        valid_states = InRollsFnData.list_states(args.dataset)
        if args.state not in valid_states:
            print(
                f"State `{args.state}` not in dataset `{args.dataset}`. "
                f"Available: {', '.join(sorted(valid_states))}"
            )
            return 1

    df = pd.read_csv(args.input)

    if args.first_name not in df.columns:
        print(f"Column `{args.first_name}` not found in the input file")
        return 1

    rdf = in_rolls_fn_gender(df, args.first_name, args.state, args.year, args.dataset)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
