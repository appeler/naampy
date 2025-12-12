#!/usr/bin/env python

import argparse
import os
import sys
from importlib import resources

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .utils import download_file, get_app_file_path

#: Harvard Dataverse URLs for Indian Electoral Roll datasets.
#:
#: Contains download URLs for different versions of the naampy gender prediction
#: datasets hosted on Harvard Dataverse. Each version contains electoral roll
#: statistics from different numbers of Indian states and territories.
#:
#: Dataset versions:
#:     - v1: 12 states dataset
#:     - v2: Full 30 states dataset
#:     - v2_1k: 30 states with 1000+ name occurrences (recommended)
#:     - v2_native: Native language scripts dataset (16 states)
#:     - v2_en: English transliteration of v2_native
IN_ROLLS_DATA = {
    "v1": "https://dataverse.harvard.edu/api/v1/access/datafile/4967581",
    "v2": "https://dataverse.harvard.edu/api/v1/access/datafile/4965696",
    "v2_1k": "https://dataverse.harvard.edu/api/v1/access/datafile/4965695",
    "v2_native": "https://dataverse.harvard.edu/api/v1/access/datafile/6292042",
    "v2_en": "https://dataverse.harvard.edu/api/v1/access/datafile/6457224",
}

IN_ROLLS_COLS = [
    "n_male",
    "n_female",
    "n_third_gender",
    "prop_female",
    "prop_male",
    "prop_third_gender",
]


class InRollsFnData:
    """
    Main class for handling Indian Electoral Roll data and gender prediction.

    This class provides methods to predict gender based on Indian first names using
    two approaches:
    1. Statistical data from Indian Electoral Rolls (31 states and union territories)
    2. Machine learning model for names not found in the electoral data

    The class maintains cached data and models for efficient repeated predictions.
    """

    __df = None
    __state = None
    __year = None
    __model = None
    __tk = None
    __dataset = None

    @staticmethod
    def load_naampy_data(dataset: str) -> str:
        """
        Download and cache the naampy dataset from Harvard Dataverse.

        This method downloads the specified dataset version if not already cached locally.
        Subsequent calls will use the cached version for faster performance.

        Args:
            dataset (str): Version of the dataset to load. Options are:
                - 'v1': 12 states dataset
                - 'v2': Full 30 states dataset
                - 'v2_1k': 30 states with 1000+ name occurrences (default)
                - 'v2_native': Native language dataset (16 states)
                - 'v2_en': English transliteration of v2_native

        Returns:
            str: Local file path to the downloaded/cached dataset

        Raises:
            Exception: If the dataset download fails

        Example:
            >>> path = InRollsFnData.load_naampy_data('v2_1k')
            >>> print(f"Data cached at: {path}")
        """
        data_fn = f"naampy_{dataset}.csv.gz"
        data_path = get_app_file_path("naampy", data_fn)
        if not os.path.exists(data_path):
            print(f"Downloading naampy data from the server ({data_fn})...")
            if not download_file(IN_ROLLS_DATA[dataset], data_path):
                raise Exception("ERROR: Cannot download naampy data file")
        else:
            print(f"Using cached naampy data from local ({data_path})...")
        return data_path

    @classmethod
    def predict_fn_gender(cls, first_names: list[str]) -> pd.DataFrame:
        """
        Predict gender using a neural network model based on character patterns in names.

        This method uses a character-level neural network trained on Indian names to predict
        gender when names are not found in the electoral roll database. The model learns
        patterns in character sequences to make predictions.

        Args:
            first_names (list[str]): List of first names to predict gender for.
                Names are automatically converted to lowercase.

        Returns:
            pd.DataFrame: DataFrame containing:
                - name (str): Input first name (lowercased)
                - pred_gender (str): Predicted gender ('male' or 'female')
                - pred_prob (float): Confidence score for the prediction (0.0 to 1.0)

        Note:
            - Names are classified as 'female' if predicted probability > 0.5
            - Names are classified as 'male' if predicted probability ≤ 0.5
            - The model handles character sequences up to 24 characters
            - Model accuracy: RMSE of 0.22 on test data

        Example:
            >>> names = ['Priya', 'Rahul', 'Unknown_Name']
            >>> result = InRollsFnData.predict_fn_gender(names)
            >>> print(result)
                  name pred_gender  pred_prob
            0    priya      female      0.945
            1    rahul        male      0.876
            2  unknown_name  female      0.623
        """
        # load model
        if cls.__model is None:
            model_path = resources.files(__package__) / "model" / "naampy_rmse"
            # Use TFSMLayer for Keras 3 compatibility with SavedModel format
            try:
                cls.__model = tf.keras.models.load_model(str(model_path))
            except ValueError:
                # Fallback for Keras 3 with SavedModel format
                tfsm_layer = tf.keras.layers.TFSMLayer(
                    str(model_path), call_endpoint="serving_default"
                )
                # Create a functional model wrapper
                inputs = tf.keras.layers.Input(shape=(24,), dtype="int64", name="input")
                outputs = tfsm_layer(inputs)
                cls.__model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        # create tokenizer
        if cls.__tk is None:
            cls.__tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            char_dict = {}
            for i, char in enumerate(alphabet):
                char_dict[char] = i + 1
            # Use char_dict to replace the tk.word_index
            cls.__tk.word_index = char_dict.copy()
            # Add 'UNK' to the vocabulary
            cls.__tk.word_index[cls.__tk.oov_token] = max(char_dict.values()) + 1

        # Handle empty input - support both lists and numpy arrays
        if len(first_names) == 0:
            return pd.DataFrame(columns=["name", "pred_gender", "pred_prob"])

        first_names = [i.lower() for i in first_names]
        sequences = cls.__tk.texts_to_sequences(first_names)
        tokens = pad_sequences(sequences, maxlen=24, padding="post")

        results = cls.__model.predict(tokens)

        # Handle both old format (direct array) and new TFSMLayer format (dictionary)
        if isinstance(results, dict):
            # TFSMLayer returns a dictionary, extract the output tensor
            output_key = list(results.keys())[
                0
            ]  # Get the first (and likely only) output
            results = results[output_key]

        gender = np.where(results > 0.5, "female", "male")[:, 0]
        score = np.where(results > 0.5, results, 1 - results)[:, 0]

        return pd.DataFrame(
            data={"name": first_names, "pred_gender": gender, "pred_prob": score}
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
        """
        Predict gender from Indian first names using Electoral Roll statistics.

        This function enriches the input DataFrame with gender statistics from the Indian
        Electoral Rolls database. For names not found in the database, it automatically
        falls back to machine learning predictions (except for v2_native dataset).

        Args:
            df (pd.DataFrame): Input DataFrame containing the first name column.
            namecol (str): Name of the column containing first names to analyze.
            state (str, optional): Specific Indian state to use for analysis.
                Available states: andaman, andhra, arunachal, assam, bihar, chandigarh,
                dadra, daman, delhi, goa, gujarat, haryana, himachal, jharkhand, jk,
                karnataka, kerala, maharashtra, manipur, meghalaya, mizoram, mp,
                nagaland, odisha, puducherry, punjab, rajasthan, sikkim, tripura,
                up, uttarakhand. Defaults to None (all states).
            year (int, optional): Specific birth year to filter data by.
                Defaults to None (all years).
            dataset (str, optional): Dataset version to use. Options:
                - 'v1': 12 states dataset
                - 'v2': Full 30 states dataset
                - 'v2_1k': 1000+ occurrences dataset (default, good balance)
                - 'v2_native': Native language dataset (no ML fallback)
                - 'v2_en': English transliteration dataset

        Returns:
            pd.DataFrame: Enhanced DataFrame with additional columns:
                - n_female (float): Count of females with this name
                - n_male (float): Count of males with this name
                - n_third_gender (float): Count of third gender individuals
                - prop_female (float): Proportion female (0.0 to 1.0)
                - prop_male (float): Proportion male (0.0 to 1.0)
                - prop_third_gender (float): Proportion third gender (0.0 to 1.0)
                - pred_gender (str): ML prediction for names not in database
                - pred_prob (float): ML prediction confidence score

        Note:
            - Names are automatically cleaned (stripped and lowercased)
            - For names not in electoral data, ML predictions are added
            - Data is cached after first download for faster subsequent use
            - Third gender category reflects Indian electoral roll classifications

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name': ['Priya', 'Rahul', 'Anjali']})
            >>> result = in_rolls_fn_gender(df, 'name')
            >>> print(result[['name', 'prop_female', 'prop_male']].head())
                 name  prop_female  prop_male
            0   priya       0.994      0.006
            1   rahul       0.008      0.992
            2  anjali       0.989      0.011
        """

        if namecol and namecol not in df.columns:
            print(f"No column `{namecol}` in the DataFrame")
            return df

        df["__first_name"] = df[namecol].str.strip().str.lower()

        if (
            cls.__df is None
            or cls.__state != state
            or cls.__year != year
            or cls.__dataset != dataset
        ):
            cls.__dataset = dataset
            data_path = InRollsFnData.load_naampy_data(dataset)
            adf = pd.read_csv(
                data_path,
                usecols=[
                    "state",
                    "birth_year",
                    "first_name",
                    "n_female",
                    "n_male",
                    "n_third_gender",
                ],
            )
            agg_dict = {"n_female": "sum", "n_male": "sum", "n_third_gender": "sum"}
            if state and year:
                adf = adf[(adf.state == state) & (adf.birth_year == year)].copy()
                del adf["birth_year"]
                del adf["state"]
            elif state:
                adf = adf.groupby(["state", "first_name"]).agg(agg_dict).reset_index()
                adf = adf[adf.state == state].copy()
                del adf["state"]
            elif year:
                adf = (
                    adf.groupby(["birth_year", "first_name"])
                    .agg(agg_dict)
                    .reset_index()
                )
                adf = adf[adf.birth_year == year].copy()
                del adf["birth_year"]
            else:
                adf = adf.groupby(["first_name"]).agg(agg_dict).reset_index()
            n = adf["n_female"] + adf["n_male"] + adf["n_third_gender"]
            adf["prop_female"] = adf["n_female"] / n
            adf["prop_male"] = adf["n_male"] / n
            adf["prop_third_gender"] = adf["n_third_gender"] / n
            cls.__df = adf
            cls.__df = cls.__df[["first_name"] + IN_ROLLS_COLS]
            cls.__df.rename(columns={"first_name": "__first_name"}, inplace=True)
        rdf = pd.merge(df, cls.__df, how="left", on="__first_name")

        if dataset != "v2_native":
            # if name does not exist in database
            not_in_db_names = rdf[rdf["prop_female"].isna()]
            if len(not_in_db_names.values) > 0:
                mdf = predict_fn_gender(not_in_db_names["__first_name"].tolist())
                rdf.at[not_in_db_names.index, "pred_gender"] = mdf["pred_gender"].values
                rdf.at[not_in_db_names.index, "pred_prob"] = mdf["pred_prob"].values

        del rdf["__first_name"]

        return rdf

    @staticmethod
    def list_states(dataset: str = "v2_1k") -> np.ndarray:
        """
        Get list of available states in the specified dataset.

        This method returns all unique states/union territories available in the
        chosen dataset version for filtering and analysis.

        Args:
            dataset (str, optional): Dataset version to query. Defaults to 'v2_1k'.
                See load_naampy_data() for available dataset options.

        Returns:
            np.ndarray: Array of state names available in the dataset.

        Example:
            >>> states = InRollsFnData.list_states('v2_1k')
            >>> print(f"Available states: {', '.join(states[:5])}...")
            Available states: andaman, andhra, arunachal, assam, bihar...
        """
        data_path = InRollsFnData.load_naampy_data(dataset)
        adf = pd.read_csv(data_path, usecols=["state"])
        return adf.state.unique()


in_rolls_fn_gender = InRollsFnData.in_rolls_fn_gender
predict_fn_gender = InRollsFnData.predict_fn_gender


def main(argv=sys.argv[1:]):
    """
    Command-line interface for naampy gender prediction.

    This function provides a command-line interface to process CSV files and
    add gender predictions based on first names using Indian Electoral Roll data.

    Args:
        argv (list[str], optional): Command line arguments.
            Defaults to sys.argv[1:].

    Returns:
        int: Exit code (0 for success, -1 for error)

    Example:
        $ in_rolls_fn_gender input.csv -f first_name -o output.csv
        $ in_rolls_fn_gender input.csv -f name -s kerala -y 1990
    """
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
    parser.add_argument(
        "-s",
        "--state",
        default=None,
        choices=InRollsFnData.list_states(),
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
        + " v2 and v2_1k for 30 states with 100 and 1,000"
        + " first name occurrences respectively"
        + " v2_native is the native language dataset of"
        + " 16 states with 10 first name occurrences per state,"
        + " and v2_en is Hindi transliteration of v2_native dataset"
        + " (default=v2_1k)",
    )

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)

    if args.first_name and (args.first_name not in df.columns):
        print(f"Column `{args.first_name}` not found in the input file")
        return -1

    rdf = in_rolls_fn_gender(df, args.first_name, args.state, args.year, args.dataset)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
