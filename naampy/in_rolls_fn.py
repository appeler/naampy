#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys
import os
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pkg_resources import resource_filename

from .utils import get_app_file_path, download_file


IN_ROLLS_DATA = {
    "v1": "https://dataverse.harvard.edu/api/v1/access/datafile/4967581",
    "v2": "https://dataverse.harvard.edu/api/v1/access/datafile/4965696",
    "v2_1k": "https://dataverse.harvard.edu/api/v1/access/datafile/4965695",
    "v2_native": "https://dataverse.harvard.edu/api/v1/access/datafile/6292042",
    "v2_en": "https://dataverse.harvard.edu/api/v1/access/datafile/6457224",
}

IN_ROLLS_COLS = ["n_male", "n_female", "n_third_gender", "prop_female", "prop_male", "prop_third_gender"]


class InRollsFnData:
    """
    InRollsFnData class.
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

        Args:
            dataset (str): version of the dataset
        Returns:
            path to the data
        """
        data_fn = f"naampy_{dataset}.csv.gz"
        data_path = get_app_file_path("naampy", data_fn)
        if not os.path.exists(data_path):
            print(f"Downloading naampy data from the server ({data_fn})...")
            if not download_file(IN_ROLLS_DATA[dataset], data_path):
                print("ERROR: Cannot download naampy data file")
                return None
        else:
            print(f"Using cached naampy data from local ({data_path})...")
        return data_path

    @classmethod
    def predict_fn_gender(cls, first_names: list[str]) -> pd.DataFrame:
        """
        Predict gender based on name
        Args:
            first_names (list of str): list of first name
        Returns:
            DataFrame: Pandas DataFrame with predicted labels and probability
        """
        # load model
        if cls.__model is None:
            model_fn = resource_filename(__name__, "model")
            cls.__model = tf.keras.models.load_model(f"{model_fn}/naampy_rmse")
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

        first_names = [i.lower() for i in first_names]
        sequences = cls.__tk.texts_to_sequences(first_names)
        tokens = pad_sequences(sequences, maxlen=24, padding="post")

        results = cls.__model.predict(tokens)
        gender = np.where(results > 0.5, "female", "male")
        score = np.where(results > 0.5, results, 1 - results)

        return pd.DataFrame(data={"name": first_names, "pred_gender": gender, "pred_prob": score})

    @classmethod
    def in_rolls_fn_gender(cls, df: pd.DataFrame, namecol: str, state: str=None, year: int=None, dataset: str="v2_1k") -> pd.DataFrame:
        """
        Appends additional columns from Female ratio data to the input DataFrame
        based on the first name.

        Removes extra space. Checks if the name is the Indian electoral rolls data.
        If it is, outputs data from that row.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first name
                column.
            namecol (str or int): Column's name or location of the name in
                DataFrame.
            state (str): The state name of Indian electoral rolls data to be used.
                (default is None for all states)
            year (int): The year of Indian electoral rolls to be used.
                (default is None for all years)

        Returns:
            DataFrame: Pandas DataFrame with additional columns:-
                'n_female', 'n_male', 'n_third_gender',
                'prop_female', 'prop_male', 'prop_third_gender' by first name

        """

        if namecol not in df.columns:
            print(f"No column `{namecol}` in the DataFrame")
            return df

        df["__first_name"] = df[namecol].str.strip()
        df["__first_name"] = df["__first_name"].str.lower()

        if cls.__df is None or cls.__state != state or cls.__year != year or cls.__dataset != dataset:
            cls.__dataset = dataset
            data_path = InRollsFnData.load_naampy_data(dataset)
            adf = pd.read_csv(
                data_path, usecols=["state", "birth_year", "first_name", "n_female", "n_male", "n_third_gender"]
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
                adf = adf.groupby(["birth_year", "first_name"]).agg(agg_dict).reset_index()
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
                mdf = predict_fn_gender(not_in_db_names["__first_name"].values)
                rdf.at[not_in_db_names.index, "pred_gender"] = mdf["pred_gender"].values
                rdf.at[not_in_db_names.index, "pred_prob"] = mdf["pred_prob"].values

        del rdf["__first_name"]

        return rdf

    @staticmethod
    def list_states(dataset: str="v2_1k") -> np.ndarray:
        """

        Args:
            dataset (str): version of the dataset
        Returns:
            list of states
        """
        data_path = InRollsFnData.load_naampy_data(dataset)
        adf = pd.read_csv(data_path, usecols=["state"])
        return adf.state.unique()


in_rolls_fn_gender = InRollsFnData.in_rolls_fn_gender
predict_fn_gender = InRollsFnData.predict_fn_gender


def main(argv=sys.argv[1:]):
    """
    Main method for shell support
    """
    title = "Appends Electoral roll columns for prop_female, n_female, n_male n_third_gender by first name"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    parser.add_argument(
        "-f", "--first-name", required=True, help="Name or index location of column contains " "the first name"
    )
    parser.add_argument(
        "-s",
        "--state",
        default=None,
        choices=InRollsFnData.list_states(),
        help="State name of Indian electoral rolls data " "(default=all)",
    )
    parser.add_argument(
        "-y", "--year", type=int, default=None, help="Birth year in Indian electoral rolls data (default=all)"
    )
    parser.add_argument(
        "-o", "--output", default="in-rolls-output.csv", help="Output file with Indian electoral rolls data columns"
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
        + " (default=v2_1k)"
    )

    args = parser.parse_args(argv)

    print(args)

    df = pd.read_csv(args.input)
    
    if col and (col not in df.columns):
        print(f"Column `{col}` not found in the input file")
        return -1

    rdf = in_rolls_fn_gender(df, args.first_name, args.state, args.year, args.dataset)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())
