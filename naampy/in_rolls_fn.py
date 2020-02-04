#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd

from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns

IN_ROLLS_DATA = resource_filename(__name__, "data/in_rolls/in_rolls_state_year_fn_naampy.csv.gz")
IN_ROLLS_COLS = ['n_male', 'n_female', 'n_third_gender', 'prop_female']


class InRollsFnData():
    __df = None
    __state = None
    __year = None

    @classmethod
    def in_rolls_fn_gender(cls, df, namecol, state=None, year=None):
        """Appends additional columns from Female ratio data to the input DataFrame
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
                'prop_female', 'n_female', 'n_male', 'n_third_gender' by first name

        """

        if namecol not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(namecol))
            return df

        df['__first_name'] = df[namecol].str.strip()
        df['__first_name'] = df['__first_name'].str.lower()

        if cls.__df is None or cls.__state != state or cls.__year != year:
            adf = pd.read_csv(IN_ROLLS_DATA, usecols=['state', 'birth_year',
                              'first_name'] + IN_ROLLS_COLS)
            agg_dict = {'n_female': 'sum', 'n_male': 'sum', 'n_third_gender': 'sum'}
            if state and year:
                adf = adf[(adf.state==state) & (adf.birth_year==year)].copy()
                del adf['birth_year']
                del adf['state']
            elif state:
                adf = adf.groupby(['state', 'first_name']).agg(agg_dict).reset_index()
                adf['prop_female'] = adf['n_female'] / (adf['n_female'] + adf['n_male'] + adf['n_third_gender'])
                adf = adf[adf.state==state].copy()
                del adf['state']
            elif year:
                adf = adf.groupby(['birth_year', 'first_name']).agg(agg_dict).reset_index()
                adf['prop_female'] = adf['n_female'] / (adf['n_female'] + adf['n_male'] + adf['n_third_gender'])
                adf = adf[adf.birth_year==year].copy()
                del adf['birth_year']
            else:
                adf = adf.groupby(['first_name']).agg(agg_dict).reset_index()
                adf['prop_female'] = adf['n_female'] / (adf['n_female'] + adf['n_male'] + adf['n_third_gender'])
            cls.__df = adf
            cls.__df = cls.__df[['first_name'] + IN_ROLLS_COLS]
            cls.__df.rename(columns={'first_name': '__first_name'}, inplace=True)
        rdf = pd.merge(df, cls.__df, how='left', on='__first_name')

        del rdf['__first_name']

        return rdf

    @staticmethod
    def list_states():
        adf = pd.read_csv(IN_ROLLS_DATA, usecols=['state'])
        return adf.state.unique()


in_rolls_fn_gender = InRollsFnData.in_rolls_fn_gender


def main(argv=sys.argv[1:]):
    title = ('Appends Electoral roll columns for prop_female, n_female, '
            'n_male n_third_gender by first name')
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-f', '--first-name', required=True,
                        help='Name or index location of column contains '
                             'the first name')
    parser.add_argument('-s', '--state', default=None,
                        choices=InRollsFnData.list_states(),
                        help='State name of Indian electoral rolls data '
                             '(default=all)')
    parser.add_argument('-y', '--year', type=int, default=None,
                        help='Birth year in Indian electoral rolls data (default=all)')
    parser.add_argument('-o', '--output', default='in-rolls-output.csv',
                        help='Output file with Indian electoral rolls data columns')

    args = parser.parse_args(argv)

    print(args)

    if not args.first_name.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.first_name = int(args.first_name)

    if not column_exists(df, args.first_name):
        return -1

    rdf = in_rolls_fn_gender(df, args.first_name, args.state, args.year)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
