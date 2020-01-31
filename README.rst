naampy: Infer Sociodemographic Characteristics from Indian Names
----------------------------------------------------------------

.. image:: https://travis-ci.org/appeler/naampy.svg?branch=master
    :target: https://travis-ci.org/appeler/naampy
.. image:: https://ci.appveyor.com/api/projects/status/q4wr4clilf4samlk?svg=true
    :target: https://ci.appveyor.com/project/soodoku/naampy
.. image:: https://img.shields.io/pypi/v/naampy.svg
    :target: https://pypi.python.org/pypi/naampy
.. image https://pepy.tech/badge/naampy
..    :target: https://pepy.tech/project/naampy


The ability to programmatically reliably infer social attributes of a person from their name can be useful for a broad set of tasks, from estimating bias in coverage of women in the media to estimating bias in lending against certain social groups. But unlike the American Census Bureau, which produces a list of last names and first names, which can (and are) used to infer the gender, race, ethnicity, etc. from names, the Indian government produces no such commensurate datasets. And hence inferring the relationship between gender, ethnicity, language group, etc. and names has generally been done with small datasets constructed in an ad-hoc manner.

We fill this yawning gap. Using data from the `Indian Electoral Rolls <https://github.com/in-rolls/electoral_rolls>`__ (parsed data `here <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MUEGDT>`__), we estimate the proportion female, male, and `third sex` (see `here <https://en.wikipedia.org/wiki/Third_gender>`__) for a particular `first name, year, and state.`

Data
~~~~

In all, we capitalize on information in the parsed electoral rolls from the following 12 states and union territories: 

* Andaman
* Andhra Pradesh
* Dadra
* Daman
* Goa
* Jammu and Kashmir
* Manipur
* Meghalaya
* Mizoram
* Nagaland
* Puducherry

How is the underlying data produced?
====================================

We split the name into first name and last name (see the python notebook for how we do this) and then aggregate per state and first_name, and tabulate `prop_male, prop_female, prop_third_gender, n_female, n_male, n_third_gender`

This is used to provide the base prediction.

Given the association between prop_female and first_name may change over time, we exploited the age. Given the data were collected in 2017, we calculate the year each person was born and then do a group by year to create `prop_male, prop_female, prop_third_gender, n_female, n_male, n_third_gender`

Issues with underlying data
==============================

Concerns:

* Voting registration lists may not be accurate, systematically underrepresenting the poor, minorities, etc.
* Voting registrations lists at best reflect the adult citizens. But to the extent that prejudice against women, etc., prevents some kinds of people to reach adulthood, the data bakes those biased in.
* Indian names are complicated. We do not have good parsers for them yet. We have gone for the default arrangement. Please go through the notebook to look at the judgments we make. We plan to improve the underlying data over time.

Gender Classifier
~~~~~~~~~~~~~~~~~

We start by providing a base model for first\_name that gives the Bayes
optimal solution providing the proportion of women with that name who
are women. We also provide a series of base models where the state of
residence is known. In the future, we plan to use LSTM to learn the relationship between
sequences of characters in the first name and gender.

Installation
~~~~~~~~~~~~~~

We strongly recommend installing `naampy` inside a Python virtual environment (see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install naampy


Usage
~~~~~

::

  usage: in_rolls_fn_gender [-h] -f FIRST_NAME [-s STATE] [-y YEAR] [-o OUTPUT]
                            input

  Appends Electoral roll columns for prop_female, n_female, n_male
  n_third_gender by first name

  positional arguments:
    input                 Input file

  optional arguments:
    -h, --help            show this help message and exit
    -f FIRST_NAME, --first-name FIRST_NAME
                          Name or index location of column contains the first
                          name
    -s STATE, --state STATE
                          State name of Indian electoral rolls data
                          (default=all)
    -y YEAR, --year YEAR  Birth year in Indian electoral rolls data
                          (default=all)
    -o OUTPUT, --output OUTPUT
                          Output file with Indian electoral rolls data columns

Using naampy
~~~~~~~~~~~~

::

  >>> import pandas as pd
  >>> from naampy import in_rolls_fn_gender

  >>> names = [{'name': 'yoga'},
  ...          {'name': 'yasmin'},
  ...          {'name': 'siri'},
  ...          {'name': 'vivek'}]

  >>> df = pd.DataFrame(names)

  >>> in_rolls_fn_gender(df, 'name')
      name  n_male  n_female  n_third_gender  prop_female
  0    yoga     202       150               0     0.426136
  1  yasmin      24      2635               0     0.990974
  2    siri     115       556               0     0.828614
  3   vivek    2252        13               0     0.005740

  >>> help(in_rolls_fn_gender)
  Help on method in_rolls_fn_gender in module naampy.in_rolls_fn:

  in_rolls_fn_gender(df, namecol, state=None, year=None) method of builtins.type instance
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
              'prop_female', 'n_female', 'n_male', 'n_third_gender' by first name


Authors
~~~~~~~

Suriyan Laohaprapanon and Gaurav Sood

License
~~~~~~~

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
