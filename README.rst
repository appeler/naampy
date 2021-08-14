naampy: Infer Sociodemographic Characteristics from Indian Names
----------------------------------------------------------------

.. image:: https://travis-ci.org/appeler/naampy.svg?branch=master
    :target: https://travis-ci.org/appeler/naampy
.. image:: https://ci.appveyor.com/api/projects/status/q4wr4clilf4samlk?svg=true
    :target: https://ci.appveyor.com/project/soodoku/naampy
.. image:: https://img.shields.io/pypi/v/naampy.svg
    :target: https://pypi.python.org/pypi/naampy
.. image:: https://pepy.tech/badge/naampy
    :target: https://pepy.tech/project/naampy


The ability to programmatically reliably infer social attributes of a person from their name can be useful for a broad set of tasks, from estimating bias in coverage of women in the media to estimating bias in lending against certain social groups. But unlike the American Census Bureau, which produces a list of last names and first names, which can (and are) used to infer the gender, race, ethnicity, etc. from names, the Indian government produces no such commensurate datasets. And hence inferring the relationship between gender, ethnicity, language group, etc. and names has generally been done with small datasets constructed in an ad-hoc manner.

We fill this yawning gap. Using data from the `Indian Electoral Rolls <https://github.com/in-rolls/electoral_rolls>`__ (parsed data `here <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MUEGDT>`__), we estimate the proportion female, male, and `third sex` (see `here <https://en.wikipedia.org/wiki/Third_gender>`__) for a particular `first name, year, and state.`

Data
~~~~

In all, we capitalize on information in the parsed electoral rolls from the following 31 states and union territories: 

.. list-table:: States
   :widths: 30 30 30 30
    
   * - Andaman
     - Delhi
     - Kerala
     - Puducherry
   
   *  - Andhra Pradesh
      - Goa
      - Madhya Pradesh
      - Punjab
   *  - Arunachal Pradesh
      - Gujarat
      - Maharashtra
      - Rajasthan 
   *  - Assam
      - Haryana
      - Manipur
      - Sikkim
   *  - Bihar
      - Himachal Pradesh
      - Meghalaya
      - Tripura 
   *  - Chandigarh
      - Jammu and Kashmir
      - Mizoram
      - Uttar Pradesh   
   *   - Dadra
       - Jharkhand
       - Nagaland
       - Uttarakhand
   *  -  Daman
      - Karnataka
      - Odisha
      - 
  

How is the underlying data produced?
====================================

We split the name into first name and last name (see the python notebook for how we do this) and then aggregate per state and first_name, and tabulate `prop_male, prop_female, prop_third_gender, n_female, n_male, n_third_gender`

This is used to provide the base prediction.

Given the association between prop_female and first_name may change over time, we exploited the age. Given the data were collected in 2017, we calculate the year each person was born and then do a group by year to create `prop_male, prop_female, prop_third_gender, n_female, n_male, n_third_gender`

Issues with underlying data
==============================

Concerns:

* Voting registration lists may not be accurate, systematically underrepresenting poor people, minorities, and similar such groups.

* Voting registration lists are at best a census of adult citizens. But to the extent there is prejudice against women, etc., that prevents them from reaching adulthood, the data bakes those biases in.

* Indian names are complicated. We do not have good parsers for them yet. We have gone for the default arrangement. Please go through the notebook to look at the judgments we make. We plan to improve the underlying data over time.

* For states with non-English rolls, we use libindic to transliterate the names. The transliterations are consistently bad. (We hope to make progress here. We also plan to provide a way to match in the original script.)

Gender Classifier
~~~~~~~~~~~~~~~~~

We start by providing a base model for first\_name that gives the Bayes
optimal solution providing the proportion of people with that name who
are women. We also provide a series of base models where the state of
residence and year of birth is known.

In the future, we plan to provide ML models that use the relationship between
sequences of characters in the first name and gender to predict gender from a name.

Installation
~~~~~~~~~~~~~~

We strongly recommend installing `naampy` inside a Python virtual environment (see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install naampy


Usage
~~~~~

::

    usage: in_rolls_fn_gender [-h] -f FIRST_NAME
                            [-s {andaman,andhra,arunachal,assam,bihar,chandigarh,dadra,daman,delhi,goa,gujarat,haryana,himachal,jharkhand,jk,karnataka,kerala,maharashtra,manipur,meghalaya,mizoram,mp,nagaland,odisha,puducherry,punjab,rajasthan,sikkim,tripura,up,uttarakhand}]
                            [-y YEAR] [-o OUTPUT]
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
    -s {andaman,andhra,arunachal,assam,bihar,chandigarh,dadra,daman,delhi,goa,gujarat,haryana,himachal,jharkhand,jk,karnataka,kerala,maharashtra,manipur,meghalaya,mizoram,mp,nagaland,odisha,puducherry,punjab,rajasthan,sikkim,tripura,up,uttarakhand},
    --state {andaman,andhra,arunachal,assam,bihar,chandigarh,dadra,daman,delhi,goa,gujarat,haryana,himachal,jharkhand,jk,karnataka,kerala,maharashtra,manipur,meghalaya,mizoram,mp,nagaland,odisha,puducherry,punjab,rajasthan,sikkim,tripura,up,uttarakhand}
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

    >>> names = [{'name': 'gaurav'},
    ...          {'name': 'yasmin'},
    ...          {'name': 'deepti'},
    ...          {'name': 'vivek'}]

    >>> df = pd.DataFrame(names)

    >>> in_rolls_fn_gender(df, 'name')
            name    n_male  n_female    n_third_gender  prop_female prop_male   prop_third_gender
        0   gaurav  25625   47  0   0.001831    0.998169    0.0
        1   yasmin  58  6079    0   0.990549    0.009451    0.0
        2   deepti  35  5784    0   0.993985    0.006015    0.0
        3   vivek   233622  1655    0   0.007034    0.992966    0.0
    
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
                'n_female', 'n_male', 'n_third_gender',
                'prop_female', 'prop_male', 'prop_third_gender' by first name

Functionality
~~~~~~~~~~~~~

When you first run `in_rolls_fn_gender`, it downloads data from `Harvard Dataverse <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WZGJBM>`__ to the local folder. Next time you run the function, it searches for local data and if it finds it, it uses it.

Authors
~~~~~~~

Suriyan Laohaprapanon and Gaurav Sood

License
~~~~~~~

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
