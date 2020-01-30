
.. image:: https://travis-ci.org/naampy/ethnicolr.svg?branch=master
    :target: https://travis-ci.org/naampy/naampy
.. image:: https://ci.appveyor.com/api/projects/status/qfvbu8h99ymtw2ub?svg=true
    :target: https://ci.appveyor.com/project/soodoku/naampy
.. image:: https://img.shields.io/pypi/v/naampy.svg
    :target: https://pypi.python.org/pypi/naampy

\

naampy: Infer Gender from Indian Names 
--------------------------------------------

A name reveals much, especially in India. And learning important social 
attributes of a person from their name can be useful for a broad set of tasks, 
from estimating to what extent women are covered in news compared to men, 
to estimating whether there is bias in lending against certain social groups. 

Unlike the American Census Bureau, which produces list of last names and first names, 
which can (and have been) used to infer the gender and race from names, India produces 
no such commensurate datasets. And learning the relationship between gender, ethnicity, 
language group, etc. and name has generally been done with small datasets constructed 
in an adhoc manner.

We fix this yawning gap in this paper, making three novel contributions. We first assemble 
a large novel dataset of Indian names. We scrape the electoral rolls that are public to build 
the first big comprehensive representative dataset of all adult Indians. We then we use it to 
build a variety of classifiers, learning relationships between names and gender, and language group. 
We show how biased the estimates can be from other smaller datasets. Lastly, we use the classifiers 
to estimate the coverage of women in major Indian newspapers.

Caveats and Notes
===================


Installation
--------------

::

    pip install naampy

**Note**: If you are installing the package on Windows, Theano installation typically needs admin. privileges. 

General API
----------------

To see the available command line options for any function, please type in 
``<function-name> --help``

::

   # _name --help
   usage: naampy_name [-h] [-y {2017}] [-o OUTPUT] -l LAST input

   Appends Census columns by last name

   positional arguments:
     input                 Input file

   optional arguments:
     -h, --help            show this help message and exit
     -y {2000,2010}, --year {2000,2010}
                           Year of Census data (default=2000)
     -o OUTPUT, --output OUTPUT
                           Output file with Census data columns
     -l LAST, --last LAST  Name or index location of column contains the last
                           name



Functions
----------

We expose 2 functions, each of which either take a pandas DataFrame or a CSV. If the CSV doesn't have a header,
we make some assumptions about where the data is

-  **bayes\_fn\_gender**

   -  Input: pandas DataFrame or CSV and a string or list of name or
      location of the column containing the first name.

   -  What it does:

      -  Removes extra space.
      -  For names in the `votereg file <https://github.com/appeler/naampy/tree/master/naampy/data/votereg>`__, it appends relevant data.

   -  Options:

      -  year: 2017
      -  if no year is given, data from the 2017 voting reg. files is appended

   -  Output: Appends the following columns to the pandas DataFrame or CSV::

        pctmale, pctfemale  


-  **pred\_fn\_gender**

   -  Input: pandas DataFrame or CSV and string or list containing the name or
      location of the column containing the first name, last name, middle
      name, and suffix, if there. The first name and last name columns are
      required. If no middle name of suffix columns are there, it is
      assumed that there are no middle names or suffixes.

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name wiki
         model <https://github.com/appeler/ethnicolr/tree/master/ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb>`__ to predict the
         race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV::

        race (categorical variable---category with the highest probability), 
        "Asian,GreaterEastAsian,EastAsian", "Asian,GreaterEastAsian,Japanese", 
        "Asian,IndianSubContinent", "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
        "GreaterEuropean,British", "GreaterEuropean,EastEuropean", 
        "GreaterEuropean,Jewish", "GreaterEuropean,WestEuropean,French",
        "GreaterEuropean,WestEuropean,Germanic", "GreaterEuropean,WestEuropean,Hispanic",
        "GreaterEuropean,WestEuropean,Italian", "GreaterEuropean,WestEuropean,Nordic"

Using ethnicolr
----------------

::

   >>> import pandas as pd

   >>> from naampy import bayes_fn_gender, pred_fn_gender
   Using TensorFlow backend.

   >>> names = [{'name': 'smita'},
   ...         {'name': 'ravi'},
   ...         {'name': 'amit'}]

   >>> df = pd.DataFrame(names)

   >>> df
         name
   0    smita
   1     ravi
   2     amit

   >>> bayes_fn_gender(df, 'name')

   >>> pred_fn_gender(df, 'name')

   >>> help(pred_fn_gender)

Examples
----------

Underlying Data
------------------

We capitalize on a novel `voting registration dataset <https://github.com/in-rolls/electoral_rolls/>`__

Authors
----------

Gaurav Sood and Atul Dhingra

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
