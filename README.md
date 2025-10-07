# naampy: Infer Sociodemographic Characteristics from Indian Names

[![image](https://github.com/appeler/naampy/actions/workflows/test.yml/badge.svg)](https://github.com/appeler/naampy/actions/workflows/test.yml)
[![Documentation](https://github.com/appeler/naampy/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/naampy/actions/workflows/docs.yml)
[![image](https://img.shields.io/pypi/v/naampy.svg)](https://pypi.python.org/pypi/naampy)
[![image](https://static.pepy.tech/badge/naampy)](https://pepy.tech/project/naampy)

The ability to programmatically and reliably infer the social attributes
of a person from their name can be useful for a broad set of tasks, from
estimating bias in coverage of women in the media to estimating bias in
lending against certain social groups. But unlike the American Census
Bureau, which produces a list of last names and first names, which can
(and are) used to infer the gender, race, ethnicity, etc., from names,
the Indian government produces no such commensurate datasets. Hence
inferring the relationship between gender, ethnicity, language group,
etc., and names has generally been done with small datasets constructed
in an ad-hoc manner.

We fill this yawning gap. Using data from the [Indian Electoral
Rolls](https://github.com/in-rolls/electoral_rolls) (parsed data
[here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MUEGDT)),
we estimate the proportion female, male, and [third sex]{.title-ref}
(see [here](https://en.wikipedia.org/wiki/Third_gender)) for a
particular [first name, year, and state.]{.title-ref}

Please also check out [pranaam](https://github.com/appeler/pranaam) that
uses land record data from Bihar to infer religion based on the name.
The package uses [indicate](https://github.com/in-rolls/indicate) to
transliterate Hindi to English.

### Streamlit App

<https://naampy.streamlit.app/>

## Data

In all, we capitalize on information in the parsed electoral rolls from
the following 31 states and union territories:

  : States

### How is the underlying data produced?

We split the name into first name and last name (see the Python notebook
for how we do this) and then aggregate per state and first_name, and
tabulate [prop_male, prop_female, prop_third_gender, n_female, n_male,
n_third_gender]{.title-ref}. We produce native language rolls and
English transliterations. (We use
[indicate](https://github.com/in-rolls/indicate) to produce
transliterations for Hindi rolls.)

This is used to provide the base prediction.

Given the association between prop_female and first_name may change over
time, we exploited the age. Given the data were collected in 2017, we
calculated the year each person was born and then did a group by year to
create [prop_male, prop_female, prop_third_gender, n_female, n_male,
n_third_gender]{.title-ref}

### Issues with underlying data

Concerns:

- Voting registration lists may not be accurate, systematically
  underrepresenting poor people, minorities, and similar groups.
- Voting registration lists are, at best, a census of adult citizens.
  But to the extent there is prejudice against women, etc., that
  prevents them from reaching adulthood, the data bakes those biases in.
- Indian names are complicated. We do not have good parsers for them
  yet. We have gone for the default arrangement. Please go through the
  notebook to look at the judgments we make. We plan to improve the
  underlying data over time.
- For state electoral rolls that are neither in English nor Hindi, we
  use libindic. The quality of transliterations is consistently bad.

### Gender Classifier

We start by providing a base model for first_name that gives the Bayes
optimal solution\-\--the proportion of people with that name who are
women. We also provide a series of base models where the state of
residence and year of birth is known.

If the name does not exist in the database, we use [ML
model](https://github.com/appeler/naampy/blob/master/naampy/data/ml_model/02_training_model.ipynb)
that uses the relationship between sequences of characters in the first
name and gender to predict gender from the name.

The model was trained as a regression problem instead of a
classification problem because men and women share names. (See the
histogram below for the female proportion for the dataset.) The model
predicts the female proportion of the name. If it is less than 0.5, we
classify it as male; otherwise, we classify it as female.

<figure class="align-center">
<img src="images/female_prop.png" width="400" height="250"
alt="images/female_prop.png" />
</figure>

**Test data**

MSE no weights - loss: .05, metric: 0.05

RMSE no weights - loss: 0.22, metric: 0.22

**Test data with weights**

MSE with weights - loss: 0.05, metric: 0.04

RMSE with weights - loss: 0.22, metric: 0.22

Below are the inference results using different models.

<figure class="align-center">
<img src="images/infer_oos.png" alt="images/infer_oos.png" />
</figure>

### Installation

We strongly recommend installing [naampy]{.title-ref} inside a Python
virtual environment (see [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

    pip install naampy

### Usage

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

        choices=["v1", "v2", "v2_1k", "v2_native", "v2_en"],

### Using naampy

    >>> import pandas as pd
    >>> from naampy import in_rolls_fn_gender

    >>> names = [{'name': 'gaurav'},
                 {'name': 'nabha'},
                 {'name': 'yasmin'},
                 {'name': 'deepti'},
                 {'name': 'hrithik'},
                 {'name': 'vivek'}]


    >>> df = pd.DataFrame(names)

    >>> in_rolls_fn_gender(df, 'name')
                name    n_male  n_female  n_third_gender  prop_female  prop_male  prop_third_gender pred_gender  pred_prob
        0   gaurav   25625.0      47.0             0.0     0.001831   0.998169                0.0         NaN        NaN
        1    nabha       NaN       NaN             NaN          NaN        NaN                NaN      female   0.755028
        2   yasmin      58.0    6079.0             0.0     0.990549   0.009451                0.0         NaN        NaN
        3   deepti      35.0    5784.0             0.0     0.993985   0.006015                0.0         NaN        NaN
        4  hrithik       NaN       NaN             NaN          NaN        NaN                NaN        male   0.922181
        5    vivek  233622.0    1655.0             0.0     0.007034   0.992966                0.0         NaN        NaN

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

    # If you want to use model prediction use `predict_fn_gender` like below
    from naampy import predict_fn_gender
    input = [
         "rajinikanth",
         "harvin",
         "Shyamsingha",
         "srihan",
         "thammam",
         "bahubali",
         "rajarajeshwari",
         "shobby",
         "tamannaah bhatia",
         "mehreen",
         "kiara",
         "shivathmika",
         "komalee",
         "nazriya",
         "nabha",
         "taapsee",
         "parineeti",
         "katrina",
         "ileana",
         "vishwaksen",
         "sampoornesh",
         "hrithik",
         "emraan",
         "rajkummar",
         "sharman",
         "ayushmann",
         "irrfan",
         "riteish"
    ]
    print(predict_fn_gender(input))

                        name pred_gender  pred_prob
    0        rajinikanth        male   0.994747
    1             harvin        male   0.840713
    2        shyamsingha        male   0.956903
    3             srihan        male   0.825542
    4            thammam      female   0.564286
    5           bahubali        male   0.901159
    6     rajarajeshwari      female   0.942478
    7             shobby        male   0.788314
    8   tamannaah bhatia      female   0.971478
    9            mehreen      female   0.659633
    10             kiara      female   0.614125
    11       shivathmika      female   0.743240
    12           komalee      female   0.901051
    13           nazriya      female   0.854167
    14             nabha      female   0.755028
    15           taapsee      female   0.665176
    16         parineeti      female   0.813237
    17           katrina      female   0.630126
    18            ileana      female   0.640331
    19        vishwaksen        male   0.992237
    20       sampoornesh        male   0.940307
    21           hrithik        male   0.922181
    22            emraan        male   0.795963
    23         rajkummar        male   0.845139
    24           sharman        male   0.858538
    25         ayushmann        male   0.964895
    26            irrfan        male   0.837053
    27           riteish        male   0.950755

### Functionality

When you first run [in_rolls_fn_gender]{.title-ref}, it downloads data
from [Harvard
Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WZGJBM)
to the local folder. The next time you run the function, it searches for
local data, and if it finds it, it uses it. Use
[predict_fn_gender]{.title-ref} to get gender predictions based on first
name.

### Authors

Suriyan Laohaprapanon, Gaurav Sood, and Rajashekar Chintalapati


## 🔗 Adjacent Repositories

- [appeler/pranaam](https://github.com/appeler/pranaam) — pranaam: predict religion based on name
- [appeler/outkast](https://github.com/appeler/outkast) — Using data from over 140M+ Indians from the SECC 2011, we map last names to caste (SC, ST, Other)
- [appeler/graphic_names](https://github.com/appeler/graphic_names) — Infer the gender of person with a particular first name using Google image search and Clarifai
- [appeler/namesexdata](https://github.com/appeler/namesexdata) — Data on international first names and sex of people with that name
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.
