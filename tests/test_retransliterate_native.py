import gzip

import pandas as pd

from model_training.retransliterate_native import write_deterministic_gzip_csv


def test_deterministic_gzip_csv_is_path_and_time_independent(tmp_path):
    table = pd.DataFrame(
        {
            "state": ["assam", "punjab"],
            "birth_year": [1980.0, 1990.0],
            "first_name": ["anita", "gurpreet"],
            "n_female": [3, 2],
            "n_male": [1, 4],
            "n_third_gender": [0, 0],
            "prop_female": [0.75, 1 / 3],
        }
    )
    first_path = tmp_path / "first.csv.gz"
    second_path = tmp_path / "second.csv.gz"

    write_deterministic_gzip_csv(table, first_path)
    write_deterministic_gzip_csv(table, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    with gzip.open(first_path, "rt", encoding="utf-8", newline="") as source:
        assert source.read().splitlines()[0] == (
            "state,birth_year,first_name,n_female,n_male,n_third_gender,prop_female"
        )
