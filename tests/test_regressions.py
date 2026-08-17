"""
Offline regression tests for correctness bugs in the inference path.

Every test here patches ``InRollsFnData.load_naampy_data`` to point at a small
fabricated lookup table, so the suite needs no network access and asserts on
known counts rather than on whatever the live Dataverse files happen to contain.
Each test fails against the pre-fix implementation.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from naampy._tables import ROLL_SCHEMA, csv_to_parquet, validate_parquet
from naampy.in_rolls_fn import OUTPUT_COLS, InRollsFnData, in_rolls_fn_gender
from naampy.utils import download_file

# state, birth_year, first_name, n_female, n_male, n_third_gender.
# priya: 100 in kerala + 300 in delhi = 400 nationally.
# rahul: 200 born in 1990 + 40 born in 1985 = 240 across all years.
# nocount: present in the table but with zero recorded voters.
FIXTURE_ROWS = [
    ("kerala", 1985, "priya", 100, 0, 0),
    ("delhi", 1985, "priya", 300, 0, 0),
    ("kerala", 1990, "rahul", 0, 50, 0),
    ("delhi", 1990, "rahul", 0, 150, 0),
    ("delhi", 1985, "rahul", 0, 40, 0),
    ("kerala", 1985, "nocount", 0, 0, 0),
]


class FixtureTestCase(unittest.TestCase):
    """Base class wiring the fabricated lookup table into InRollsFnData."""

    @classmethod
    def setUpClass(cls):
        """Write the fabricated lookup table to typed Parquet on disk."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.fixture_path = Path(cls._tmpdir.name) / "naampy_fixture.parquet"
        rows = [dict(zip(ROLL_SCHEMA.names, row, strict=True)) for row in FIXTURE_ROWS]
        pq.write_table(pa.Table.from_pylist(rows, schema=ROLL_SCHEMA), cls.fixture_path)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary fixture directory."""
        cls._tmpdir.cleanup()

    def setUp(self):
        """Clear the class-level lookup cache and redirect dataset loading."""
        InRollsFnData._InRollsFnData__df = None
        InRollsFnData._InRollsFnData__cache_key = None
        patcher = mock.patch.object(
            InRollsFnData, "load_naampy_data", return_value=self.fixture_path
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class TestCacheInvalidation(FixtureTestCase):
    """The cached lookup table must not outlive the filter that produced it."""

    def test_state_filtered_cache_is_not_reused_nationally(self):
        """A national query after a state query must not reuse the filtered table."""
        df = pd.DataFrame({"name": ["priya"]})

        kerala = in_rolls_fn_gender(df, "name", state="kerala")
        national = in_rolls_fn_gender(df, "name")

        self.assertEqual(kerala.loc[0, "n_female"], 100)
        self.assertEqual(national.loc[0, "n_female"], 400)

    def test_year_filtered_cache_is_not_reused_across_all_years(self):
        """An all-years query after a year query must not reuse the filtered table."""
        df = pd.DataFrame({"name": ["rahul"]})

        year_1990 = in_rolls_fn_gender(df, "name", year=1990)
        all_years = in_rolls_fn_gender(df, "name")

        self.assertEqual(year_1990.loc[0, "n_male"], 200)
        self.assertEqual(all_years.loc[0, "n_male"], 240)

    def test_repeated_identical_query_is_stable(self):
        """Two identical calls must return identical counts."""
        df = pd.DataFrame({"name": ["priya"]})

        first = in_rolls_fn_gender(df, "name", state="delhi")
        second = in_rolls_fn_gender(df, "name", state="delhi")

        self.assertEqual(first.loc[0, "n_female"], second.loc[0, "n_female"])
        self.assertEqual(first.loc[0, "n_female"], 300)

    def test_unknown_state_is_rejected(self):
        """A state typo must not silently turn every row into an ML fallback."""
        with self.assertRaisesRegex(ValueError, "State 'punajb' is not in dataset"):
            in_rolls_fn_gender(
                pd.DataFrame({"name": ["priya"]}), "name", state="punajb"
            )

    def test_unknown_state_year_is_rejected(self):
        """A year unavailable for the requested state is reported explicitly."""
        with self.assertRaisesRegex(ValueError, "Birth year 1900.*state 'delhi'"):
            in_rolls_fn_gender(
                pd.DataFrame({"name": ["priya"]}),
                "name",
                state="delhi",
                year=1900,
            )


class TestMissingNames(FixtureTestCase):
    """Rows without a usable name must never reach the model."""

    def test_missing_names_are_not_scored(self):
        """None, NaN, empty and whitespace-only names come back unscored."""
        df = pd.DataFrame({"name": ["priya", None, "", "   ", float("nan")]})

        result = in_rolls_fn_gender(df, "name")

        for row in range(1, 5):
            self.assertTrue(
                pd.isna(result.loc[row, "prop_female"]),
                f"row {row} got electoral data for a missing name",
            )
            self.assertTrue(
                pd.isna(result.loc[row, "pred_gender"]),
                f"row {row} was assigned a gender despite having no name",
            )
            self.assertTrue(pd.isna(result.loc[row, "pred_prob"]))

    def test_zero_count_name_yields_nan_proportions(self):
        """A name with zero recorded voters yields NaN, not a divide-by-zero value."""
        df = pd.DataFrame({"name": ["nocount"]})

        result = in_rolls_fn_gender(df, "name")

        self.assertEqual(result.loc[0, "n_female"], 0)
        self.assertTrue(pd.isna(result.loc[0, "prop_female"]))


class TestOutputContract(FixtureTestCase):
    """The shape of the returned frame must not depend on the hit rate."""

    def test_ml_columns_present_when_every_name_is_found(self):
        """pred_gender/pred_prob exist even when nothing falls through to the model."""
        df = pd.DataFrame({"name": ["priya", "rahul"]})

        result = in_rolls_fn_gender(df, "name")

        self.assertIn("pred_gender", result.columns)
        self.assertIn("pred_prob", result.columns)
        self.assertTrue(result["pred_gender"].isna().all())

    def test_ml_columns_absent_for_v2_native(self):
        """v2_native opts out of the model entirely."""
        df = pd.DataFrame({"name": ["priya", "unseen_name"]})

        result = in_rolls_fn_gender(df, "name", dataset="v2_native")

        self.assertNotIn("pred_gender", result.columns)
        self.assertNotIn("pred_prob", result.columns)

    def test_input_dataframe_is_not_mutated(self):
        """The caller's frame is left exactly as it was passed in."""
        df = pd.DataFrame({"name": ["priya", "rahul"]})
        before = df.copy()

        in_rolls_fn_gender(df, "name")

        pd.testing.assert_frame_equal(df, before)

    def test_empty_input_does_not_load_the_dataset(self):
        """An empty frame gets the output schema without a network or disk lookup."""
        with mock.patch.object(InRollsFnData, "load_naampy_data") as load:
            result = in_rolls_fn_gender(pd.DataFrame({"name": []}), "name")

        load.assert_not_called()
        self.assertEqual(result.columns.tolist(), ["name", *OUTPUT_COLS])

    def test_empty_input_still_rejects_an_unknown_dataset(self):
        """Dataset validation does not depend on whether input rows exist."""
        with self.assertRaisesRegex(ValueError, "Unknown dataset 'not-a-dataset'"):
            in_rolls_fn_gender(
                pd.DataFrame({"name": []}), "name", dataset="not-a-dataset"
            )

    def test_rerunning_on_own_output_is_idempotent(self):
        """Feeding naampy's output back in must not crash or duplicate columns."""
        df = pd.DataFrame({"name": ["priya", "aadhyashree"]})

        first = in_rolls_fn_gender(df, "name")
        second = in_rolls_fn_gender(first, "name")

        suffixed = [c for c in second.columns if c.endswith(("_x", "_y"))]
        self.assertEqual(suffixed, [], f"merge produced suffixed columns: {suffixed}")
        pd.testing.assert_frame_equal(first, second)

    def test_existing_output_columns_are_replaced_without_suffixes(self):
        """Documented outputs replace collisions and retain their canonical names."""
        df = pd.DataFrame({"name": ["priya"], "n_female": [-1], "prop_female": [-1.0]})

        result = in_rolls_fn_gender(df, "name")

        self.assertEqual(result.loc[0, "n_female"], 400)
        self.assertEqual(result.columns.tolist().count("n_female"), 1)
        self.assertFalse(any(column.endswith(("_x", "_y")) for column in result))

    def test_meaningful_duplicate_index_is_preserved(self):
        """Lookup enrichment preserves index values, names, and duplicates."""
        df = pd.DataFrame(
            {"name": ["priya", "rahul"]},
            index=pd.Index([7, 7], name="respondent_id"),
        )

        result = in_rolls_fn_gender(df, "name")

        pd.testing.assert_index_equal(result.index, df.index)
        self.assertEqual(result["n_female"].tolist(), [400, 0])

    def test_native_lookup_preserves_unrelated_prediction_columns(self):
        """Native mode does not delete columns it does not produce."""
        df = pd.DataFrame(
            {"name": ["priya"], "pred_gender": ["reviewed"], "pred_prob": [1.0]}
        )

        result = in_rolls_fn_gender(df, "name", dataset="v2_native")

        self.assertEqual(result.loc[0, "pred_gender"], "reviewed")
        self.assertEqual(result.loc[0, "pred_prob"], 1.0)

    def test_missing_namecol_returns_input_unchanged(self):
        """An absent or empty name column is reported, not raised."""
        df = pd.DataFrame({"name": ["priya"]})

        self.assertEqual(len(in_rolls_fn_gender(df, "nonexistent")), 1)
        self.assertEqual(len(in_rolls_fn_gender(df, "")), 1)


class TestNonLatinNames(FixtureTestCase):
    """Names the character model cannot read must not be given a gender."""

    def test_devanagari_name_is_not_labeled_male(self):
        """A name with no a-z characters comes back unscored under an English dataset."""
        df = pd.DataFrame({"name": ["हेमा", "ಅಂಕಿತಾ", "priya"]})

        result = in_rolls_fn_gender(df, "name")

        for row in (0, 1):
            self.assertTrue(
                pd.isna(result.loc[row, "pred_gender"]),
                f"row {row} was assigned a gender from an unreadable name",
            )
            self.assertTrue(pd.isna(result.loc[row, "pred_prob"]))

    def test_latin_name_outside_the_rolls_still_gets_a_prediction(self):
        """The model fallback still fires for readable names missing from the rolls."""
        df = pd.DataFrame({"name": ["aadhyashree"]})

        result = in_rolls_fn_gender(df, "name")

        self.assertTrue(pd.isna(result.loc[0, "prop_female"]))
        self.assertIn(result.loc[0, "pred_gender"], ("male", "female"))
        self.assertGreaterEqual(result.loc[0, "pred_prob"], 0.5)


class TestDownloadFile(unittest.TestCase):
    """A failed download must not leave anything behind for the cache to find."""

    def setUp(self):
        """Create a scratch directory for download targets."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.scratch = Path(self._tmpdir.name)
        self.target = self.scratch / "naampy_data.csv.gz"

    def _response(self, chunks, status=200, length=None):
        """Build a stub requests response yielding the given chunks.

        Args:
            chunks: Iterable of byte chunks, or a callable raising mid-stream.
            status: HTTP status code the stub should report.
            length: Value for the Content-Length header, if any.

        Returns:
            mock.Mock: A stand-in for a streamed requests response.
        """
        response = mock.Mock()
        response.status_code = status
        response.headers = {} if length is None else {"Content-Length": str(length)}
        response.iter_content.return_value = chunks
        if status != 200:
            response.raise_for_status.side_effect = requests.HTTPError(str(status))
        return response

    def test_successful_download_writes_target(self):
        """A clean transfer lands at the target path with no leftovers."""
        with mock.patch(
            "naampy.utils.requests.get",
            return_value=self._response([b"abc", b"def"], length=6),
        ):
            self.assertTrue(download_file("http://example.invalid/f", self.target))

        with self.target.open("rb") as fh:
            self.assertEqual(fh.read(), b"abcdef")
        self.assertEqual(
            [path.name for path in self.scratch.iterdir()], [self.target.name]
        )

    def test_interrupted_download_leaves_no_file(self):
        """A mid-stream failure leaves neither a target nor a partial file."""

        def exploding_chunks():
            yield b"abc"
            raise OSError("connection reset")

        with mock.patch(
            "naampy.utils.requests.get",
            return_value=self._response(exploding_chunks(), length=6),
        ):
            self.assertFalse(download_file("http://example.invalid/f", self.target))

        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_truncated_download_is_rejected(self):
        """A short read against a known Content-Length is treated as a failure."""
        with mock.patch(
            "naampy.utils.requests.get",
            return_value=self._response([b"abc"], length=6),
        ):
            self.assertFalse(download_file("http://example.invalid/f", self.target))

        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_http_error_is_reported(self):
        """A non-200 response returns False without writing anything."""
        with mock.patch(
            "naampy.utils.requests.get", return_value=self._response([], status=404)
        ):
            self.assertFalse(download_file("http://example.invalid/f", self.target))

        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])


class TestFixtureSanity(FixtureTestCase):
    """Guard the fixture itself so a broken fixture cannot mask a real failure."""

    def test_fixture_is_readable(self):
        """The Parquet fixture holds the rows the other tests assert on."""
        table = pq.read_table(self.fixture_path)
        self.assertEqual(table.num_rows, len(FIXTURE_ROWS))
        self.assertTrue(table.schema.equals(ROLL_SCHEMA, check_metadata=False))


class TestTypedCache(unittest.TestCase):
    """The downloaded CSV transport becomes a validated Parquet cache."""

    def setUp(self):
        """Create scratch paths for source and cache files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.source = Path(self._tmpdir.name) / "rolls.csv.gz"
        self.target = Path(self._tmpdir.name) / "rolls.parquet"

    def _write_source(self, rows=FIXTURE_ROWS):
        """Write source rows in the Dataverse transport format."""
        frame = pd.DataFrame(rows, columns=ROLL_SCHEMA.names)
        frame["birth_year"] = frame["birth_year"].astype(float)
        frame.to_csv(self.source, index=False, compression="gzip")

    def test_conversion_has_the_declared_schema(self):
        """A valid source is atomically converted to the canonical schema."""
        self._write_source()

        result = csv_to_parquet(self.source, self.target)

        self.assertEqual(result, self.target)
        self.assertEqual(validate_parquet(result), self.target)
        self.assertEqual(pq.read_table(result).num_rows, len(FIXTURE_ROWS))

    def test_missing_names_are_dropped_during_conversion(self):
        """Rows with no lookup key are not retained in the runtime table."""
        self._write_source([*FIXTURE_ROWS, ("delhi", 1990, None, 1, 0, 0)])

        csv_to_parquet(self.source, self.target)

        table = pq.read_table(self.target)
        self.assertEqual(table.num_rows, len(FIXTURE_ROWS))
        self.assertEqual(table.column("first_name").null_count, 0)

    def test_negative_counts_are_rejected_without_a_cache(self):
        """Invalid counts fail conversion and leave no durable cache."""
        self._write_source([("kerala", 1985, "priya", -1, 0, 0)])

        with self.assertRaisesRegex(ValueError, "contains negatives"):
            csv_to_parquet(self.source, self.target)

        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
