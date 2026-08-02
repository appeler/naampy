#!/usr/bin/env python

"""
Offline regression tests for correctness bugs in the inference path.

Every test here patches ``InRollsFnData.load_naampy_data`` to point at a small
fabricated lookup table, so the suite needs no network access and asserts on
known counts rather than on whatever the live Dataverse files happen to contain.
Each test fails against the pre-fix implementation.
"""

import gzip
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import requests

from naampy.in_rolls_fn import InRollsFnData, in_rolls_fn_gender
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
        """Write the fabricated lookup table to a gzipped CSV on disk."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.fixture_path = os.path.join(cls._tmpdir.name, "naampy_fixture.csv.gz")
        pd.DataFrame(
            FIXTURE_ROWS,
            columns=[
                "state",
                "birth_year",
                "first_name",
                "n_female",
                "n_male",
                "n_third_gender",
            ],
        ).to_csv(cls.fixture_path, index=False, compression="gzip")

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

    def test_rerunning_on_own_output_is_idempotent(self):
        """Feeding naampy's output back in must not crash or duplicate columns."""
        df = pd.DataFrame({"name": ["priya", "aadhyashree"]})

        first = in_rolls_fn_gender(df, "name")
        second = in_rolls_fn_gender(first, "name")

        suffixed = [c for c in second.columns if c.endswith(("_x", "_y"))]
        self.assertEqual(suffixed, [], f"merge produced suffixed columns: {suffixed}")
        pd.testing.assert_frame_equal(first, second)

    def test_missing_namecol_returns_input_unchanged(self):
        """An absent or empty name column is reported, not raised."""
        df = pd.DataFrame({"name": ["priya"]})

        self.assertEqual(len(in_rolls_fn_gender(df, "nonexistent")), 1)
        self.assertEqual(len(in_rolls_fn_gender(df, "")), 1)


class TestNonLatinNames(FixtureTestCase):
    """Names the character model cannot read must not be given a gender."""

    def test_devanagari_name_is_not_labelled_male(self):
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
        self.target = os.path.join(self._tmpdir.name, "naampy_data.csv.gz")

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

        with open(self.target, "rb") as fh:
            self.assertEqual(fh.read(), b"abcdef")
        self.assertEqual(os.listdir(self._tmpdir.name), ["naampy_data.csv.gz"])

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

        self.assertFalse(os.path.exists(self.target))
        self.assertEqual(os.listdir(self._tmpdir.name), [])

    def test_truncated_download_is_rejected(self):
        """A short read against a known Content-Length is treated as a failure."""
        with mock.patch(
            "naampy.utils.requests.get",
            return_value=self._response([b"abc"], length=6),
        ):
            self.assertFalse(download_file("http://example.invalid/f", self.target))

        self.assertFalse(os.path.exists(self.target))
        self.assertEqual(os.listdir(self._tmpdir.name), [])

    def test_http_error_is_reported(self):
        """A non-200 response returns False without writing anything."""
        with mock.patch(
            "naampy.utils.requests.get", return_value=self._response([], status=404)
        ):
            self.assertFalse(download_file("http://example.invalid/f", self.target))

        self.assertFalse(os.path.exists(self.target))
        self.assertEqual(os.listdir(self._tmpdir.name), [])


class TestFixtureSanity(FixtureTestCase):
    """Guard the fixture itself so a broken fixture cannot mask a real failure."""

    def test_fixture_is_readable(self):
        """The gzipped fixture holds the rows the other tests assert on."""
        with gzip.open(self.fixture_path, "rt") as fh:
            rows = fh.read().strip().splitlines()
        self.assertEqual(len(rows), len(FIXTURE_ROWS) + 1)


if __name__ == "__main__":
    unittest.main()
