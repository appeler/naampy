#!/usr/bin/env python

"""
Tests for electoral roll functionality (in_rolls_fn_gender).

Tests core electoral roll data processing without ML model integration.
"""

import unittest

import pandas as pd

from naampy.in_rolls_fn import in_rolls_fn_gender


class TestElectoralRoll(unittest.TestCase):
    """Test electoral roll gender prediction functionality."""

    def setUp(self):
        """Set up test data with known names from electoral rolls."""
        # Use names that should be in electoral database
        names = [{"name": "yasmin"}, {"name": "vivek"}]
        self.df = pd.DataFrame(names)

    def test_basic_electoral_roll_prediction(self):
        """Test basic electoral roll prediction functionality."""
        result = in_rolls_fn_gender(self.df, "name")

        # Check required columns are present
        expected_cols = [
            "prop_female",
            "prop_male",
            "prop_third_gender",
            "n_female",
            "n_male",
            "n_third_gender",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Validate gender predictions for known names
        # yasmin should be predominantly female
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        self.assertTrue(yasmin_row["prop_female"] > 0.9)

        # vivek should be predominantly male
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        self.assertTrue(vivek_row["prop_female"] < 0.1)

    def test_state_filtering(self):
        """Test state-specific electoral roll data."""
        result = in_rolls_fn_gender(self.df, "name", state="andaman")

        # Should still have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should still be reasonable
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        self.assertTrue(
            yasmin_row["prop_female"] > 0.8
        )  # Slightly more lenient for state-specific
        self.assertTrue(vivek_row["prop_female"] < 0.2)

    def test_year_filtering(self):
        """Test year-specific electoral roll data."""
        result = in_rolls_fn_gender(self.df, "name", state="andhra", year=1985)

        # Should still have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should still be reasonable
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        self.assertTrue(yasmin_row["prop_female"] > 0.8)
        self.assertTrue(vivek_row["prop_female"] < 0.2)

    def test_dataset_v1(self):
        """Test v1 dataset functionality."""
        result = in_rolls_fn_gender(
            self.df, "name", state="andhra", year=1985, dataset="v1"
        )

        # Should have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should be consistent
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        self.assertTrue(yasmin_row["prop_female"] > 0.8)
        self.assertTrue(vivek_row["prop_female"] < 0.2)

    def test_dataset_v2(self):
        """Test v2 dataset functionality."""
        result = in_rolls_fn_gender(
            self.df, "name", state="andhra", year=1985, dataset="v2"
        )

        # Should have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should be consistent
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        self.assertTrue(yasmin_row["prop_female"] > 0.8)
        self.assertTrue(vivek_row["prop_female"] < 0.2)

    def test_column_types_and_values(self):
        """Test that output columns have correct types and value ranges."""
        result = in_rolls_fn_gender(self.df, "name")

        # Proportion columns should be between 0 and 1
        for col in ["prop_female", "prop_male", "prop_third_gender"]:
            if col in result.columns:
                valid_props = result[col].dropna()
                self.assertTrue(all(valid_props >= 0), f"{col} has values < 0")
                self.assertTrue(all(valid_props <= 1), f"{col} has values > 1")

        # Count columns should be non-negative
        for col in ["n_female", "n_male", "n_third_gender"]:
            if col in result.columns:
                valid_counts = result[col].dropna()
                self.assertTrue(all(valid_counts >= 0), f"{col} has negative values")

    def test_proportion_sum_consistency(self):
        """Test that gender proportions sum to approximately 1 for valid data."""
        result = in_rolls_fn_gender(self.df, "name")

        for idx, row in result.iterrows():
            if pd.notna(row["prop_female"]) and pd.notna(row["prop_male"]):
                prop_sum = (
                    row["prop_female"]
                    + row["prop_male"]
                    + row.get("prop_third_gender", 0)
                )
                self.assertAlmostEqual(
                    prop_sum,
                    1.0,
                    places=2,
                    msg=f"Proportions don't sum to 1 for row {idx}",
                )

    def test_invalid_column_name(self):
        """Test behavior with invalid column name."""
        result = in_rolls_fn_gender(self.df, "nonexistent_column")

        # Should handle gracefully and return original df
        self.assertEqual(len(result), len(self.df))

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["name"])
        result = in_rolls_fn_gender(empty_df, "name")

        # Should return empty DataFrame with appropriate columns
        self.assertEqual(len(result), 0)
        self.assertIn("name", result.columns)


if __name__ == "__main__":
    unittest.main()
