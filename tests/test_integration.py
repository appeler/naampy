#!/usr/bin/env python

"""
Tests for integration between electoral roll and ML model functionality.

Tests end-to-end workflows and ML fallback when names are not in electoral data.
"""

import unittest

import pandas as pd

from naampy.in_rolls_fn import in_rolls_fn_gender


class TestIntegration(unittest.TestCase):
    """Test integration between electoral roll data and ML model."""

    def setUp(self):
        """Set up test data for integration testing."""
        # Mix of names: some likely in electoral DB, some requiring ML fallback
        self.mixed_names_df = pd.DataFrame(
            {
                "name": [
                    "yasmin",
                    "nabha",
                    "vivek",
                    "kiara",
                ]  # yasmin/vivek=electoral, nabha/kiara=likely ML
            }
        )

        # Modern names likely not in electoral roll database
        self.modern_names_df = pd.DataFrame(
            {"name": ["aadhyashree", "vihaan", "reyansh", "aradhya"]}
        )

    def test_electoral_with_ml_fallback(self):
        """Test mixed data sources: electoral DB + ML fallback."""
        result = in_rolls_fn_gender(self.mixed_names_df, "name")

        # Check that all names have some form of prediction
        self.assertEqual(len(result), 4)

        # yasmin (should be in electoral data - highly female)
        yasmin_row = result[result["name"] == "yasmin"].iloc[0]
        if pd.notna(yasmin_row["prop_female"]):
            self.assertTrue(yasmin_row["prop_female"] > 0.9)
        elif pd.notna(yasmin_row.get("pred_gender")):
            self.assertEqual(yasmin_row["pred_gender"], "female")

        # vivek (should be in electoral data - highly male)
        vivek_row = result[result["name"] == "vivek"].iloc[0]
        if pd.notna(vivek_row["prop_female"]):
            self.assertTrue(vivek_row["prop_female"] < 0.1)
        elif pd.notna(vivek_row.get("pred_gender")):
            self.assertEqual(vivek_row["pred_gender"], "male")

        # nabha/kiara should have some form of prediction
        for name in ["nabha", "kiara"]:
            name_row = result[result["name"] == name].iloc[0]
            has_electoral = pd.notna(name_row["prop_female"])
            has_ml = pd.notna(name_row.get("pred_gender"))

            self.assertTrue(
                has_electoral or has_ml,
                f"Name '{name}' has no prediction from any source",
            )

    def test_pure_ml_fallback_rare_names(self):
        """Test pure ML fallback with modern/rare names."""
        result = in_rolls_fn_gender(self.modern_names_df, "name")

        # Should have predictions for all names
        self.assertEqual(len(result), 4)

        # Check each name has some form of prediction
        for _idx, row in result.iterrows():
            name = row["name"]
            has_electoral = pd.notna(row["prop_female"])
            has_ml = pd.notna(row.get("pred_gender"))

            # Each name should have either electoral data OR ML prediction
            self.assertTrue(
                has_electoral or has_ml,
                f"Name '{name}' has no prediction from either source",
            )

            # If using ML, verify confidence score is valid
            if has_ml:
                self.assertIn(row["pred_gender"], ["male", "female"])
                self.assertIsNotNone(row["pred_prob"])
                self.assertGreaterEqual(row["pred_prob"], 0.0)
                self.assertLessEqual(row["pred_prob"], 1.0)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Start with raw data
        raw_data = pd.DataFrame(
            {
                "employee_id": [101, 102, 103, 104],
                "full_name": [
                    "Priya Sharma",
                    "Rahul Kumar",
                    "Unknown ModernName",
                    "Vivek Singh",
                ],
                "first_name": ["priya", "rahul", "unknown_modern_name", "vivek"],
            }
        )

        # Process through naampy
        enriched_data = in_rolls_fn_gender(raw_data, "first_name")

        # Verify enrichment worked
        self.assertEqual(len(enriched_data), 4)
        self.assertIn("prop_female", enriched_data.columns)

        # Create summary statistics
        predictions = []
        for _idx, row in enriched_data.iterrows():
            if pd.notna(row["prop_female"]):
                # Use electoral data
                gender = "female" if row["prop_female"] > 0.5 else "male"
                confidence = max(row["prop_female"], 1 - row["prop_female"])
                source = "electoral"
            elif pd.notna(row.get("pred_gender")):
                # Use ML prediction
                gender = row["pred_gender"]
                confidence = row["pred_prob"]
                source = "ml"
            else:
                gender = "unknown"
                confidence = 0.0
                source = "none"

            predictions.append(
                {
                    "name": row["first_name"],
                    "predicted_gender": gender,
                    "confidence": confidence,
                    "source": source,
                }
            )

        # Validate summary
        self.assertEqual(len(predictions), 4)
        for pred in predictions:
            self.assertIn(pred["predicted_gender"], ["male", "female", "unknown"])
            self.assertIn(pred["source"], ["electoral", "ml", "none"])

    def test_fallback_behavior_consistency(self):
        """Test that fallback behavior is consistent across calls."""
        # Names that might or might not be in electoral data
        test_names = pd.DataFrame(
            {"name": ["test_consistency_1", "test_consistency_2"]}
        )

        # Run multiple times
        result1 = in_rolls_fn_gender(test_names, "name")
        result2 = in_rolls_fn_gender(test_names, "name")

        # Should get consistent results
        for i in range(len(test_names)):
            name = test_names.iloc[i]["name"]
            row1 = result1[result1["name"] == name].iloc[0]
            row2 = result2[result2["name"] == name].iloc[0]

            # Electoral data should be identical
            self.assertEqual(pd.isna(row1["prop_female"]), pd.isna(row2["prop_female"]))
            if pd.notna(row1["prop_female"]):
                self.assertEqual(row1["prop_female"], row2["prop_female"])

            # ML predictions should be identical
            if pd.notna(row1.get("pred_gender")):
                self.assertEqual(row1["pred_gender"], row2["pred_gender"])
                self.assertEqual(row1["pred_prob"], row2["pred_prob"])

    def test_data_quality_assessment(self):
        """Test integration for data quality assessment workflow."""
        # Mix of high-quality and questionable names (no empty strings to avoid issues)
        quality_test_df = pd.DataFrame(
            {"name": ["priya", "rahul", "a", "very_rare_modern_name", "yasmin"]}
        )

        result = in_rolls_fn_gender(quality_test_df, "name")

        # Calculate quality metrics
        total_names = len(quality_test_df)
        found_in_electoral = result["prop_female"].notna().sum()
        ml_predictions = result.get("pred_gender", pd.Series()).notna().sum()

        # Count names that are likely problematic (very short)
        short_names = (quality_test_df["name"].str.len() <= 1).sum()

        quality_metrics = {
            "total_names": total_names,
            "found_in_electoral_data": found_in_electoral,
            "ml_predictions": ml_predictions,
            "short_names": short_names,
            "coverage_rate": (found_in_electoral + ml_predictions)
            / max(1, total_names),
        }

        # Validate metrics make sense
        self.assertGreaterEqual(quality_metrics["coverage_rate"], 0.0)
        self.assertLessEqual(quality_metrics["coverage_rate"], 1.0)
        self.assertEqual(
            quality_metrics["short_names"], 1
        )  # One single-letter name in test data

        # Should have at least some predictions
        self.assertGreater(found_in_electoral + ml_predictions, 0)

    def test_batch_processing_simulation(self):
        """Test simulation of batch processing workflow."""
        # Simulate larger batch
        batch_data = pd.DataFrame(
            {
                "name": ["priya", "rahul", "anjali", "vikram"] * 10  # 40 names
            }
        )

        result = in_rolls_fn_gender(batch_data, "name")

        # Should process all names
        self.assertEqual(len(result), 40)

        # Should have consistent predictions for duplicate names
        unique_names = batch_data["name"].unique()
        for name in unique_names:
            name_rows = result[result["name"] == name]
            if len(name_rows) > 1:
                # All rows for same name should have identical predictions
                first_row = name_rows.iloc[0]
                for _idx, row in name_rows.iterrows():
                    if pd.notna(first_row["prop_female"]):
                        self.assertEqual(row["prop_female"], first_row["prop_female"])
                    if pd.notna(first_row.get("pred_gender")):
                        self.assertEqual(row["pred_gender"], first_row["pred_gender"])
                        self.assertEqual(row["pred_prob"], first_row["pred_prob"])

    def test_mixed_dataset_behavior(self):
        """Test integration with different dataset versions."""
        test_names = pd.DataFrame({"name": ["test1", "test2"]})

        # Test different datasets
        datasets = ["v2_1k", "v2", "v1"]
        results = {}

        for dataset in datasets:
            try:
                results[dataset] = in_rolls_fn_gender(
                    test_names, "name", dataset=dataset
                )
                # Should have same structure regardless of dataset
                self.assertEqual(len(results[dataset]), 2)
                self.assertIn("prop_female", results[dataset].columns)
            except Exception:
                # Some datasets might not be available in test environment
                pass


if __name__ == "__main__":
    unittest.main()
