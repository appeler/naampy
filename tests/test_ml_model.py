#!/usr/bin/env python

"""
Tests for ML model functionality (predict_fn_gender).

Tests pure neural network-based gender prediction without electoral roll data.
"""

import unittest

from naampy.in_rolls_fn import predict_fn_gender


class TestMLModel(unittest.TestCase):
    """Test ML model gender prediction functionality."""

    def setUp(self):
        """Set up test data with names for ML prediction."""
        # Use names that should work well with the ML model
        self.common_names = ["nabha", "ayushmann", "priya", "rahul"]
        self.confident_names = ["priya", "rahul", "anjali", "vikram"]

    def test_basic_ml_prediction(self):
        """Test basic ML model prediction functionality."""
        result = predict_fn_gender(self.common_names)

        # Check required columns are present
        expected_cols = ["name", "pred_gender", "pred_prob"]
        for col in expected_cols:
            self.assertIn(col, result.columns)

        # Check we have predictions for all names
        self.assertEqual(len(result), len(self.common_names))

        # Validate specific predictions for known patterns
        nabha_row = result[result['name'] == 'nabha'].iloc[0]
        self.assertEqual(nabha_row['pred_gender'], 'female')

        ayushmann_row = result[result['name'] == 'ayushmann'].iloc[0]
        self.assertEqual(ayushmann_row['pred_gender'], 'male')

    def test_confidence_scores_valid_range(self):
        """Test that confidence scores are in valid range [0, 1]."""
        result = predict_fn_gender(self.common_names)

        for _idx, row in result.iterrows():
            self.assertGreaterEqual(row['pred_prob'], 0.0,
                                   f"Confidence score below 0 for {row['name']}")
            self.assertLessEqual(row['pred_prob'], 1.0,
                                f"Confidence score above 1 for {row['name']}")

    def test_gender_values_valid(self):
        """Test that predicted gender values are valid."""
        result = predict_fn_gender(self.common_names)

        valid_genders = {'male', 'female'}
        for _idx, row in result.iterrows():
            self.assertIn(row['pred_gender'], valid_genders,
                         f"Invalid gender prediction for {row['name']}")

    def test_high_confidence_common_names(self):
        """Test that common names have high confidence scores."""
        result = predict_fn_gender(self.confident_names)

        # All should have confidence > 0.6 for these well-known names
        for _idx, row in result.iterrows():
            self.assertGreaterEqual(row['pred_prob'], 0.6,
                                   f"Low confidence for common name: {row['name']}")

    def test_edge_case_names(self):
        """Test ML model with edge case names."""
        edge_cases = [
            'a',  # Very short
            'abcdefghijklmnopqr',  # Long name
            'X',  # Single letter
            'zee'  # Short but valid
        ]

        result = predict_fn_gender(edge_cases)

        # Should handle all edge cases without crashing
        self.assertEqual(len(result), len(edge_cases))

        # All should have valid predictions
        for _idx, row in result.iterrows():
            self.assertIn(row['pred_gender'], ['male', 'female'])
            self.assertGreaterEqual(row['pred_prob'], 0.0)
            self.assertLessEqual(row['pred_prob'], 1.0)

    def test_name_preprocessing(self):
        """Test that names are properly preprocessed (lowercased)."""
        mixed_case_names = ["PRIYA", "Rahul", "aNjAlI", "vikram"]
        result = predict_fn_gender(mixed_case_names)

        # All names in result should be lowercase
        for _idx, row in result.iterrows():
            self.assertEqual(row['name'], row['name'].lower(),
                           f"Name not lowercased: {row['name']}")

    def test_empty_input(self):
        """Test ML model behavior with empty input."""
        # TensorFlow model expects non-empty input, so this should handle gracefully
        try:
            result = predict_fn_gender([])
            # If it succeeds, should return empty DataFrame with correct columns
            self.assertEqual(len(result), 0)
            expected_cols = ["name", "pred_gender", "pred_prob"]
            for col in expected_cols:
                self.assertIn(col, result.columns)
        except ValueError as e:
            # TensorFlow models may not handle empty input, which is acceptable behavior
            self.assertIn("empty", str(e).lower())
        except Exception as e:
            # Other exceptions are not expected
            self.fail(f"Unexpected exception with empty input: {e}")

    def test_single_name_input(self):
        """Test ML model with single name input."""
        result = predict_fn_gender(["deepika"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['name'], 'deepika')
        self.assertIn(result.iloc[0]['pred_gender'], ['male', 'female'])
        self.assertIsNotNone(result.iloc[0]['pred_prob'])

    def test_duplicate_names(self):
        """Test ML model with duplicate names in input."""
        names_with_duplicates = ["priya", "rahul", "priya", "anjali"]
        result = predict_fn_gender(names_with_duplicates)

        # Should handle duplicates and return predictions for all
        self.assertEqual(len(result), len(names_with_duplicates))

        # Check that duplicate names have consistent predictions
        priya_rows = result[result['name'] == 'priya']
        if len(priya_rows) > 1:
            first_prediction = priya_rows.iloc[0]['pred_gender']
            for _idx, row in priya_rows.iterrows():
                self.assertEqual(row['pred_gender'], first_prediction,
                               "Inconsistent predictions for duplicate names")

    def test_special_characters_handling(self):
        """Test ML model with names containing special characters."""
        special_names = ["mary-jane", "o'connor", "jean-luc"]

        # Model should handle these without crashing
        result = predict_fn_gender(special_names)
        self.assertEqual(len(result), len(special_names))

        # All should have valid predictions
        for _idx, row in result.iterrows():
            self.assertIn(row['pred_gender'], ['male', 'female'])

    def test_model_consistency(self):
        """Test that ML model gives consistent predictions for same input."""
        test_names = ["consistent_test_name"]

        # Run prediction multiple times
        result1 = predict_fn_gender(test_names)
        result2 = predict_fn_gender(test_names)

        # Should get identical results
        self.assertEqual(result1.iloc[0]['pred_gender'], result2.iloc[0]['pred_gender'])
        self.assertEqual(result1.iloc[0]['pred_prob'], result2.iloc[0]['pred_prob'])


if __name__ == "__main__":
    unittest.main()
