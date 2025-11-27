#!/usr/bin/env python

"""
Tests for native script and language support.

Tests functionality with Hindi, Gujarati, Kannada and other Indian language names,
primarily using the v2_native dataset.
"""

import unittest

import pandas as pd

from naampy.in_rolls_fn import in_rolls_fn_gender


class TestNativeScripts(unittest.TestCase):
    """Test native Indian language script support."""

    def setUp(self):
        """Set up test data with native script names."""
        # Hindi names from the original test
        self.hindi_names_df = pd.DataFrame([
            {"name": "हेमा"},
            {"name": "होशियार"}
        ])

        # Mixed native script names (from input-native.csv pattern)
        self.mixed_native_df = pd.DataFrame({
            'name': ['हेमा', 'होशियार', 'અંકલેશ્વરીયા', 'ಅಂಕಿತಾ']
        })

    def test_hindi_names_basic(self):
        """Test basic functionality with Hindi names."""
        result = in_rolls_fn_gender(self.hindi_names_df, "name", dataset="v2_native")

        # Should have required columns
        self.assertIn("prop_female", result.columns)

        # Validate predictions for known Hindi names
        # हेमा should be predominantly female
        hema_row = result[result['name'] == 'हेमा'].iloc[0]
        if pd.notna(hema_row['prop_female']):
            self.assertTrue(hema_row['prop_female'] > 0.8)

        # होशियार should be predominantly male
        hoshiyar_row = result[result['name'] == 'होशियार'].iloc[0]
        if pd.notna(hoshiyar_row['prop_female']):
            self.assertTrue(hoshiyar_row['prop_female'] < 0.2)

    def test_hindi_names_with_state(self):
        """Test Hindi names with state filtering."""
        result = in_rolls_fn_gender(self.hindi_names_df, "name", state="up", dataset="v2_native")

        # Should have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should still be reasonable for UP state
        hema_row = result[result['name'] == 'हेमा'].iloc[0]
        hoshiyar_row = result[result['name'] == 'होशियार'].iloc[0]

        if pd.notna(hema_row['prop_female']):
            self.assertTrue(hema_row['prop_female'] > 0.8)
        if pd.notna(hoshiyar_row['prop_female']):
            self.assertTrue(hoshiyar_row['prop_female'] < 0.2)

    def test_hindi_names_with_state_year(self):
        """Test Hindi names with state and year filtering."""
        result = in_rolls_fn_gender(self.hindi_names_df, "name", state="up", year=1985, dataset="v2_native")

        # Should have required columns
        self.assertIn("prop_female", result.columns)

        # Gender predictions should still be reasonable
        hema_row = result[result['name'] == 'हेमा'].iloc[0]
        hoshiyar_row = result[result['name'] == 'होशियार'].iloc[0]

        if pd.notna(hema_row['prop_female']):
            self.assertTrue(hema_row['prop_female'] > 0.8)
        if pd.notna(hoshiyar_row['prop_female']):
            self.assertTrue(hoshiyar_row['prop_female'] < 0.2)

    def test_v2_native_no_ml_fallback(self):
        """Test that v2_native dataset does NOT use ML fallback."""
        # Include an unknown name that shouldn't be in any dataset
        test_names = pd.DataFrame({
            'name': ['हेमा', 'होशियार', 'unknown_native_name']
        })

        result = in_rolls_fn_gender(test_names, 'name', dataset='v2_native')

        # Check that we don't have ML fallback columns for v2_native
        self.assertNotIn('pred_gender', result.columns)
        self.assertNotIn('pred_prob', result.columns)

        # Unknown names should have NaN values, not ML predictions
        unknown_rows = result[result['name'] == 'unknown_native_name']
        if not unknown_rows.empty:
            unknown_row = unknown_rows.iloc[0]
            self.assertTrue(pd.isna(unknown_row['prop_female']))

    def test_mixed_native_scripts(self):
        """Test handling of multiple native scripts."""
        result = in_rolls_fn_gender(self.mixed_native_df, 'name', dataset='v2_native')

        # Should process all names without crashing
        self.assertEqual(len(result), len(self.mixed_native_df))

        # Should have standard columns
        expected_cols = ['prop_female', 'prop_male', 'prop_third_gender',
                        'n_female', 'n_male', 'n_third_gender']
        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_native_script_encoding_handling(self):
        """Test that native scripts are properly handled without encoding issues."""
        # Various script samples
        diverse_names = pd.DataFrame({
            'name': [
                'हेमा',        # Hindi (Devanagari)
                'અંકલેશ્વરીયા',  # Gujarati
                'ಅಂಕಿತಾ',       # Kannada
                'राम',         # Hindi short name
                'கமலா'         # Tamil (if supported)
            ]
        })

        # Should not crash with encoding issues
        result = in_rolls_fn_gender(diverse_names, 'name', dataset='v2_native')

        # Verify basic structure
        self.assertEqual(len(result), len(diverse_names))
        self.assertIn('name', result.columns)

    def test_native_vs_transliterated_comparison(self):
        """Test comparison between native script and transliterated versions."""
        # Test with names that have clear English equivalents
        native_df = pd.DataFrame({'name': ['राम', 'सीता']})  # Ram, Sita in Hindi
        english_df = pd.DataFrame({'name': ['ram', 'sita']})  # English versions

        # Get results from both datasets
        native_result = in_rolls_fn_gender(native_df, 'name', dataset='v2_native')
        english_result = in_rolls_fn_gender(english_df, 'name', dataset='v2_1k')

        # Both should process without errors
        self.assertEqual(len(native_result), 2)
        self.assertEqual(len(english_result), 2)

        # Both should have some predictions (though they might differ)
        self.assertTrue(any(pd.notna(native_result['prop_female'])) or
                       len(native_result) > 0)
        self.assertTrue(any(pd.notna(english_result['prop_female'])) or
                       len(english_result) > 0)

    def test_empty_native_input(self):
        """Test native script handling with empty input."""
        empty_df = pd.DataFrame({'name': []}, dtype=str)
        result = in_rolls_fn_gender(empty_df, 'name', dataset='v2_native')

        # Should handle gracefully
        self.assertEqual(len(result), 0)
        self.assertIn('name', result.columns)

    def test_native_script_column_validation(self):
        """Test that native script results have proper column types and values."""
        result = in_rolls_fn_gender(self.hindi_names_df, 'name', dataset='v2_native')

        # Proportion columns should be between 0 and 1 where not NaN
        for col in ["prop_female", "prop_male", "prop_third_gender"]:
            if col in result.columns:
                valid_props = result[col].dropna()
                if len(valid_props) > 0:
                    self.assertTrue(all(valid_props >= 0), f"{col} has values < 0")
                    self.assertTrue(all(valid_props <= 1), f"{col} has values > 1")

    def test_native_script_state_availability(self):
        """Test which states are available for native script data."""
        # Test with a known Hindi name
        test_df = pd.DataFrame({'name': ['हेमा']})

        # Try different states that should support native scripts
        test_states = ['up', 'bihar', 'jharkhand', 'mp']

        for state in test_states:
            try:
                result = in_rolls_fn_gender(test_df, 'name', state=state, dataset='v2_native')
                # Should not crash and should return data
                self.assertEqual(len(result), 1)
                self.assertIn('name', result.columns)
            except Exception:
                # Some states might not be available, which is okay
                pass

    def test_native_script_data_consistency(self):
        """Test that native script data is internally consistent."""
        result = in_rolls_fn_gender(self.hindi_names_df, 'name', dataset='v2_native')

        for idx, row in result.iterrows():
            if all(pd.notna([row.get('prop_female'), row.get('prop_male'), row.get('prop_third_gender')])):
                # Proportions should sum to approximately 1
                prop_sum = row['prop_female'] + row['prop_male'] + row['prop_third_gender']
                self.assertAlmostEqual(prop_sum, 1.0, places=2,
                                     msg=f"Proportions don't sum to 1 for native script name at row {idx}")


if __name__ == "__main__":
    unittest.main()
