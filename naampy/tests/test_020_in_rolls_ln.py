#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for in_rolls_fn.py

"""

import unittest
import pandas as pd
from naampy.in_rolls_fn import in_rolls_fn_gender, predict_fn_gender


class TestInRollsFn(unittest.TestCase):
    def setUp(self):
        names = [{"name": "हेमा"}, {"name": "होशियार"}]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_in_rolls_fn(self):
        odf = in_rolls_fn_gender(self.df, "name", dataset="v2_native")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.8)
        self.assertTrue(odf.iloc[1].prop_female < 0.2)

    def test_in_rolls_fn_state(self):
        odf = in_rolls_fn_gender(self.df, "name", "up", dataset="v2_native")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.8)
        self.assertTrue(odf.iloc[1].prop_female < 0.2)

    def test_in_rolls_fn_state_year(self):
        odf = in_rolls_fn_gender(self.df, "name", "up", 1985, dataset="v2_native")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.8)
        self.assertTrue(odf.iloc[1].prop_female < 0.2)


if __name__ == "__main__":
    unittest.main()
