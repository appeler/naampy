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
        names = [{"name": "yasmin"}, {"name": "vivek"}]
        self.f_names = ["nabha", "ayushmann"]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_in_rolls_fn(self):
        odf = in_rolls_fn_gender(self.df, "name")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state(self):
        odf = in_rolls_fn_gender(self.df, "name", "andaman")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state_year(self):
        odf = in_rolls_fn_gender(self.df, "name", "andhra", 1985)
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state_year_v1(self):
        odf = in_rolls_fn_gender(self.df, "name", "andhra", 1985, "v1")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state_year_v2(self):
        odf = in_rolls_fn_gender(self.df, "name", "andhra", 1985, "v2")
        self.assertIn("prop_female", odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_predict_fn_gender(self):
        odf = predict_fn_gender(self.f_names)
        self.assertIn("pred_gender", odf.columns)
        self.assertTrue(odf.iloc[0].pred_gender == "female")
        self.assertTrue(odf.iloc[1].pred_gender == "male")


if __name__ == "__main__":
    unittest.main()
