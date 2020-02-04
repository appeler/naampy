#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for in_rolls_fn.py

"""

import os
import shutil
import unittest
import pandas as pd
from naampy.in_rolls_fn import in_rolls_fn_gender
from . import capture


class TestInRollsFn(unittest.TestCase):

    def setUp(self):
        names = [{'name': 'yasmin'},
                 {'name': 'vivek'}]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_in_rolls_fn(self):
        odf = in_rolls_fn_gender(self.df, 'name')
        self.assertIn('prop_female', odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state(self):
        odf = in_rolls_fn_gender(self.df, 'name', 'andaman')
        self.assertIn('prop_female', odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)

    def test_in_rolls_fn_state_year(self):
        odf = in_rolls_fn_gender(self.df, 'name', 'andhra', 1985)
        self.assertIn('prop_female', odf.columns)
        self.assertTrue(odf.iloc[0].prop_female > 0.9)
        self.assertTrue(odf.iloc[1].prop_female < 0.1)


if __name__ == '__main__':
    unittest.main()
