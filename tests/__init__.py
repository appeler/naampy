"""Shared test helpers."""

import os
import unittest

#: Tests that download the multi-hundred-megabyte Dataverse datasets are opt-in.
#: The rest of the suite runs offline against fabricated fixtures, so a default
#: `pytest` run is fast and does not depend on an external host being up.
requires_network = unittest.skipUnless(
    os.environ.get("NAAMPY_NETWORK_TESTS"),
    "set NAAMPY_NETWORK_TESTS=1 to run tests that download Dataverse datasets",
)
