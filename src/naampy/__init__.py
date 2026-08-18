"""Population-level pattern estimates and exact compositions from first names."""

from importlib.metadata import version

from .inference import estimate_first_name_pattern, lookup_first_name_composition

__version__ = version("naampy")

__all__ = [
    "__version__",
    "estimate_first_name_pattern",
    "lookup_first_name_composition",
]
