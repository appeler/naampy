# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.abspath("../../"))

# Read project metadata from pyproject.toml
pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project_info = pyproject_data["project"]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = project_info["name"]
version = project_info["version"]
release = version
author = ", ".join([author["name"] for author in project_info["authors"]])
copyright = f"2023, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

# Source file configuration
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".ipynb": None,  # Handled by nbsphinx
}

# Set markdown as primary source
primary_source_suffix = ".md"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    # TensorFlow doesn't provide objects.inv file
    # 'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# NBSphinx settings
nbsphinx_execute = "always"  # Execute notebooks during build
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 150}",
]
nbsphinx_timeout = 300  # 5 minutes timeout for notebook execution
nbsphinx_allow_errors = False  # Fail build if notebook has errors

templates_path = ["_templates"]
exclude_patterns = ["naampy.tests*", "_build", "Thumbs.db", ".DS_Store"]

# Autodoc exclusions
autodoc_mock_imports = []
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#336790",
        "color-brand-content": "#336790",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4db8ff",
        "color-brand-content": "#4db8ff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
}

# Custom sidebar templates for furo theme
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
