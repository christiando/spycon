import os
import sys
from recommonmark.parser import CommonMarkParser
from xml.sax.handler import feature_external_ges

sys.path.insert(0, os.path.abspath("../"))
mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "spycon"
copyright = "2023, Christian Donner"
author = "Christian Donner"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
]

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
}


autosummary_generate = True

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "repository_url": "https://gitlab.renkulab.io/christian.donner/spycon",
    "use_repository_button": True,
    "show_toc_level": 2,
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
html_logo = "_static/spycon.png"

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

source_parsers = {
    ".md": CommonMarkParser,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
