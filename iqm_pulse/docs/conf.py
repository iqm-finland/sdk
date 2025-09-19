# -*- coding: utf-8 -*-
#
# All configuration values have a default; values that are commented out
# serve to show the default.

from __future__ import annotations

import os
import sys

from packaging.version import parse

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
py_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../src")
sys.path.insert(0, os.path.abspath(py_path))


# -- Project information -----------------------------------------------------

project = "iqm.pulse"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ""
# The full version, including alpha/beta/rc tags.
release = ""
try:
    from iqm.pulse import __version__ as release
except ImportError:
    pass
else:
    version = parse(release).base_version

copyright = "2019-2025, IQM Finland Oy, Release {}".format(release)

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "7.2"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Autodoc ------------------------------------------------------------

# member ordering in autodoc output (default: 'alphabetical')
autodoc_member_order = "bysource"

# mock some imports when building the docs
# autodoc_mock_imports = ["django"]

# where should signature annotations appear in the docs, function signature or parameter description?
autodoc_typehints = "description"
# autodoc_typehints = 'description' puts the __init__ annotations into its docstring,
# which we thus have to include in the class documentation.
autoclass_content = "both"

# Sphinx 3.3+: manually clean up type alias rendering in the docs
# autodoc_type_aliases = {'TypeAlias': 'exa.experiment.somemodule.TypeAlias'}


# -- Autosummary ------------------------------------------------------------

# use autosummary to generate stub pages for API docs
autosummary_generate = True


# -- Options for HTML output ---------------------------------------------------

import sphinx_book_theme

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [sphinx_book_theme.get_html_theme_path()]

html_context = dict(display_github=False)

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 5,
    "use_download_button": False,
}

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/images/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/images/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%Y-%m-%d"

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# Shorten class names in function type hints
python_use_unqualified_type_names = True

# Output file base name for HTML help builder.
htmlhelp_basename = "iqm-pulse-doc"


# -- MathJax options ----------------------------------------------------------

# Here we configure MathJax, mostly to define LaTeX macros.
mathjax3_config = {
    "TeX": {
        "Macros": {
            "I": r"\mathbb{I}",
            "R": r"\mathbb{R}",
            "C": r"\mathbb{C}",
            "tr": r"\text{Tr}",
            "diag": r"\text{diag}",
            "ket": [r"\left| #1 \right\rangle", 1],
            "bra": [r"\left\langle #1 \right|", 1],
        }
    }
}


# -- External mapping ------------------------------------------------------------

python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.6.3/reference", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "exa.common": ("../exa-common", "../../exa-common/build/sphinx/objects.inv"),
    "iqm.models": ("../iqm-data-definitions", "https://iqm.gitlab-pages.iqm.fi/qccsw/iqm-data-definitions/objects.inv"),
}

use_local_target = os.getenv("USE_LOCAL_TARGET", "").lower()
if use_local_target == "true":
    intersphinx_mapping.update(
        {
            "iqm.data_definitions": (
                "../iqm-data-definitions",
                "https://iqm.gitlab-pages.iqm.fi/qccsw/iqm-data-definitions/objects.inv",
            ),
            "exa.common": (
                "../exa-common",
                "https://iqm.gitlab-pages.iqm.fi/qccsw/exa/exa-repo/exa-common/objects.inv",
            ),
        }
    )
else:
    extlinks = {
        "issue": ("https://jira.iqm.fi/browse/%s", "issue %s"),
        "mr": ("https://gitlab.iqm.fi/iqm/qccsw/exa/exa-repo/-/merge_requests/%s", "MR %s"),
    }

# -- sphinxcontrib.bibtex -------------------------------------------------

# List of all bibliography files used.
bibtex_bibfiles = ["references.bib"]
