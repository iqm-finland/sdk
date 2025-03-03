# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Beautiful Soup'
copyright = '2004-2025 Leonard Richardson'
author = 'Leonard Richardson'
release = '4.13.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en,ja,ko,ru,pt,zh'

html_sidebars = { '**': ['localtoc.html', 'searchbox.html'] }

# I've got type aliases documented in the apidoc, but
# references to those aliases don't link to the definitions.
#
# https://github.com/sphinx-doc/sphinx/issues/10785
#autodoc_type_aliases = {
#    "_IncomingMarkup" : 'bs4.typing._IncomingMarkup'
#}
#from bs4 import _typing
#import typing
#for k, v in _typing.__dict__.items():
#    if type(v) in (typing._UnionGenericAlias, typing._GenericAlias):
#        autodoc_type_aliases[k] = f'bs4._typing_{k}'


intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
#html_static_path = ['_static']

default_role = 'any'
