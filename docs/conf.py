# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'pypolar'
copyright = '2020-21, Scott Prahl'
author = 'Scott Prahl'

# The full version, including alpha/beta/rc tags
release = '0.8.2'

master_doc = 'index'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'nbsphinx',
]
numpydoc_show_class_members = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_custom_sections = [('Returns', 'params_style')]

# List of patterns, relative to source directory, to exclude
exclude_patterns = ['_build', 
                    '**.ipynb_checkpoints',
                    '.DS_Store',
                    'omlc.org',
                    'refractiveindex.info',
                    '10d-Ellipsometry.ipynb',
                    'pandas-tut.ipynb',
                    ]

# I execute the notebooks manually in advance.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_scaled_image_link = False
html_sourcelink_suffix = ''

exclude_patterns = ['_build', '**.ipynb_checkpoints']
