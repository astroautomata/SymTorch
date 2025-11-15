# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'SymTorch'
author = 'Liz Tan'
release = '1.1.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"

language = 'en'


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "_static/symtorch_favicon.png"
html_favicon = "_static/symtorch_favicon.png"   # optional
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
intersphinx_timeout = 10  # Timeout in seconds (default is 30s)

# Type hints configuration
always_document_param_types = True
typehints_use_signature = True
typehints_use_signature_return = True

# MathJax configuration - MyST-NB handles MathJax automatically
# We only need to ensure the proper delimiters are configured
myst_enable_extensions = ["dollarmath"]
