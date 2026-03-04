import os
import sys

# Add the project root to sys.path so autodoc can import the modules
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'COMP532 Assignment 1 — Multi-Armed Bandit'
copyright = '2026, Zhiheng Wang'
author = 'Zhiheng Wang'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
	'sphinx.ext.inheritance_diagram',
	'sphinx_autodoc_typehints',
	'sphinxcontrib.plantuml',
]

# PlantUML executable
plantuml = 'plantuml'
plantuml_output_format = 'png'

# autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
add_module_names = False

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = False
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output (not used, but required) -----------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for LaTeX / PDF output -----------------------------------------
latex_engine = 'pdflatex'

latex_elements = {
	'papersize': 'a4paper',
	'pointsize': '11pt',
	'preamble' : r'''
\usepackage{charter}
\usepackage[defaultsans]{lato}
\usepackage{inconsolata}
''',
}

latex_documents = [
	(
		'index',  # source start file
		'COMP532_Assignment1.tex',  # target filename
		'COMP532 Assignment 1\\\\Multi-Armed Bandit',  # title
		'Zhiheng Wang',  # author
		'manual',  # document class
	),
]
