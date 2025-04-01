# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hestonpy'
copyright = '2025, Théophile Schmutz'
author = 'Théophile Schmutz'
release = 'v0.5.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Active le support des fichiers Markdown
    "sphinx_design", # Pour les grilles et les cartes
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    '_static/css/custom.css',
    'css/custom.css',
]

html_theme_options = {
   "logo": {
      "image_light": "_static/logo-light.png",
      "image_dark": "_static/logo-light.png",
   },

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SarcasticMatrix/hestonpy",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],

    "show_toc_level": 3,
    "show_nav_level": 2,
    "navigation_depth": 2,

    "pygments_light_style": "xcode",
    "pygments_dark_style": "lightbulb"
}