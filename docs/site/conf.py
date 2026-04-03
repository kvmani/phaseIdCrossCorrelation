from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

project = "phaseIdCrossCorrelation"
author = "phaseIdCrossCorrelation contributors"
copyright = "2026, phaseIdCrossCorrelation contributors"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "phaseIdCrossCorrelation Documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}

html_logo = None
html_favicon = None
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 4
myst_url_schemes = ("http", "https", "mailto")

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

mermaid_version = "11.9.0"

nitpicky = False
