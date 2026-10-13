"""Sphinx configuration for colabutilscs4337.

The package imports torch, fastai, and matplotlib only inside functions, so
``sphinx-build`` needs nothing but ``docs/requirements.txt``.
"""

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GITHUB_USER = "TXSTCODEPLAYGROUND"
GITHUB_REPO = "ColabUtilsCS4337"
PYPI_NAME = "colabutilscs4337"

project = PYPI_NAME
author = "CS4337 Course Staff"
copyright = "2026, CS4337 Course Staff"
try:
    release = metadata.version(PYPI_NAME)
except metadata.PackageNotFoundError:
    from colabutilscs4337 import __version__ as release
version = release.split("+")[0]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = f"{project} {version}"
html_baseurl = f"https://{GITHUB_USER.lower()}.github.io/{GITHUB_REPO}/"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}

# "Edit on GitHub" link at the top right of every page; "blob" opens the file for
# reading, not GitHub's editor.
html_context = {
    "display_github": True,
    "github_user": GITHUB_USER,
    "github_repo": GITHUB_REPO,
    "github_version": "main",
    "conf_py_path": "/docs/",
    "vcs_pageview_mode": "blob",
}

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "class"

# The docstrings use both Google ("Args:") and NumPy ("Parameters\n----------") styles.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

myst_heading_anchors = 3
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
