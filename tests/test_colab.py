import os
import subprocess
import sys
from pathlib import Path

import pytest

import colabutilscs4337
from colabutilscs4337 import _colab


def test_import_is_light():
    code = (
        "import sys, colabutilscs4337 as cu; cu.env; cu.drive; cu.tools; cu.models; cu.viz; "
        "heavy = [m for m in ('torch', 'fastai', 'numpy', 'matplotlib') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    src = str(Path(colabutilscs4337.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == ""


def test_clean_name():
    assert _colab._clean_name("/content/Lab 1.ipynb") == "Lab 1"
    assert _colab._clean_name("") is None
    assert _colab._clean_name(None) is None


def test_notebook_name_env_override(monkeypatch):
    monkeypatch.setenv(_colab.NOTEBOOK_NAME_ENV, "My Lab.ipynb")
    assert _colab.notebook_name() == "My Lab"


def test_notebook_name_colab_sources(monkeypatch):
    monkeypatch.setattr(_colab, "is_colab", lambda: True)

    def broken():
        raise RuntimeError("frontend unavailable")

    monkeypatch.setattr(_colab, "_name_from_colab_frontend", broken)
    monkeypatch.setattr(_colab, "_name_from_sessions_api", lambda: "From Sessions")
    assert _colab.notebook_name() == "From Sessions"


def test_notebook_name_default(monkeypatch):
    monkeypatch.setattr(_colab, "is_colab", lambda: False)
    monkeypatch.setattr(_colab, "_name_from_local_jupyter", lambda: None)
    assert _colab.notebook_name() == "untitled"
    assert _colab.notebook_name(default="nb") == "nb"


def test_mount_outside_colab_uses_local_folder(local):
    with pytest.warns(UserWarning):
        root = _colab.mount()
    assert root == local and root.is_dir()
