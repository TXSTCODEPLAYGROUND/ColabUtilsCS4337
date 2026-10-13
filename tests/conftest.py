from __future__ import annotations

from pathlib import Path

import pytest

from colabutilscs4337 import _colab

_ENV_VARS = (
    "COLAB_RELEASE_TAG",
    "COLABUTILS_NOTEBOOK_NAME",
    "COLABUTILS_LOCAL_DRIVE",
    "COLABUTILS_DATA_DIR",
    "COLABUTILS_RUNS_DIR",
    "COLABUTILS_TZ",
    "JPY_SESSION_NAME",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def local(monkeypatch, tmp_path) -> Path:
    """Not in Colab; the local Drive stand-in lives under tmp_path."""
    monkeypatch.setattr(_colab, "is_colab", lambda: False)
    monkeypatch.setenv("COLABUTILS_LOCAL_DRIVE", str(tmp_path / "drive"))
    return tmp_path / "drive" / "MyDrive"


class FakeColab:
    def __init__(self, root: Path):
        self.root = root
        self.mount_calls = 0
        self.served_ports: list[int] = []
        self.notebook = "Lab 1 - Intro to CNNs"

    def mount(self, force_remount: bool = False, timeout_ms: int = 0) -> Path:
        self.mount_calls += 1
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root


@pytest.fixture
def colab(monkeypatch, tmp_path) -> FakeColab:
    """Pretend to be in Colab with Drive mounted at tmp_path/content/drive/MyDrive."""
    fake = FakeColab(tmp_path / "content" / "drive" / "MyDrive")
    monkeypatch.setattr(_colab, "is_colab", lambda: True)
    monkeypatch.setattr(_colab, "drive_root", lambda: fake.root)
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: fake.root.is_dir())
    monkeypatch.setattr(_colab, "mount", fake.mount)
    monkeypatch.setattr(_colab, "notebook_name", lambda default="untitled": fake.notebook)
    monkeypatch.setattr(
        _colab, "serve_port", lambda port, height=600, path="/": fake.served_ports.append(port)
    )
    return fake
