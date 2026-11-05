from __future__ import annotations

import os
import stat

import pytest

from colabutilscs4337 import _colab, env


@pytest.fixture(autouse=True)
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_copies_relative_to_my_drive(colab, tmp_path):
    (colab.root / "keys").mkdir(parents=True)
    (colab.root / "keys" / "cs.env").write_text("A_KEY=1\n")
    dest = env.sync_env_file("keys/cs.env", dest=tmp_path / "repo" / ".env", verbose=False)
    assert dest.read_text() == "A_KEY=1\n"
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


def test_default_source_is_the_colabutils_env(colab, tmp_path):
    (colab.root / ".colabutilscs4337").mkdir(parents=True)
    (colab.root / ".colabutilscs4337" / ".env").write_text("B_KEY=2\n")
    dest = env.sync_env_file(dest=tmp_path / ".env", verbose=False)
    assert dest.read_text() == "B_KEY=2\n"


def test_missing_file_raises_with_path(colab, tmp_path):
    colab.root.mkdir(parents=True)
    with pytest.raises(env.EnvFileError, match="nope.env"):
        env.sync_env_file("nope.env", dest=tmp_path / ".env")
    assert not (tmp_path / ".env").exists()


def test_not_mounted_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(_colab, "is_colab", lambda: True)
    monkeypatch.setattr(_colab, "drive_root", lambda: tmp_path / "drive" / "MyDrive")
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: False)
    monkeypatch.setattr(_colab, "mount", lambda **kw: tmp_path)
    with pytest.raises(env.EnvFileError, match="not mounted"):
        env.sync_env_file("x.env", dest=tmp_path / ".env")


def test_mounts_when_needed(monkeypatch, tmp_path):
    root = tmp_path / "drive" / "MyDrive"
    state = {"mounted": False, "calls": 0}

    def mount(**kw):
        state["mounted"] = True
        state["calls"] += 1
        root.mkdir(parents=True)
        (root / "x.env").write_text("C=3\n")
        return root

    monkeypatch.setattr(_colab, "is_colab", lambda: True)
    monkeypatch.setattr(_colab, "drive_root", lambda: root)
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: state["mounted"])
    monkeypatch.setattr(_colab, "mount", mount)
    env.sync_env_file("x.env", dest=tmp_path / ".env", verbose=False)
    assert state["calls"] == 1


def test_append_and_load(colab, tmp_path, monkeypatch):
    colab.root.mkdir(parents=True)
    (colab.root / "k.env").write_text("CU_DRIVE_KEY=from-drive\n")
    dest = tmp_path / ".env"
    dest.write_text("DATA_DIR=/content/data")  # no trailing newline
    monkeypatch.delenv("CU_DRIVE_KEY", raising=False)
    monkeypatch.setattr(_colab, "colab_secret", lambda name: None)
    env.sync_env_file("k.env", dest=dest, append=True, load=True, verbose=False)
    assert dest.read_text() == "DATA_DIR=/content/data\nCU_DRIVE_KEY=from-drive\n"
    assert os.environ["CU_DRIVE_KEY"] == "from-drive"


def test_load_secrets_persist_to(local, tmp_path, monkeypatch):
    monkeypatch.delenv("CU_P_KEY", raising=False)
    monkeypatch.delenv("CU_P_OTHER", raising=False)
    monkeypatch.setattr(_colab, "colab_secret", lambda name: "s3" if name == "CU_P_KEY" else None)
    target = tmp_path / "repo" / ".env"
    env.load_secrets("CU_P_KEY", "CU_P_OTHER", persist_to=target, verbose=False)
    assert target.read_text() == "CU_P_KEY=s3\n"
    monkeypatch.delenv("CU_P_KEY")
    env.load_secrets("CU_P_KEY", persist_to=target, verbose=False)
    assert target.read_text() == "CU_P_KEY=s3\n"  # not duplicated


def test_ssh_session_messages_point_to_the_cli(monkeypatch, tmp_path):
    """No notebook in an SSH session: say how to mount and where the file goes."""
    root = tmp_path / "drive" / "MyDrive"
    monkeypatch.setattr(_colab, "is_colab", lambda: True)
    monkeypatch.setattr(_colab, "is_notebook", lambda: False)
    monkeypatch.setattr(_colab, "drive_root", lambda: root)
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: False)
    with pytest.raises(env.EnvFileError, match="colab drivemount"):
        env.sync_env_file("x.env", dest=tmp_path / ".env", mount=False)

    root.mkdir(parents=True)
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: True)
    with pytest.raises(env.EnvFileError, match=r"Create it on Drive.*x\.env"):
        env.sync_env_file("x.env", dest=tmp_path / ".env")


def test_mount_in_ssh_session_explains_drivemount(monkeypatch):
    monkeypatch.setattr(_colab, "is_colab", lambda: True)
    monkeypatch.setattr(_colab, "is_notebook", lambda: False)
    monkeypatch.setattr(_colab, "is_drive_mounted", lambda: False)
    with pytest.raises(RuntimeError, match="colab drivemount"):
        _colab.mount()
