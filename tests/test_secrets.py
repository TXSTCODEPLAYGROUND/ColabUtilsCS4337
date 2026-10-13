import os

import pytest

from colabutilscs4337 import _colab, env

NAMES = ("CU_TEST_TOKEN", "CU_TEST_KEY", "CU_TEST_USER")


@pytest.fixture(autouse=True)
def restore_environ(local):
    saved = dict(os.environ)
    for name in NAMES:
        os.environ.pop(name, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_read_env_file(tmp_path):
    path = tmp_path / "x.env"
    path.write_text(
        "# comment\n"
        "export CU_TEST_TOKEN='abc 123'\n"
        'CU_TEST_KEY="k=v"\n'
        "CU_TEST_USER=alice  # trailing comment\n"
        "not a pair\n"
    )
    assert env.read_env_file(path) == {
        "CU_TEST_TOKEN": "abc 123",
        "CU_TEST_KEY": "k=v",
        "CU_TEST_USER": "alice",
    }


def test_sources_in_order(monkeypatch, tmp_path, capsys):
    os.environ["CU_TEST_TOKEN"] = "from-env"
    monkeypatch.setattr(
        _colab, "colab_secret", lambda name: "from-colab" if name == "CU_TEST_KEY" else None
    )
    (tmp_path / ".env").write_text("CU_TEST_KEY=from-file\nCU_TEST_USER=from-file\n")

    result = env.load_secrets(*NAMES)
    assert result == {
        "CU_TEST_TOKEN": "environment",
        "CU_TEST_KEY": "Colab Secrets",
        "CU_TEST_USER": str(tmp_path / ".env"),
    }
    assert os.environ["CU_TEST_KEY"] == "from-colab"
    assert os.environ["CU_TEST_USER"] == "from-file"
    out = capsys.readouterr().out
    assert "from-colab" not in out and "from-file" not in out


def test_load_all_from_env_file(tmp_path):
    path = tmp_path / "keys.env"
    path.write_text("CU_TEST_TOKEN=t\nCU_TEST_KEY=k\n")
    result = env.load_secrets(env_file=path, verbose=False)
    assert set(result) == {"CU_TEST_TOKEN", "CU_TEST_KEY"}
    assert os.environ["CU_TEST_KEY"] == "k"


def test_drive_env_file_outside_colab(local):
    path = local / ".colabutilscs4337" / ".env"
    path.parent.mkdir(parents=True)
    path.write_text("CU_TEST_TOKEN=drive\n")
    assert env.get_secret("CU_TEST_TOKEN") == "drive"


def test_missing_required_raises():
    with pytest.raises(env.MissingSecretError, match="key icon"):
        env.load_secrets("CU_TEST_TOKEN", required=True, verbose=False)


def test_missing_not_required_reports(capsys):
    assert env.load_secrets("CU_TEST_TOKEN") == {"CU_TEST_TOKEN": None}
    assert "Missing: CU_TEST_TOKEN" in capsys.readouterr().out
    assert "CU_TEST_TOKEN" not in os.environ


def test_overwrite_prefers_secret_store(monkeypatch):
    os.environ["CU_TEST_TOKEN"] = "old"
    monkeypatch.setattr(_colab, "colab_secret", lambda name: "new")
    env.load_secrets("CU_TEST_TOKEN", overwrite=True, verbose=False)
    assert os.environ["CU_TEST_TOKEN"] == "new"


def test_notebook_access_hint(monkeypatch, capsys):
    def no_access(name):
        raise _colab.SecretAccessError(name)

    monkeypatch.setattr(_colab, "colab_secret", no_access)
    assert env.load_secrets("CU_TEST_TOKEN") == {"CU_TEST_TOKEN": None}
    assert "Notebook access" in capsys.readouterr().out


def test_prompt(monkeypatch):
    monkeypatch.setattr("getpass.getpass", lambda prompt: "typed")
    assert env.load_secrets("CU_TEST_TOKEN", prompt=True, verbose=False) == {
        "CU_TEST_TOKEN": "prompt"
    }
    assert os.environ["CU_TEST_TOKEN"] == "typed"


def test_get_secret_default_and_no_side_effect():
    assert env.get_secret("CU_TEST_TOKEN", default="dflt") == "dflt"
    assert "CU_TEST_TOKEN" not in os.environ
