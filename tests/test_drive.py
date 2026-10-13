import json
import random
import re
from zoneinfo import ZoneInfoNotFoundError

import pytest

from colabutilscs4337 import drive


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Lab 3 - CNNs (v2)", "Lab-3-CNNs-v2"),
        ("resnet34 lr=1e-3", "resnet34-lr-1e-3"),
        ("  ", "run"),
        ("...", "run"),
    ],
)
def test_slugify(text, slug):
    assert drive.slugify(text) == slug


def test_random_slug_is_deterministic_with_seed():
    assert drive.random_slug(random.Random(0)) == drive.random_slug(random.Random(0))
    assert re.fullmatch(r"[a-z]+-[a-z]+", drive.random_slug())


def test_new_run_in_colab(colab):
    run = drive.new_run("baseline")
    assert colab.mount_calls == 1
    assert run.path.parent == colab.root / "runs"
    assert re.fullmatch(r"Lab-1-Intro-to-CNNs_\d{8}-\d{6}_baseline", run.name)
    info = json.loads((run / "run.json").read_text())
    assert info["notebook"] == colab.notebook
    assert info["slug"] == "baseline"
    assert "packages" in info["runtime"]


def test_new_run_random_slug_and_unique(colab):
    first = drive.new_run("same", notebook="nb")
    second = drive.new_run("same", notebook="nb")
    assert first.path != second.path
    assert drive.new_run(notebook="nb").slug


def test_run_is_pathlike(colab):
    run = drive.new_run("p", notebook="nb")
    with open(run / "log.txt", "w") as f:
        f.write("ok")
    assert (run.path / "log.txt").read_text() == "ok"
    assert run.subdir("checkpoints").is_dir()


def test_list_and_latest_runs(colab):
    runs_dir = colab.root / "runs"
    for name in ["nb_20260101-000000_a", "nb_20260301-000000_b", "other_20260201-000000_c"]:
        (runs_dir / name).mkdir(parents=True)
    assert [p.name for p in drive.list_runs("nb")] == [
        "nb_20260101-000000_a",
        "nb_20260301-000000_b",
    ]
    assert drive.latest_run().name == "nb_20260301-000000_b"


def test_default_config_in_colab(colab):
    cfg = drive.get_config()
    assert cfg.data_dir == "/content/data"
    assert cfg.runs_path == colab.root / "runs"
    assert cfg.timezone == "UTC"


def test_config_precedence(colab, monkeypatch, tmp_path):
    drive.set_config(data_dir=tmp_path / "saved", timezone="America/Chicago")
    assert drive.config_path().is_file()
    assert drive.get_config().data_dir == str(tmp_path / "saved")

    monkeypatch.setenv("COLABUTILS_DATA_DIR", str(tmp_path / "env"))
    assert drive.get_config().data_dir == str(tmp_path / "env")
    assert drive.get_config(data_dir="/explicit").data_dir == "/explicit"
    assert drive.get_config().timezone == "America/Chicago"

    drive.reset_config()
    monkeypatch.delenv("COLABUTILS_DATA_DIR")
    assert drive.get_config().data_dir == "/content/data"


def test_set_config_rejects_bad_timezone(colab):
    with pytest.raises(ZoneInfoNotFoundError):
        drive.set_config(timezone="Not/AZone")


def test_data_dir_creates_folder(local, tmp_path):
    path = drive.data_dir("pets")
    assert path == tmp_path / "data" / "pets"
    assert path.is_dir()


def test_new_run_outside_colab(local):
    with pytest.warns(UserWarning):
        run = drive.new_run("x", notebook="nb")
    assert run.path.parent == local / "runs"
