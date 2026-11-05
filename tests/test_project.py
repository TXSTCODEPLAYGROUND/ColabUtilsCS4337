from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import warnings

import pytest

from colabutilscs4337 import project
from colabutilscs4337.project import Project


@pytest.fixture(autouse=True)
def clean_project_env(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "MyProj" / "configs").mkdir(parents=True)
    (repo / "MyProj" / "configs" / "config01.json").write_text('{"seed": 1}')
    return repo


@pytest.fixture
def proj(repo) -> Project:
    return Project(repo / "MyProj")


def test_default_paths(proj, repo):
    assert proj.name == "MyProj"
    assert proj.data_dir() == repo / "data"
    assert proj.data_dir("a", "b") == repo / "data" / "a" / "b"
    assert proj.runs_dir() == repo / "runs" / "MyProj"


def test_env_file_sets_dirs_and_keeps_existing_environ(proj, repo, tmp_path, monkeypatch):
    (repo / ".env").write_text(f"DATA_DIR=/abs/data\nOUTPUT_DIR={tmp_path}/out\nKEEP=file\n")
    monkeypatch.setenv("KEEP", "environ")
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert proj.load_env() is True
    assert os.environ["KEEP"] == "environ"
    assert str(proj.data_dir()) == "/abs/data"
    assert proj.output_dir() == tmp_path / "out"


def test_load_env_without_file(proj):
    assert proj.load_env() is False


def test_resolve_config_variants(proj, repo):
    expected = (repo / "MyProj" / "configs" / "config01.json").resolve()
    for name in ("config01", "config01.json", "configs/config01.json"):
        assert proj.resolve_config(name) == expected
    path, config = proj.read_config("config01")
    assert path == expected and config == {"seed": 1}


def test_resolve_config_missing_lists_available(proj):
    with pytest.raises(FileNotFoundError, match=r"config01\.json"):
        proj.resolve_config("nope")


def _make_run(proj, config, run_id, files):
    run = proj.new_run_dir(config, run_id)
    for name in files:
        (run.path / name).write_text("x")
    return run.path


def test_new_run_dir_layout(proj):
    run = proj.new_run_dir("config01", "2026-01-01_00-00-00")
    assert run.path == proj.runs_dir() / "config01" / "2026-01-01_00-00-00"
    assert run.path.is_dir()
    auto = proj.new_run_dir("config01")
    assert len(auto.run_id) == len("2026-01-01_00-00-00")


def test_find_checkpoint_newest_run_and_which(proj):
    old = _make_run(proj, "config01", "2026-01-01_00-00-00", ["best_epoch01.ckpt", "last.ckpt"])
    new = _make_run(proj, "config01", "2026-01-02_00-00-00", ["best_epoch02.ckpt"])
    assert proj.find_checkpoint("config01") == new / "best_epoch02.ckpt"
    assert proj.find_checkpoint("config01/2026-01-01_00-00-00", "last") == old / "last.ckpt"
    assert proj.find_checkpoint(new / "best_epoch02.ckpt") == new / "best_epoch02.ckpt"
    with pytest.raises(ValueError):
        proj.find_checkpoint("config01", "middle")


def test_find_resume_checkpoint_prefers_last(proj):
    run = _make_run(proj, "config01", "2026-01-01_00-00-00", ["best_epoch01.ckpt", "last.ckpt"])
    assert proj.find_resume_checkpoint("config01") == run / "last.ckpt"
    run2 = _make_run(proj, "config02", "2026-01-01_00-00-00", ["best_epoch05.ckpt"])
    assert proj.find_resume_checkpoint("config02") == run2 / "best_epoch05.ckpt"


def test_resolve_run_errors(proj):
    with pytest.raises(FileNotFoundError, match="mount Google Drive"):
        proj.resolve_run("config01")
    _make_run(proj, "config01", "2026-01-01_00-00-00", ["best_epoch01.ckpt"])
    with pytest.raises(FileNotFoundError, match=r"Runs saved for this project: \['config01'\]"):
        proj.resolve_run("config09")


def test_save_run_files_and_run_config(proj):
    run = proj.new_run_dir("config01", "2026-01-01_00-00-00")
    config_path = proj.resolve_config("config01")
    ckpt = run.path / "best.ckpt"
    ckpt.write_text("x")
    with pytest.warns(UserWarning, match="did not finish"):
        assert proj.run_config(ckpt) == {"seed": 1}
    proj.save_run_files(run, config_path, {"lr": 0.1}, resumed_from="a.ckpt")
    assert json.loads((run / "hparams.json").read_text()) == {"lr": 0.1, "resumed_from": "a.ckpt"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert proj.run_config(ckpt) == {"seed": 1}


def test_append_run_summary_grows_columns(tmp_path):
    path = tmp_path / "runs_summary.csv"
    project.append_run_summary(path, {"run_id": "a", "acc": "0.1"})
    project.append_run_summary(path, {"run_id": "b", "acc": "0.2"})
    project.append_run_summary(path, {"run_id": "c", "acc": "0.3", "optimizer": "adam"})
    rows = list(csv.DictReader(path.open()))
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]
    assert [r["optimizer"] for r in rows] == ["", "", "adam"]


def test_compare_runs_gain(tmp_path):
    for config, strategy, score in [("c1", "from scratch", "0.40"), ("c2", "fine-tune", "0.55")]:
        (tmp_path / config).mkdir()
        project.append_run_summary(
            tmp_path / config / "runs_summary.csv", {"strategy": strategy, "map": score}
        )
    rows = project.compare_runs(tmp_path, "map")
    assert [r["config"] for r in rows] == ["c1", "c2"]
    assert rows[0]["gain_vs_baseline"] == 0
    assert rows[1]["gain_vs_baseline"] == pytest.approx(0.15)


def test_repo_name_from_url():
    assert project.repo_name_from_url("https://github.com/o/Repo.git") == "Repo"
    assert project.repo_name_from_url("https://github.com/o/Repo/") == "Repo"


def _git_repo(path):
    path.mkdir(parents=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    (path / ".envcolab").write_text("DATA_DIR=/content/data\n")
    (path / "requirements.txt").write_text("# none needed\n")
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=path, check=True, env=env, capture_output=True)
    return path


def test_setup_colab_outside_colab_only_updates_path(local, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    repo = project.setup_colab("https://example.invalid/R.git", local_repo=tmp_path)
    assert repo == tmp_path.resolve()
    assert str(repo) in sys.path


def test_setup_colab_clones_writes_env_and_errors_without_drive_file(colab, tmp_path, monkeypatch):
    origin = _git_repo(tmp_path / "origin" / "Course")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "content"
    repo = project.setup_colab(str(origin), base_dir=base, requirements=None, chdir=False)
    assert repo == base / "Course"
    assert (repo / ".env").read_text() == "DATA_DIR=/content/data\n"
    assert colab.mount_calls == 1

    with pytest.raises(FileNotFoundError, match="keys.env"):
        project.setup_colab(
            str(origin), base_dir=base, requirements=None, chdir=False, env_source="keys.env"
        )

    (colab.root / "keys.env").write_text("WANDB_API_KEY=secret\n")
    project.setup_colab(
        str(origin), base_dir=base, requirements=None, chdir=False, env_source="keys.env"
    )  # second call pulls instead of cloning
    assert (repo / ".env").read_text() == "DATA_DIR=/content/data\nWANDB_API_KEY=secret\n"
