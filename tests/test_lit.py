from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("lightning")
torch = pytest.importorskip("torch")

from colabutilscs4337 import lit  # noqa: E402
from colabutilscs4337.project import Project  # noqa: E402


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    (tmp_path / "repo" / "P" / "configs").mkdir(parents=True)
    (tmp_path / "repo" / "P" / "configs" / "config01.json").write_text(
        json.dumps({"seed": 1, "training": {"epochs": 3}})
    )
    return Project(tmp_path / "repo" / "P")


def test_start_run_fresh(proj):
    run = lit.start_run(proj, "config01.json")
    assert run.epochs == 3
    assert run.resume_checkpoint is None
    assert run.run_dir.parent == proj.runs_dir() / "config01"
    assert run.run_dir.is_dir()
    assert run.csv_logger.log_dir == str(run.run_dir)
    assert run.checkpoint.dirpath == str(run.run_dir)
    assert run.last_checkpoint.filename == "last"
    assert run.checkpoint.filename == "best_epoch{epoch:02d}_valacc{val_acc:.4f}"

    run.save({"lr": 0.1})
    assert (run.run_dir / "config.json").is_file()
    assert json.loads((run.run_dir / "hparams.json").read_text()) == {"lr": 0.1}
    run.append_summary({"run_id": run.run_id, "acc": "0.9"})
    assert (run.run_dir.parent / "runs_summary.csv").is_file()


def test_start_run_resume_adds_completed_epochs(proj):
    earlier = proj.new_run_dir("config01", "2026-01-01_00-00-00")
    torch.save({"epoch": 4}, earlier.path / "last.ckpt")
    (earlier.path / "best_epoch02_valacc0.9000.ckpt").write_text("x")
    run = lit.start_run(proj, "config01", resume_from="config01")
    assert run.epochs == 3 + 5
    assert run.resume_checkpoint == (earlier.path / "last.ckpt").resolve()
    run.save({})
    hparams = json.loads((run.run_dir / "hparams.json").read_text())
    assert hparams["resumed_from"] == str(run.resume_checkpoint)


def test_start_run_reads_cli(proj, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--config", "config01"])
    run = lit.start_run(proj, description="Train a thing.\nMore text.")
    assert run.config_name == "config01"


def test_resumable_early_stopping_restores_only_best_score():
    stopper = lit.ResumableEarlyStopping(monitor="val_acc", mode="max", patience=2)
    stopper.wait_count = 5
    stopper.load_state_dict({"best_score": torch.tensor(0.9), "wait_count": 5})
    assert stopper.best_score == torch.tensor(0.9)
    assert stopper.wait_count == 0


def test_wandb_offline_without_key(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_SILENT", "true")
    with pytest.warns(UserWarning, match="offline"):
        run = lit.connect_wandb_run("cu-test", "n", "g", tmp_path, {"a": 1})
    assert run.settings.mode == "offline"
    wandb.finish()


def test_litlogger_needs_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("LIGHTNING_USER_ID", raising=False)
    monkeypatch.delenv("LIGHTNING_API_KEY", raising=False)
    pytest.importorskip("litlogger")
    with pytest.warns(UserWarning, match="LIGHTNING_USER_ID and LIGHTNING_API_KEY"):
        assert lit.connect_litlogger(tmp_path, "x") is None
