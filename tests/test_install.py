import pytest

from colabutilscs4337 import _install
from colabutilscs4337._install import (
    InstallError,
    InstallResult,
    Plan,
    constraints_text,
    parse_report,
    requirement_name,
    resolve_targets,
)


def test_resolve_targets_expands_extras():
    assert resolve_targets(["tools"]) == ["torchinfo>=1.8", "netron>=7.0"]
    assert resolve_targets(["all", "tools"]) == ["torchinfo>=1.8", "netron>=7.0", "grad-cam>=1.5"]
    assert resolve_targets(["rich==13.0"]) == ["rich==13.0"]


@pytest.mark.parametrize(
    ("req", "name"),
    [("grad-cam>=1.5", "grad-cam"), ("Pillow", "pillow"), ("opencv_python==5", "opencv-python")],
)
def test_requirement_name(req, name):
    assert requirement_name(req) == name


def test_constraints_text_pins_all_but_requested():
    installed = {
        "torch": "2.11.0+cu130",
        "numpy": "2.1.3",
        "netron": "8.0",
        "colabutilscs4337": "0.1",
    }
    text = constraints_text(installed, exclude=["netron"])
    assert "torch==2.11.0+cu130\n" in text
    assert "numpy==2.1.3\n" in text
    assert "netron" not in text
    assert "colabutilscs4337" not in text


def test_parse_report_splits_added_and_changed():
    report = {
        "install": [
            {"metadata": {"name": "ttach", "version": "0.0.3"}},
            {"metadata": {"name": "NumPy", "version": "2.3.0"}},
        ]
    }
    plan = parse_report(report, {"numpy": "2.1.3"})
    assert plan.added == {"ttach": "0.0.3"}
    assert plan.changed == {"numpy": ("2.1.3", "2.3.0")}


def _fake_env(monkeypatch, before, after, plan):
    snapshots = iter([before, after])
    calls = []
    monkeypatch.setattr(_install, "snapshot", lambda: next(snapshots))
    monkeypatch.setattr(_install, "_pip_check_problems", lambda: set())
    monkeypatch.setattr(_install, "dry_run", lambda *a, **k: plan)
    monkeypatch.setattr(_install, "_pip", lambda *args, **k: calls.append(args))
    return calls


def test_install_adds_packages(monkeypatch):
    before = {"torch": "2.11.0"}
    after = {"torch": "2.11.0", "torchinfo": "1.8.0", "netron": "9.3.1"}
    calls = _fake_env(
        monkeypatch, before, after, Plan(added={"torchinfo": "1.8.0", "netron": "9.3.1"})
    )
    result = _install.install("tools", verbose=False)
    assert result.added == {"torchinfo": "1.8.0", "netron": "9.3.1"}
    assert result.changed == {}
    assert len(calls) == 1 and "-c" in calls[0]


def test_install_refuses_to_change_preinstalled(monkeypatch):
    plan = Plan(added={"grad-cam": "1.5.7"}, changed={"torch": ("2.11.0", "2.12.0")})
    calls = _fake_env(monkeypatch, {"torch": "2.11.0"}, {}, plan)
    with pytest.raises(InstallError, match="torch 2.11.0 -> 2.12.0"):
        _install.install("cam", verbose=False)
    assert calls == []


def test_install_refuses_explicit_colab_package_change(monkeypatch):
    plan = Plan(changed={"numpy": ("2.1.3", "2.3.0")})
    _fake_env(monkeypatch, {"numpy": "2.1.3"}, {}, plan)
    with pytest.raises(InstallError, match="numpy 2.1.3 -> 2.3.0"):
        _install.install("numpy==2.3.0", verbose=False)
    _fake_env(monkeypatch, {"numpy": "2.1.3"}, {}, plan)
    result = _install.install("numpy==2.3.0", allow_changes=True, dry=True, verbose=False)
    assert result.changed == {"numpy": ("2.1.3", "2.3.0")}


def test_install_allows_upgrading_requested_extra(monkeypatch):
    plan = Plan(changed={"netron": ("8.0", "9.3.1")})
    _fake_env(monkeypatch, {"netron": "8.0"}, {}, plan)
    result = _install.install("netron", upgrade=True, dry=True, verbose=False)
    assert result.changed == {"netron": ("8.0", "9.3.1")}


def test_install_dry_does_not_install(monkeypatch):
    calls = _fake_env(monkeypatch, {}, {}, Plan(added={"netron": "9.3.1"}))
    result = _install.install("netron", dry=True, verbose=False)
    assert result.added == {"netron": "9.3.1"}
    assert calls == []


def test_install_result_str():
    result = InstallResult(["netron"], {"netron": "9.3.1"}, {}, [], restart_needed=True)
    text = str(result)
    assert "Added: netron 9.3.1" in text
    assert "Restart the session" in text
