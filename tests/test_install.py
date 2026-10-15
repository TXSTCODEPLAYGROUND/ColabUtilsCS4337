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
    tools = resolve_targets(["tools"])
    assert [requirement_name(r) for r in tools] == ["torchinfo", "netron", "onnx", "onnxscript"]
    assert resolve_targets(["all", "tools"]) == [*tools, "grad-cam>=1.5"]
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


def _fake_env(monkeypatch, before, after, plan, uv=None):
    snapshots = iter([before, after])
    calls = []
    monkeypatch.setattr(_install, "_uv_path", lambda: uv)
    monkeypatch.setattr(_install, "snapshot", lambda: next(snapshots))
    monkeypatch.setattr(_install, "_pip_check_problems", lambda: set())
    monkeypatch.setattr(_install, "dry_run", lambda *a, **k: plan)
    monkeypatch.setattr(_install, "_pip", lambda *args, **k: calls.append(("pip", *args)))
    monkeypatch.setattr(_install, "_uv", lambda *args, **k: calls.append(("uv", *args)))
    return calls


UV_DRY_RUN = """Using Python 3.13.15 environment at: /usr
Resolved 12 packages in 85ms
Would download 2 packages
Would uninstall 1 package
Would install 3 packages
 + grad-cam==1.5.7
 - numpy==2.1.3
 + numpy==2.3.0
 + ttach==0.0.3
 + torch==2.11.0+cu130
"""


def test_parse_uv_dry_run():
    plan = _install.parse_uv_dry_run(UV_DRY_RUN, {"numpy": "2.1.3", "torch": "2.11.0+cu130"})
    assert plan.added == {"grad-cam": "1.5.7", "ttach": "0.0.3"}
    assert plan.changed == {"numpy": ("2.1.3", "2.3.0"), "torch": ("2.11.0+cu130", "2.11.0+cu130")}
    assert _install.parse_uv_dry_run("Would make no changes\n", {}).is_noop


def test_choose_backend(monkeypatch):
    monkeypatch.setattr(_install, "_uv_path", lambda: "/usr/local/bin/uv")
    assert _install.choose_backend() == "uv"
    assert _install.choose_backend("pip") == "pip"
    monkeypatch.setattr(_install, "_uv_path", lambda: None)
    assert _install.choose_backend("auto") == "pip"
    with pytest.raises(InstallError, match="uv is not installed"):
        _install.choose_backend("uv")
    with pytest.raises(ValueError):
        _install.choose_backend("conda")


def test_uv_upgrade_is_limited_to_requested_packages():
    args = _install._install_args(["grad-cam>=1.5"], None, upgrade=True, backend="uv")
    assert args == ["install", "--upgrade-package", "grad-cam", "grad-cam>=1.5"]
    assert "--upgrade" in _install._install_args(["x"], None, upgrade=True)


def test_install_uses_uv_when_available(monkeypatch):
    before = {"torch": "2.11.0"}
    after = {"torch": "2.11.0", "grad-cam": "1.5.7"}
    calls = _fake_env(
        monkeypatch, before, after, Plan(added={"grad-cam": "1.5.7"}), uv="/usr/local/bin/uv"
    )
    result = _install.install("cam", verbose=False)
    assert result.backend == "uv"
    assert result.added == {"grad-cam": "1.5.7"}
    assert len(calls) == 1 and calls[0][0] == "uv" and "-c" in calls[0]


def test_install_falls_back_to_pip_when_uv_fails(monkeypatch):
    calls = _fake_env(monkeypatch, {}, {"netron": "9.3.1"}, None, uv="/usr/local/bin/uv")
    plan = Plan(added={"netron": "9.3.1"})

    def fake_dry_run(*args, backend="pip", **kwargs):
        if backend == "uv":
            raise InstallError("uv exploded")
        return plan

    monkeypatch.setattr(_install, "dry_run", fake_dry_run)
    with pytest.warns(UserWarning, match="retrying with pip"):
        result = _install.install("netron", verbose=False)
    assert result.backend == "pip" and calls[0][0] == "pip"
    monkeypatch.setattr(_install, "snapshot", lambda: {})
    with pytest.raises(InstallError, match="uv exploded"):
        _install.install("netron", backend="uv", verbose=False)


def test_unresolvable_with_uv_does_not_retry_with_pip(monkeypatch, recwarn):
    _fake_env(monkeypatch, {}, {}, None, uv="/usr/local/bin/uv")
    tried = []

    def fake_dry_run(*args, backend="pip", **kwargs):
        tried.append(backend)
        raise _install.UnresolvableError("No solution found")

    monkeypatch.setattr(_install, "dry_run", fake_dry_run)
    with pytest.raises(_install.UnresolvableError):
        _install.install("onnx==1.23.2", verbose=False)
    assert tried == ["uv"] and not recwarn


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
