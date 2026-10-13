from colabutilscs4337 import env
from colabutilscs4337.env import _check


def test_tested_runtime_is_packaged():
    tested = env.tested_runtime()
    assert tested["python"]
    assert "torch" in tested["packages"]


def test_public_version():
    assert _check.public_version("2.11.0+cu130") == "2.11.0"
    assert _check.public_version(None) is None


def test_check_statuses(monkeypatch, capsys):
    tested = {"python": "3.13.15", "packages": {"torch": "2.11.0+cu130", "fastai": "2.8.12"}}
    installed = {"torch": "2.11.0+cpu", "fastai": "2.8.11", "netron": "9.3.1"}
    monkeypatch.setattr(_check, "tested_runtime", lambda: tested)
    monkeypatch.setattr(_check, "installed_version", installed.get)

    rows = {r.package: r for r in env.check()}
    assert rows["torch"].status == "ok"
    assert rows["fastai"].status == "different"
    assert rows["numpy"].status == "missing"
    assert rows["netron"].status == "ok"
    assert rows["netron"].source == "installed by colabutilscs4337"
    assert rows["torchinfo"].status == "missing"

    out = capsys.readouterr().out
    assert "differ from the tested Colab runtime" in out
    assert "cu.install('all')" in out


def test_runtime_info(local):
    info = env.runtime_info()
    assert info["is_colab"] is False
    assert set(info["packages"]) >= {"torch", "fastai", "netron"}
