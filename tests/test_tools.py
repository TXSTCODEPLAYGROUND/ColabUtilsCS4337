import pytest

from colabutilscs4337 import tools
from colabutilscs4337._compat import MissingDependencyError, require

torch = pytest.importorskip("torch")
nn = torch.nn


def tiny_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


class FakeLearner:
    def __init__(self, model):
        self.model = model


def test_require_missing_module_message():
    with pytest.raises(MissingDependencyError, match=r"cu.install\('tools'\)"):
        require("definitely_not_a_module_xyz", extra="tools")


def test_unwrap_model():
    model = tiny_cnn()
    assert tools.unwrap_model(model) is model
    assert tools.unwrap_model(FakeLearner(model)) is model
    with pytest.raises(TypeError):
        tools.unwrap_model(object())


def test_summary_counts_params():
    pytest.importorskip("torchinfo")
    stats = tools.summary(FakeLearner(tiny_cnn()), input_size=(1, 3, 16, 16), verbose=0)
    assert stats.total_params == sum(p.numel() for p in tiny_cnn().parameters())


def test_export_for_netron_torchscript(tmp_path):
    path = tools.export_for_netron(tiny_cnn(), tmp_path / "m.pt", input_size=(1, 3, 16, 16))
    assert path.is_file()
    loaded = torch.jit.load(str(path))
    assert loaded(torch.zeros(1, 3, 16, 16)).shape == (1, 2)


def test_export_for_netron_torch_export(tmp_path):
    path = tools.export_for_netron(tiny_cnn(), input_size=(1, 3, 16, 16), format="export")
    assert path.suffix == ".pt2" and path.is_file()
    with pytest.raises(ValueError):
        tools.export_for_netron(tiny_cnn(), format="keras")


def test_netron_serves_through_colab_proxy(colab, monkeypatch, tmp_path):
    netron = pytest.importorskip("netron")
    started, stopped = [], []
    monkeypatch.setattr(
        netron,
        "start",
        lambda file, address=None, browse=True: started.append(file) or ("localhost", 8081),
    )
    monkeypatch.setattr(netron, "stop", lambda address: stopped.append(address))

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"x")
    assert tools.netron(model_file) == ("localhost", 8081)
    assert colab.served_ports == [8081]

    tools.netron(model_file)
    assert stopped == [("localhost", 8081)]
    tools.stop_netron()
    assert len(stopped) == 2


def test_netron_missing_file():
    pytest.importorskip("netron")
    with pytest.raises(FileNotFoundError):
        tools.netron("does-not-exist.pt")
