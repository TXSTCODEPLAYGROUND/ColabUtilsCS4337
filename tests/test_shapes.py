import pytest

from colabutilscs4337 import tools

torch = pytest.importorskip("torch")
nn = torch.nn


class Merge(nn.Module):
    def forward(self, a, b, scale=None):
        return a + b, a * b


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16)),
        )
        self.merge = Merge()
        self.shared = nn.ReLU()
        self.unused = nn.Linear(2, 2)

    def forward(self, x):
        x = self.shared(self.features(x))
        s, p = self.merge(x, x, scale=torch.ones(1))
        return self.shared(s.mean((2, 3)) + p.mean((2, 3)))


class FakeLearner:
    def __init__(self, model):
        self.model = model


def test_block_by_object_and_name(capsys):
    net = Net()
    block = net.features[1]
    s = tools.layer_shapes(net, block, input_size=(2, 3, 32, 32))
    assert (s.name, s.type) == ("features.1", "Sequential")
    assert s.input == (2, 8, 32, 32) and s.output == (2, 16, 16, 16)
    assert "features.1 (Sequential): (2, 8, 32, 32) -> (2, 16, 16, 16)" in capsys.readouterr().out
    assert tools.layer_shapes(net, "features.1.0", (1, 3, 8, 8), verbose=False).output == (
        1,
        16,
        4,
        4,
    )


def test_multiple_inputs_kwargs_and_outputs():
    s = tools.layer_shapes(Net(), "merge", (1, 3, 16, 16), verbose=False)
    assert s.inputs == [(1, 16, 8, 8), (1, 16, 8, 8)]
    assert s.kwargs == {"scale": (1,)}
    assert s.outputs == [(1, 16, 8, 8), (1, 16, 8, 8)]


def test_list_of_layers_and_first_call_of_shared_module():
    out = tools.layer_shapes(Net(), ["features.0", "shared"], (1, 3, 16, 16), verbose=False)
    assert list(out) == ["features.0", "shared"]
    assert out["shared"].input == (1, 16, 8, 8)


def test_unused_layer_raises():
    with pytest.raises(ValueError, match="unused"):
        tools.layer_shapes(Net(), "unused", (1, 3, 16, 16), verbose=False)


def test_learner_real_input_and_no_side_effects():
    net = Net().train()
    bn = net.features[1][1]
    before = bn.running_mean.clone()
    x = torch.randn(1, 3, 16, 16)
    s = tools.layer_shapes(FakeLearner(net), net.features[1], x=x, verbose=False)
    assert s.output == (1, 16, 8, 8)
    assert net.training
    assert torch.equal(bn.running_mean, before)
    assert not net.features[1]._forward_pre_hooks and not net.features[1]._forward_hooks
