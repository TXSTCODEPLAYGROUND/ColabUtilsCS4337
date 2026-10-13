import pytest

from colabutilscs4337 import models

torch = pytest.importorskip("torch")
nn = torch.nn


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

    def forward(self, x):
        return self.head(self.features(x))


@pytest.fixture
def net():
    torch.manual_seed(0)
    return TinyNet()


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k].cpu()) for k in sa)


def test_state_dict_needs_arch(tmp_path, net):
    path = tmp_path / "w.pth"
    torch.save(net.state_dict(), path)
    with pytest.raises(ValueError, match="arch"):
        models.load_model(path, device="cpu")
    loaded = models.load_model(path, arch=TinyNet, device="cpu")
    assert same_weights(net, loaded)
    assert not loaded.training


def test_fastai_style_checkpoint_and_dataparallel_prefix(tmp_path, net):
    path = tmp_path / "fastai.pth"
    state = {f"module.{k}": v for k, v in net.state_dict().items()}
    torch.save({"model": state, "opt": {"state": {}}}, path)
    loaded = models.load_model(path, arch=TinyNet(), device="cpu")
    assert same_weights(net, loaded)


def test_full_model_falls_back_with_warning(tmp_path, net):
    path = tmp_path / "full.pt"
    torch.save(net, path)
    with pytest.warns(UserWarning, match="trust"):
        loaded = models.load_model(path, device="cpu")
    assert isinstance(loaded, TinyNet)


def test_torchscript(tmp_path, net):
    path = tmp_path / "scripted.pt"
    with pytest.warns(DeprecationWarning):
        torch.jit.trace(net.eval(), torch.zeros(1, 3, 8, 8)).save(str(path))
    assert models.detect_format(path) == "torchscript"
    loaded = models.load_model(path, device="cpu")
    x = torch.randn(1, 3, 8, 8)
    assert torch.allclose(loaded(x), net(x))


def test_torch_export(tmp_path, net):
    path = tmp_path / "exported.pt2"
    torch.export.save(torch.export.export(net.eval(), (torch.zeros(1, 3, 8, 8),)), str(path))
    loaded = models.load_model(path, device="cpu")
    x = torch.randn(1, 3, 8, 8)
    assert torch.allclose(loaded(x), net(x))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        models.load_model("nope.pth")


def test_fastai_export(tmp_path):
    vision = pytest.importorskip("fastai.vision.all")
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "imgs"
    for label in ("cat", "dog"):
        (img_dir / label).mkdir(parents=True)
        for i in range(4):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / label / f"{i}.png")
    dls = vision.ImageDataLoaders.from_folder(
        img_dir,
        valid_pct=0.25,
        seed=0,
        item_tfms=vision.Resize(32),
        bs=2,
        num_workers=0,
        device=torch.device("cpu"),
    )
    learn = vision.Learner(dls, TinyNet(), loss_func=vision.CrossEntropyLossFlat())
    learn.export(tmp_path / "export.pkl")

    loaded = models.load_model(tmp_path / "export.pkl", device="cpu")
    assert isinstance(loaded, TinyNet)
    assert same_weights(learn.model, loaded)
    learner = models.load_model(tmp_path / "export.pkl", device="cpu", learner=True)
    assert hasattr(learner, "dls")

    learn.save("ckpt")
    ckpt = learn.path / learn.model_dir / "ckpt.pth"
    assert same_weights(learn.model, models.load_model(ckpt, arch=TinyNet, device="cpu"))
