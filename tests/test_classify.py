import pytest

from colabutilscs4337 import viz

torch = pytest.importorskip("torch")
nn = torch.nn
pd = pytest.importorskip("pandas")


class Tiny(nn.Module):
    """Global-average-pool classifier so a small input is enough."""

    def __init__(self, n=4):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1))
        self.fc = nn.Linear(4, n)

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x))).mean((2, 3))
        return self.fc(x)


CATEGORIES = ["dog", "tabby cat", "catch", "tiger cat"]


def test_find_categories_like_list_and_dict():
    assert viz.find_categories_like(CATEGORIES, r"\bcat\b") == [
        ("tabby cat", 1),
        ("tiger cat", 3),
    ]
    found = viz.find_categories_like({7: "house cat", 8: "catch"}, r"\bcat\b")
    assert found == [("house cat", 7)]


def test_classification_topk_sorted_and_limited():
    torch.manual_seed(0)
    model = Tiny().eval()
    image = torch.randn(3, 16, 16)
    df = viz.get_classification_topk(model, image, CATEGORIES, k=2)
    assert list(df.columns) == ["Index", "label", "probability"]
    assert len(df) == 2
    assert df["probability"].is_monotonic_decreasing
    assert df["probability"].sum() <= 1
    assert df["label"].tolist() == [CATEGORIES[i] for i in df["Index"]]

    every = viz.get_classification_topk(model, image[None], CATEGORIES, k=None)
    assert len(every) == 4
    with pytest.raises(ValueError, match="k must"):
        viz.get_classification_topk(model, image, k=0)


def test_visualize_gradcam_grid_and_single_row():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    torch.manual_seed(0)
    model = Tiny().eval()
    image = torch.randn(1, 3, 16, 16)

    grid = viz.visualize_gradcam(
        model,
        image,
        target_indices=[1, 3],
        target_layers=[model.layer1[-1], "layer2"],
        categories=CATEGORIES,
    )
    pictures = [ax for ax in grid.axes if ax.get_title()]
    assert len(pictures) == 4
    assert any(ax.get_ylabel() == "importance" for ax in grid.axes)
    titles = [ax.get_title() for ax in pictures]
    assert titles[0].startswith("layer1.0") and "tabby cat" in titles[0]
    assert titles[1].startswith("layer2")

    one_class = viz.visualize_gradcam(
        model, image, target_indices=[1], target_layers=[model.layer1, model.layer2]
    )
    assert one_class.axes[0].get_subplotspec().get_gridspec().nrows == 1
    one_layer = viz.visualize_gradcam(model, image, [0, 1], model.layer2[-1], categories=CATEGORIES)
    assert one_layer.axes[0].get_subplotspec().get_gridspec().nrows == 1
    assert "dog" in one_layer.axes[0].get_title()


def test_visualize_accepts_a_tensor_of_indices_and_pil():
    pytest.importorskip("matplotlib")
    from PIL import Image

    torch.manual_seed(0)
    model = Tiny().eval()
    indices = torch.tensor([0, 2])
    fig = viz.visualize_gradcam(
        model, torch.rand(3, 8, 8), indices, [model.layer1[-1]], show_color_bar=False
    )
    assert len([ax for ax in fig.axes if ax.get_title()]) == 2

    pil = Image.fromarray(__import__("numpy").zeros((12, 12, 3), dtype="uint8"))
    called = {}

    def preprocess(img):
        called["ok"] = True
        return torch.zeros(3, 8, 8)

    viz.get_classification_topk(model, pil, CATEGORIES, k=1, preprocess=preprocess)
    assert called["ok"]
    with pytest.raises(ValueError, match="preprocess"):
        viz.get_classification_topk(model, pil, CATEGORIES)
