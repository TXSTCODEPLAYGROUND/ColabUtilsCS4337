import pytest

from colabutilscs4337 import viz

torch = pytest.importorskip("torch")
plt = pytest.importorskip("matplotlib.pyplot")
plt.switch_backend("Agg")

LABELS_MAP = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot",
}  # fmt: skip


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TinyDataset(torch.utils.data.Dataset):
    classes = ["zero", "one", "two"]

    def __init__(self, n=30, channels=1):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, channels, 8, 8, generator=g)
        self.y = torch.arange(n) % 3

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def test_tensor_to_image_row_and_column():
    x = torch.rand(4, 1, 8, 10)
    row = viz.tensor_to_image(x)
    assert row.mode == "L" and row.size == (40, 8)  # PIL size is (width, height)
    col = viz.tensor_to_image(x, dim=1)
    assert col.size == (10, 32)
    assert viz.tensor_to_image(x, pad=2).size == (46, 8)


def test_tensor_to_image_multiple_inputs_and_formats():
    rgb = viz.tensor_to_image(torch.rand(3, 8, 8), torch.rand(2, 3, 8, 8))
    assert rgb.mode == "RGB" and rgb.size == (24, 8)
    as_uint8 = viz.tensor_to_image(torch.full((1, 4, 4), 255, dtype=torch.uint8))
    assert as_uint8.getpixel((0, 0)) == 255
    normalized = torch.randn(2, 3, 8, 8)
    out = viz.tensor_to_image(normalized, mean=viz.IMAGENET_MEAN, std=viz.IMAGENET_STD)
    assert out.size == (16, 8)
    with pytest.raises(ValueError):
        viz.tensor_to_image(normalized, dim=0)


def test_show_images_from_batch_with_preds():
    x, y = torch.rand(12, 1, 8, 8), torch.arange(12) % 10
    preds = y.clone()
    preds[0] = 9
    fig = viz.show_images(x, y, LABELS_MAP, preds=preds, ncols=4)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(fig.axes) == 12 and titles[0] == "Ankle Boot / T-Shirt"
    assert fig.axes[0].title.get_color() == "red"
    assert fig.axes[1].title.get_color() == "green"


def test_show_images_from_dataset_and_loader():
    ds = TinyDataset()
    fig = viz.show_images(ds, n=6, seed=0)
    titles = {ax.get_title() for ax in fig.axes}
    assert titles <= set(TinyDataset.classes) and len(fig.axes) == 6

    loader = torch.utils.data.DataLoader(ds, batch_size=5)
    fig = viz.show_images(loader, n=10)
    assert len([ax for ax in fig.axes if ax.get_title()]) == 10

    batch = next(iter(loader))  # a raw (x, y) batch carries no class names
    assert [ax.get_title() for ax in viz.show_images(batch, n=3).axes] == ["0", "1", "2"]
    fig = viz.show_images(batch, classes=ds.classes, n=3)
    assert [ax.get_title() for ax in fig.axes] == ["zero", "one", "two"]


def test_show_images_from_pil_list():
    np = pytest.importorskip("numpy")
    from PIL import Image

    images = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(3)]
    fig = viz.show_images(images)
    assert len(fig.axes) == 3


def test_show_by_class_from_tensors():
    x = torch.rand(40, 1, 8, 8)
    y = (torch.arange(40) % 9)[:, None]  # class 9 has no examples; (B, 1) labels
    fig = viz.show_by_class(x, y, LABELS_MAP, n_per_class=4)
    assert len(fig.axes) == 10
    assert [ax.get_ylabel() for ax in fig.axes] == list(LABELS_MAP.values())


def test_show_by_class_from_dataset_mixed_sizes():
    ds = TinyDataset(channels=3)
    fig = viz.show_by_class(ds, n_per_class=3, seed=1)
    assert [ax.get_ylabel() for ax in fig.axes] == TinyDataset.classes

    fig = viz.show_by_class(torch.rand(6, 3, 8, 8), torch.tensor([0, 1, 2, 0, 1, 4]))
    assert [ax.get_ylabel() for ax in fig.axes] == ["0", "1", "2", "3", "4"]


def test_show_by_class_needs_labels():
    with pytest.raises(ValueError, match="labels"):
        viz.show_by_class(torch.rand(4, 1, 8, 8))
