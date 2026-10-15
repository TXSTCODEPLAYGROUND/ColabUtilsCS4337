import pytest

from colabutilscs4337 import viz

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
nn = torch.nn


@pytest.fixture(scope="module")
def resnet():
    torch.manual_seed(0)
    return torchvision.models.resnet18(num_classes=5).eval()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 3, 64, 64)


def test_suggest_target_layer(resnet, x):
    assert viz.suggest_target_layer(resnet, x) == "layer4"
    assert viz.suggest_target_layer(resnet) == "layer4.1.conv2"


def test_suggest_target_layer_fastai_style_sequential(x):
    body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1))
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3))
    assert viz.suggest_target_layer(nn.Sequential(body, head), x) == "0"


def test_viz_preserves_train_mode_and_bn_stats(x):
    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=3).train()
    stats = model.bn1.running_mean.clone()
    viz.suggest_target_layer(model, x)
    viz.activations(model, x[:1], layers=["layer1"])
    viz.gradcam(model, x[:1])
    assert model.training
    assert torch.equal(model.bn1.running_mean, stats)


def test_activations(resnet, x):
    acts = viz.activations(resnet, x, layers=["layer1", resnet.layer4])
    assert set(acts) == {"layer1", "layer4"}
    assert acts["layer1"].shape == (2, 64, 16, 16)
    assert not acts["layer4"].requires_grad
    assert len(viz.activations(resnet, x)) == len(viz.layer_names(resnet, nn.Conv2d))


def test_gradcam_builtin(resnet, x):
    result = viz.gradcam(resnet, x)
    assert result.cams.shape == (2, 64, 64)
    assert result.cam.shape == (64, 64)
    assert float(result.cams.min()) >= 0 and float(result.cams.max()) <= 1
    assert result.class_idx == resnet(x).argmax(1).tolist()
    assert result.layer == "layer4" and result.method == "gradcam"
    assert all(p.grad is None for p in resnet.parameters())


def test_gradcam_matches_pytorch_grad_cam(resnet, x):
    pgc = pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    ours = viz.gradcam(resnet, x, target_layer="layer4", class_idx=[0, 3])
    with pgc.GradCAM(model=resnet, target_layers=[resnet.layer4]) as cam:
        ref = cam(input_tensor=x, targets=[ClassifierOutputTarget(0), ClassifierOutputTarget(3)])
    corr = torch.corrcoef(torch.stack([ours.cams.flatten(), torch.from_numpy(ref).flatten()]))
    assert corr[0, 1] > 0.95


def test_gradcam_backend_method(resnet, x):
    pytest.importorskip("pytorch_grad_cam")
    result = viz.gradcam(resnet, x[:1], method="gradcam++", class_idx=2)
    assert result.cams.shape == (1, 64, 64)
    assert result.class_idx == [2]


def test_gradcam_unknown_method(resnet, x):
    with pytest.raises(ValueError, match="Unknown method"):
        viz.gradcam(resnet, x, method="nope")


def test_plots(resnet, x):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    result = viz.gradcam(resnet, x[0])
    fig = viz.show_cam(x[0], result, mean=viz.IMAGENET_MEAN, std=viz.IMAGENET_STD)
    assert len(fig.axes) == 3  # input, overlay, color bar
    assert fig.axes[-1].get_ylabel() == "importance"
    assert len(viz.show_cam(x[0], result, show_color_bar=False).axes) == 2
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    viz.show_cam(x[0], result, ax=ax)
    assert len(ax.figure.axes) == 2
    acts = viz.activations(resnet, x, layers=["layer1"])
    fig = viz.show_feature_maps(acts["layer1"], n=10, ncols=5)
    assert len(fig.axes) == 10
    blended = viz.overlay(torch.rand(3, 32, 32), result)
    assert blended.shape == (32, 32, 3)


def test_fastai_learner_end_to_end(tmp_path):
    vision = pytest.importorskip("fastai.vision.all")
    pytest.importorskip("pytorch_grad_cam")
    import numpy as np
    from PIL import Image

    for label in ("a", "b"):
        (tmp_path / label).mkdir()
        for i in range(4):
            arr = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / label / f"{i}.png")
    dls = vision.ImageDataLoaders.from_folder(
        tmp_path,
        valid_pct=0.25,
        item_tfms=vision.Resize(64),
        bs=2,
        num_workers=0,
        device=torch.device("cpu"),
    )
    learn = vision.vision_learner(dls, vision.resnet18, pretrained=False)
    xb = viz.learner_input(learn, vision.get_image_files(tmp_path)[0])

    assert viz.suggest_target_layer(learn, xb) == "0"
    for method in ("gradcam", "gradcam++"):
        result = viz.gradcam(learn, xb, method=method)
        assert result.cams.shape == (1, 64, 64)
    assert viz.activations(learn, xb, layers=["0.4"])["0.4"].shape[0] == 1
    assert learn.model.training


def test_to_image_variants():
    np = pytest.importorskip("numpy")
    from PIL import Image

    pil = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    assert viz.to_image(pil).shape == (8, 8, 3)
    assert viz.to_image(torch.randn(1, 3, 8, 8)).max() <= 1
    assert viz.to_image(np.ones((8, 8))).shape == (8, 8, 3)
