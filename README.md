# colabutilscs4337

[![Documentation](https://img.shields.io/badge/docs-sphinx-blue)](https://txstcodeplayground.github.io/ColabUtilsCS4337/)
[![PyPI](https://img.shields.io/pypi/v/colabutilscs4337)](https://pypi.org/project/colabutilscs4337/)
[![CI](https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337/actions/workflows/ci.yml/badge.svg)](https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337/actions/workflows/ci.yml)

Google Colab utilities for CS4337 (computer vision with PyTorch and fastai).

**Documentation:** <https://txstcodeplayground.github.io/ColabUtilsCS4337/> |
**PyPI:** <https://pypi.org/project/colabutilscs4337/> |
**GitHub:** <https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337>

The documentation has a user guide with examples for every feature and the full API
reference. This README is a short overview.

- **`env`**: detect Colab, get the notebook name, load secrets (API keys) into the environment, check package versions against the Colab runtime this release was tested on.
- **`drive`**: mount Google Drive, create run folders like `MyDrive/runs/<notebook>_<YYYYmmdd-HHMMSS>_<slug>/`, keep default paths (`data_dir`, `runs_dir`) across sessions.
- **`tools`**: `torchinfo` summaries and the Netron graph viewer inside Colab.
- **`models`**: `load_model()` for fastai exports/checkpoints and PyTorch models/state dicts.
- **`viz`**: image grids for datasets and batches, Grad-CAM, layer activations, and feature-map plots.

## Quick start (first cell of every notebook)

```python
%pip install -q --upgrade colabutilscs4337   # also installs torchinfo, netron, onnx, onnxscript
import colabutilscs4337 as cu
cu.env.check()      # compares installed versions with the tested Colab runtime
```

If `%pip install --upgrade` replaced a version you had already imported, use **Runtime > Restart session**.

## What gets installed

Colab ships torch, torchvision, fastai, numpy, and matplotlib built for its GPU runtime.
This package never declares them as dependencies, so installing it cannot replace them.
It does install the four tools Colab lacks (`torchinfo`, `netron`, `onnx`,
`onnxscript`); CI checks inside Google's Colab image that they change no preinstalled
package, and their versions are capped so a future release cannot pull a newer numpy.

`cu.install()` adds optional packages the same safe way: it pins every installed
package to its current version, dry-runs the install, and refuses if anything already
installed would change. It uses Colab's preinstalled `uv` for speed and falls back to pip.

```python
cu.install("cam")  # pytorch-grad-cam, for gradcam(..., method="gradcam++")
cu.install("torchmetrics", dry=True)  # preview only
```

## Secrets (API keys and tokens)

Never paste keys into a notebook. Store them once, then load them by name:

1. In Colab, click the **key icon** in the left sidebar, choose **Add new secret**,
   enter a name (for example `HF_TOKEN`) and the value, and switch on **Notebook access**.
2. In the notebook:

```python
cu.env.load_secrets("HF_TOKEN", "WANDB_API_KEY")
#   ok  HF_TOKEN  (Colab Secrets)
#   --  WANDB_API_KEY
#   Missing: WANDB_API_KEY. In Colab: click the key icon ...
```

The values are now in `os.environ`, where libraries such as Hugging Face, Weights &
Biases, and Kaggle look for them. Values are never printed.

Each name is looked up in this order:

1. Already set in the environment.
2. Colab Secrets.
3. A `.env` file: the `env_file=` argument, else `./.env`, else
   `MyDrive/.colabutilscs4337/.env` when Drive is mounted (handy for keys you reuse
   across notebooks).
4. A hidden password-style prompt, only with `prompt=True`.

```python
cu.env.load_secrets("KAGGLE_USERNAME", "KAGGLE_KEY", required=True)  # error if missing
cu.env.load_secrets(env_file="/content/drive/MyDrive/keys.env")  # every key in a file
cu.env.load_secrets("OPENAI_API_KEY", prompt=True)  # ask if not found
token = cu.env.get_secret("HF_TOKEN")  # just return the value
```

A `.env` file is plain `NAME=value` lines (`#` comments and quotes are fine).

## Looking at image data

Pass whatever you have: a `Dataset`, a `DataLoader`, an `(x, y)` batch, a tensor plus
labels, or a list of PIL images. Class names come from `dataset.classes` automatically.

```python
train_ds = torchvision.datasets.FashionMNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)

cu.viz.show_by_class(train_ds)  # one row per class, 8 examples each
cu.viz.show_images(train_ds)  # 32 random images titled with their class
cu.viz.show_images(train_loader)  # the first batch of a DataLoader

x, y = next(iter(train_loader))
cu.viz.show_images(x, y, train_ds.classes, n=16)
cu.viz.show_images(x, y, train_ds.classes, preds=model(x).argmax(1))  # "pred / true", red if wrong
cu.viz.show_by_class(
    x, y, labels_map, n_per_class=5
)  # labels_map can be a list or {0: "T-Shirt", ...}

img = cu.viz.tensor_to_image(x[:8])  # PIL image: 8 images side by side
img = cu.viz.tensor_to_image(x[:8], dim=1)  # ... stacked top to bottom
```

| Function | Shows |
|---|---|
| `show_by_class(data, labels=None, classes=None, n_per_class=8)` | one row per class |
| `show_images(data, labels=None, classes=None, preds=None, n=32, ncols=8)` | a grid with class (and prediction) titles |
| `tensor_to_image(*tensors, dim=2, pad=0)` | returns a `PIL.Image` strip |

Normalized images (for example ImageNet `Normalize`) are rescaled automatically for
display; pass `mean=cu.viz.IMAGENET_MEAN, std=cu.viz.IMAGENET_STD` to undo the
normalization exactly. Grayscale is drawn in gray, colour in RGB. With a fastai
`DataLoaders`, `show_images(dls)` calls fastai's own `show_batch`.

## Runs and data

```python
run = cu.drive.new_run("resnet34-baseline")  # mounts Drive if needed
print(run)  # /content/drive/MyDrive/runs/Lab-3-CNNs_20261012-221500_resnet34-baseline
learn.path = run.path  # fastai: save models into the run folder
torch.save(model.state_dict(), run / "model.pth")

path = cu.drive.data_dir("pets")  # /content/data/pets (fast local disk)
cu.drive.set_config(timezone="America/Chicago", data_dir="/content/data")  # saved on Drive
cu.drive.latest_run()  # most recent run folder
```

Each run folder contains a `run.json` with the notebook name, timestamp, and the
runtime's Python, GPU, and package versions. Without a slug, a random one such as
`brave-otter` is used.

Settings come from (highest first): function arguments, environment variables
(`COLABUTILS_DATA_DIR`, `COLABUTILS_RUNS_DIR`, `COLABUTILS_TZ`,
`COLABUTILS_NOTEBOOK_NAME`), `MyDrive/.colabutilscs4337/config.json`, then defaults.

## Inspecting models

```python
cu.tools.summary(learn, input_size=(1, 3, 224, 224))  # fastai Learner or nn.Module
cu.tools.layer_shapes(backbone, backbone.features[2][0], (1, 3, 160, 160))
# features.2.0 (MBConv): (1, 16, 80, 80) -> (1, 24, 40, 40)
cu.tools.netron(learn)  # interactive graph in the cell output
cu.tools.netron("model.onnx")  # or any file Netron supports

model = cu.models.load_model("export.pkl")  # fastai learn.export()
model = cu.models.load_model("model.pth", arch=torchvision.models.resnet18(num_classes=37))
```

## Visualizing what a CNN looks at

```python
cam = cu.viz.gradcam(model, x, class_idx=None)  # x: (1, 3, H, W) normalized tensor
cu.viz.show_cam(image, cam)  # image: PIL / HxWx3 array / tensor; 0-1 color bar included
df = cu.viz.get_classification_topk(model, img, categories, preprocess=preprocess)
cu.viz.visualize_gradcam(
    model,
    img,
    target_indices=[281, 282],
    target_layers=[model.layer4[-1]],
    preprocess=preprocess,
    categories=categories,
)

acts = cu.viz.activations(model, x, layers=["layer1", "layer4"])
cu.viz.show_feature_maps(acts["layer1"], n=16)
```

`target_layer` defaults to `cu.viz.suggest_target_layer(model)` (the last conv
layer). With `cu.install("cam")`, `cu.viz.gradcam(..., method="gradcam++")` uses the
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) backend.

## Compatibility with Colab

The Colab runtime is tracked from
[googlecolab/backend-info](https://github.com/googlecolab/backend-info):

- `scripts/sync_colab_runtime.py` writes `constraints/colab-gpu.txt` and
  `src/colabutilscs4337/_tested.json`.
- CI runs the tests inside Google's Colab runtime image
  (`us-docker.pkg.dev/colab-images/public/cpu-runtime`) and checks that
  `pip install ".[all]"` changes no preinstalled package.
- A weekly job reruns everything and opens a PR when Colab's runtime changes.

## Development

```bash
uv venv --python 3.13
uv pip install -c constraints/colab-gpu.txt -e ".[all,dev]" torch torchvision fastai
pytest
ruff check .
```

Build the documentation (Sphinx; PyTorch not needed):

```bash
uv venv .venv-docs --python 3.13
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/sphinx-build -W -b html docs docs/_build/html
open docs/_build/html/index.html
```

`.github/workflows/docs.yml` builds the site on every pull request and publishes it to
GitHub Pages on every push to `main`. See the
[development guide](https://txstcodeplayground.github.io/ColabUtilsCS4337/development.html)
for releases.

Release: push a tag `vX.Y.Z`. CI publishes to TestPyPI, then PyPI after approval.
Before approving, run `notebooks/smoke_test.ipynb` on a Colab GPU runtime.
