# Changelog

Each release lists the Google Colab runtime it was tested against
(from [googlecolab/backend-info](https://github.com/googlecolab/backend-info)).

## 0.1.4 (unreleased)

- `tools.layer_shapes()`: input and output shapes of any layer or block (all positional
  inputs, keyword tensors, and every output), from one forward pass in eval mode.
- `viz.show_cam(..., show_color_bar=True)`: a 0-1 importance color bar next to the overlay.
- `install()` uses Colab's preinstalled `uv` (much faster), falling back to pip.
  `backend="auto" | "uv" | "pip"`. `upgrade=True` never upgrades dependencies.

## 0.1.3 (2026-10-13)

- `torchinfo`, `netron`, `onnx`, and `onnxscript` are now installed with the package
  (versions capped; verified to change no preinstalled Colab package). `onnx` stays below
  1.23, which would require upgrading Colab's protobuf. `cu.install("tools")` is no longer
  needed.
- `tools.netron()` / `export_for_netron()` export to ONNX by default, with the graph's
  input and output named `input` and `output`, and hide torch's internal `treespec` warning.

## 0.1.1 (2026-10-12)

- `viz.show_images()`, `show_by_class()`, `show_cam()`, `show_feature_maps()`: the plot no
  longer appears twice when the call is the last line of a notebook cell.

## 0.1.0 (2026-10-12)

Tested against: Colab GPU runtime, Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, fastai 2.8.12.

- `env`: runtime info, notebook name detection, version check against the tested Colab runtime.
- `env.load_secrets()` / `env.get_secret()`: API keys from the environment, Colab Secrets, a `.env` file, or a prompt.
- `viz.show_images()`, `viz.show_by_class()`, `viz.tensor_to_image()`: look at datasets, loaders, and batches.
- `drive`: Drive mounting, timestamped run folders, persistent config (`data_dir`, `runs_dir`).
- `tools`: torchinfo summary wrapper, Netron viewer inside Colab.
- `install()`: locked installer that refuses to change packages Colab already provides.
- `models`: `load_model()` for fastai exports/checkpoints and PyTorch models/state dicts.
- `viz`: Grad-CAM, layer activations, feature-map plotting.
- Sphinx documentation (user guide and API reference), published to GitHub Pages.
