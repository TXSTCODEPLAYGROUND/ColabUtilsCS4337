# Changelog

Each release lists the Google Colab runtime it was tested against
(from [googlecolab/backend-info](https://github.com/googlecolab/backend-info)).

## 0.1.0 (unreleased)

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
