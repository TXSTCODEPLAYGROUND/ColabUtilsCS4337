Development and releases
========================

Set up
------

.. code-block:: bash

   git clone https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337.git
   cd ColabUtilsCS4337
   uv venv --python 3.13
   uv pip install -c constraints/colab-gpu.txt -e ".[all,dev]" torch torchvision fastai
   pytest
   ruff check . && ruff format --check .

``constraints/colab-gpu.txt`` pins Colab's exact versions, so your local environment
matches the runtime students use.

Building these pages
--------------------

Only the documentation requirements are needed (no PyTorch):

.. code-block:: bash

   python -m venv .venv-docs
   source .venv-docs/bin/activate
   pip install -r docs/requirements.txt
   cd docs
   make html

Open ``docs/_build/html/index.html`` in a browser. Warnings are errors, as in CI.
The site is published to GitHub Pages by ``.github/workflows/docs.yml`` on every
push to ``main``; ``.readthedocs.yaml`` lets Read the Docs build the same site.

Repository layout
-----------------

.. code-block:: text

   src/colabutilscs4337/   the package (env, drive, tools, models, viz)
   tests/                  pytest suite (Colab is mocked; torch/fastai tests run on CPU)
   docs/                   this documentation (Sphinx)
   constraints/            Colab's published package versions (generated)
   scripts/                sync_colab_runtime.py, check_no_reinstall.py, ci_colab_image.sh
   notebooks/              smoke_test.ipynb: manual pre-release check in Colab
   .github/workflows/      ci, colab-drift, docs, release

Following Colab updates
-----------------------

.. code-block:: bash

   python scripts/sync_colab_runtime.py --check   # has Colab changed?
   python scripts/sync_colab_runtime.py           # update constraints/ and _tested.json

The weekly ``colab-drift`` workflow does this automatically and opens a pull request.

Releasing
---------

#. Update ``CHANGELOG.md`` with the Colab runtime the release was tested on.
#. Push to ``main`` and tag the same commit:
   ``git push && git tag v0.2.0 && git push origin v0.2.0``. The release workflow
   reuses (or waits for) the CI run of that ``main`` push instead of running the
   ~6 minute Colab image job again, then builds the package and publishes it to
   TestPyPI. If you tag a commit that was never pushed to ``main``, it runs the
   full CI first.
#. Open ``notebooks/smoke_test.ipynb`` in Colab on a GPU runtime, set ``VERSION``,
   and **Run all**.
#. Approve the ``pypi`` environment in the GitHub Actions run to publish to PyPI.

One-time setup: add a Trusted Publisher for this repository (workflow
``release.yml``, environments ``pypi`` and ``testpypi``) on pypi.org and
test.pypi.org, and create those two environments in the GitHub repository settings
(with yourself as a required reviewer on ``pypi``).
