Colab compatibility
===================

Google updates Colab's runtime (Python, PyTorch, fastai, ...) every few weeks. This
package is built and tested against that runtime, not against "whatever pip picks".

Checking your runtime
---------------------

.. code-block:: python

   cu.env.check()

.. code-block:: text

   Runtime: Google Colab, Python 3.13.15 (tested: 3.13.15, Colab gpu runtime synced 2026-10-12)
   package        source                         installed     tested        status
   torch          provided by Colab              2.11.0+cu130  2.11.0+cu130  ok
   fastai         provided by Colab              2.8.12        2.8.12        ok
   ...
   torchinfo      installed by colabutilscs4337  1.8.0         -             ok

``different`` usually means Colab updated since this release, or the notebook is
pinned to an older runtime (Runtime > Change runtime type > Runtime version).
Things normally still work; if not, upgrade the package and restart the session.

How compatibility is maintained
-------------------------------

* **Nothing heavy is a dependency.** PyTorch, torchvision, fastai, NumPy, and
  Matplotlib are used from Colab, so installing this package can never replace them.
* **Tested inside Colab's own image.** Every change runs the tests inside Google's
  published Colab runtime image (``us-docker.pkg.dev/colab-images/public/cpu-runtime``)
  and checks that ``pip install "colabutilscs4337[all]"`` changes no preinstalled
  package.
* **Tested with Colab's exact versions.** A second job installs the versions listed
  in `googlecolab/backend-info <https://github.com/googlecolab/backend-info>`_ on
  Python 3.12 and 3.13.
* **Weekly drift check.** A scheduled job re-runs everything against the newest
  Colab image and opens a pull request when Google's published versions change.
* **Manual GPU check before each release** with ``notebooks/smoke_test.ipynb``.
