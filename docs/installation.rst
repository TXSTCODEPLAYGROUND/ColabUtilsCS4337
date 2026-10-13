Installation
============

In Google Colab
---------------

Put this in the first cell of every notebook:

.. code-block:: python

   %pip install -q --upgrade colabutilscs4337
   import colabutilscs4337 as cu
   cu.install("all")
   cu.env.check()

The first line can never break your runtime: the package has **no required
dependencies**. PyTorch, torchvision, fastai, NumPy, and Matplotlib are already
installed in Colab, built for its GPU, and this package uses those copies.

``cu.install("all")`` adds the optional tools. Each one comes from an *extra*:

============  ===================================  ==================================
Extra         Installs                             Needed for
============  ===================================  ==================================
``tools``     ``torchinfo``, ``netron``            :func:`~colabutilscs4337.tools.summary`,
                                                   :func:`~colabutilscs4337.tools.netron`
``cam``       ``grad-cam``                         Grad-CAM++, Score-CAM, and other
                                                   :func:`~colabutilscs4337.viz.gradcam` methods
``all``       ``tools`` + ``cam``
``fastai``    ``fastai``                           only if your runtime lacks fastai
============  ===================================  ==================================

If you call a function whose tool is missing, the error message tells you which
``cu.install(...)`` to run.

Why ``cu.install()`` instead of ``pip install``?
------------------------------------------------

``pip`` happily upgrades or downgrades packages that are already installed when a
new package asks for it. In Colab that can replace the GPU build of PyTorch and
cost you a runtime restart (or a broken runtime). :func:`colabutilscs4337.install`
prevents that:

#. It records every installed package and version.
#. It asks pip, in a dry run, what it *would* do with all of those versions pinned.
#. It installs only if nothing already installed changes; otherwise it stops and
   explains what would have changed.

.. code-block:: python

   cu.install("cam", dry=True)        # preview: "[dry run] Added: grad-cam 1.5.7, ttach 0.0.3"
   cu.install("cam")                  # install
   cu.install("onnx")                 # any other package works too
   cu.install("numpy==2.3.0")         # refused: would change Colab's numpy
   cu.install("numpy==2.3.0", allow_changes=True)   # only if you really mean it

Plain pip works too, with the same protection, by pinning what is installed:

.. code-block:: bash

   pip list --format=freeze | grep -viE '^(colabutilscs4337|torchinfo|netron|grad-cam|ttach)==' > /tmp/c.txt
   pip install -c /tmp/c.txt --upgrade "colabutilscs4337[all]"

Upgrading
---------

Run the first cell again. If the upgrade replaced something you had already
imported, ``cu.install`` prints a reminder; use **Runtime > Restart session** and
run the notebook from the top.

Outside Colab
-------------

Everything also works in local Jupyter, VS Code, or plain Python (Python 3.12+).
Install PyTorch yourself first, then ``pip install "colabutilscs4337[all]"``.
Google Drive is replaced by a local ``./drive/MyDrive`` folder.
