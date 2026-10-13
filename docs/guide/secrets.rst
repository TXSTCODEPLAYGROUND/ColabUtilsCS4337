Secrets (API keys and tokens)
=============================

Weights & Biases, Hugging Face, Kaggle, and OpenAI all need a key. **Never paste a
key into a notebook**: notebooks get shared, committed, and submitted. Store the key
once, then load it by name.

Store the key in Colab
----------------------

#. Click the **key icon** in Colab's left sidebar.
#. Click **Add new secret**. Enter a *name* (for example ``WANDB_API_KEY``) and the
   *value*.
#. Switch on **Notebook access** for that secret.

Secrets belong to your Google account, so they are available in every notebook you
open, after you switch on access for that notebook.

Load it
-------

.. code-block:: python

   cu.env.load_secrets("WANDB_API_KEY", "HF_TOKEN")

.. code-block:: text

     ok  WANDB_API_KEY  (Colab Secrets)
     --  HF_TOKEN
     Missing: HF_TOKEN. In Colab: click the key icon in the left sidebar > 'Add new secret', ...

The values are now environment variables (``os.environ``), which is where these
libraries look for them, so ``wandb.login()`` and Hugging Face downloads just work.
Values are never printed.

Where keys are looked up
------------------------

For each name, the first place that has it wins:

#. An environment variable that is already set.
#. Colab Secrets.
#. A ``.env`` file: the ``env_file=`` argument; otherwise ``./.env``; otherwise
   ``MyDrive/.colabutilscs4337/.env`` if Drive is mounted.
#. A hidden, password-style prompt, only with ``prompt=True``.

A ``.env`` file is plain text, one ``NAME=value`` per line:

.. code-block:: bash

   # MyDrive/.colabutilscs4337/.env
   WANDB_API_KEY=0123456789abcdef
   KAGGLE_USERNAME=alice
   KAGGLE_KEY="abc123"

Recipes
-------

.. code-block:: python

   # Stop with a clear error if anything is missing
   cu.env.load_secrets("KAGGLE_USERNAME", "KAGGLE_KEY", required=True)

   # Load every key in a file
   cu.env.load_secrets(env_file="/content/drive/MyDrive/keys.env")

   # Ask for the key if it is not stored anywhere (typing is hidden)
   cu.env.load_secrets("OPENAI_API_KEY", prompt=True)

   # Just get the value, without touching os.environ
   token = cu.env.get_secret("HF_TOKEN")

   # Replace a value that is already in the environment
   cu.env.load_secrets("WANDB_API_KEY", overwrite=True)

Troubleshooting
---------------

``Colab has 'NAME' but this notebook can't read it``
   The secret exists, but **Notebook access** is off for this notebook. Open the key
   panel and switch it on.

The name is reported missing although you added it
   Names are case-sensitive and must match exactly (``WANDB_API_KEY``, not
   ``wandb_api_key``).

See :func:`colabutilscs4337.env.load_secrets` and
:func:`colabutilscs4337.env.get_secret` for every option.
