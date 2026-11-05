``cu.env``: runtime and secrets
===============================

.. automodule:: colabutilscs4337.env
   :no-members:

Secrets
-------

.. autofunction:: colabutilscs4337.env.load_secrets
.. autofunction:: colabutilscs4337.env.get_secret
.. autofunction:: colabutilscs4337.env.read_env_file
.. autoexception:: colabutilscs4337.env.MissingSecretError

Env file from Drive
-------------------

.. autofunction:: colabutilscs4337.env.sync_env_file
.. autofunction:: colabutilscs4337.env.drive_env_path
.. autoexception:: colabutilscs4337.env.EnvFileError

Runtime
-------

.. autofunction:: colabutilscs4337.env.check
.. autofunction:: colabutilscs4337.env.runtime_info
.. autofunction:: colabutilscs4337.env.is_colab
.. autofunction:: colabutilscs4337.env.notebook_name
.. autofunction:: colabutilscs4337.env.gpu_info
.. autofunction:: colabutilscs4337.env.installed_version
.. autofunction:: colabutilscs4337.env.tested_runtime
.. autoclass:: colabutilscs4337.env.Row
