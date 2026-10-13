``cu.drive``: Drive, runs, and settings
=======================================

.. automodule:: colabutilscs4337.drive
   :no-members:

Run folders
-----------

.. autofunction:: colabutilscs4337.drive.new_run
.. autoclass:: colabutilscs4337.drive.Run
   :members: name, subdir
.. autofunction:: colabutilscs4337.drive.list_runs
.. autofunction:: colabutilscs4337.drive.latest_run
.. autofunction:: colabutilscs4337.drive.slugify
.. autofunction:: colabutilscs4337.drive.random_slug

Drive
-----

.. autofunction:: colabutilscs4337.drive.mount
.. autofunction:: colabutilscs4337.drive.is_mounted

Settings and data folder
------------------------

.. autofunction:: colabutilscs4337.drive.data_dir
.. autofunction:: colabutilscs4337.drive.get_config
.. autofunction:: colabutilscs4337.drive.set_config
.. autofunction:: colabutilscs4337.drive.reset_config
.. autofunction:: colabutilscs4337.drive.config_path
.. autofunction:: colabutilscs4337.drive.default_config
.. autoclass:: colabutilscs4337.drive.Config
   :members: data_path, runs_path
