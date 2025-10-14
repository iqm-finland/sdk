.. _User guide CLI:

=========================
IQM Client CLI User Guide
=========================

Command-line interface (CLI) for managing user authentication when using IQM quantum computers.

Installing IQM Client CLI
-------------------------

.. code-block:: bash

  $ pip install iqm-client[cli]

Using IQM Client CLI
--------------------

For general usage instructions, run

.. code-block:: bash

  $ iqmclient --help

Initialization
^^^^^^^^^^^^^^

First, IQM Client CLI needs initialization, which produces a configuration file:

.. code-block:: bash

  $ iqmclient init

IQM Client CLI will ask a few questions. You can also pass the values via command line to avoid having an interactive
prompt. See ``iqmclient init --help`` for details.

Login
^^^^^

To log in, use

.. code-block:: bash

  $ iqmclient auth login

This will ask you to enter your username and password. If you have a temporary password you will be asked to go to the
authentication server and enter a new password. URL of the authentication server will be provided.

After a successful authentication, tokens will be saved into a tokens file (path specified in the configuration file),
and a token manager daemon will start in the background. Token manager will periodically refresh the session and
re-write the tokens file.

To use the token manager in a foreground mode (not as daemon), run ``iqmclient auth login --no-daemon``. This requires
keeping the shell session alive. However, you can start the process in the background by adding ``&`` after the
command: ``iqmclient auth login --no-daemon &``. This applies to Bash, zsh and similar shells, but may not be available
on all shells.

To login and get tokens once, without starting a token manager at all, run ``iqmclient auth login --no-refresh``.

If the tokens file already exists, then running ``iqmclient auth login`` will first attempt to refresh the session
without asking you for a username and password. If that fails (because existing tokens may already have expired), you'll
be asked to re-enter your credentials.

See ``iqmclient auth login --help`` for more details.

Use with Cirq on IQM, Qiskit on IQM, etc.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adapters based on IQM Client, such as Cirq on IQM and Qiskit on IQM, can take advantage of the tokens file maintained by
IQM Client CLI. This way you won't need to provide the authentication server URL, username, or password to the adapter
library itself. To achieve this, follow the instructions printed on the screen after running ``iqmclient auth login``.
Namely, set the ``IQM_TOKENS_FILE`` environment variable to point to your tokens file.

On Linux:

.. code-block:: bash

  $ export IQM_TOKENS_FILE=/home/<username>/tokens.json

On Windows:

.. code-block:: batch

  set IQM_TOKENS_FILE=C:\Users\<username>\.cache\iqm-client-cli\tokens.json

Once set, this environment variable is read by the instance of IQM Client associated with the adapter. As a result,
from the point of view of the adapter it looks like authentication is simply not required (i.e. no
authentication-related information has to be provided to the adapter).

Status
^^^^^^

To see the current status of the token manager, use:

.. code-block:: bash

  $ iqmclient auth status

If the tokens file exists, ``iqmclient auth status`` will report whether the corresponding token
manager is running. It will also print the time of the last successful refresh request, and
how much time is left until current tokens expire.

See ``iqmclient auth status --help`` for more details.

Logout
^^^^^^

To log out, run

.. code-block:: bash

  $ iqmclient auth logout

This will send a logout request to the authentication server, kill the token manager daemon (if any), and delete the
tokens file.

You may want to stop the token manager, but maintain the session on the server and keep the tokens file intact.
To do so, run:

.. code-block:: bash

  $ iqmclient auth logout --keep-tokens

See ``iqmclient auth logout --help`` for more details.

Multiple configuration files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, all IQM Client CLI commands read the configuration file from the default location
``~/.config/iqm-client-cli/config.json``. You can specify a different filepath by providing the ``--config-file`` value,
for example:

.. code-block:: bash

  $ iqmclient auth status --config-file /home/joe/config.json
  $ iqmclient auth login --config-file /home/joe/config.json
  $ iqmclient auth logout --config-file /home/joe/config.json
