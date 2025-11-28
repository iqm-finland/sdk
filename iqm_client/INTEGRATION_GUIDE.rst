=================
Integration Guide
=================

``iqm-client`` is the Python client for connecting to IQM's quantum computers, for application-level
quantum computing frameworks.  For examples of integrations maintained by IQM, please refer to the
:ref:`Qiskit <User guide Qiskit>` and :ref:`Cirq <User guide Cirq>` packages.

IQM client offers the functionality to submit quantum circuit execution jobs to the quantum computer,
track the statuses of jobs, and query various properties of the quantum computer.

The following sections illustrate how to integrate IQM quantum computers into your quantum computing
framework.


Authentication
--------------

IQM uses bearer token authentication to manage access to quantum computers.
Get your personal API token from the IQM Server web dashboard. The generated token can be provided to
IQM client via an environment variable :envvar:`IQM_TOKEN`.
Alternatively, the token can be provided as the ``token`` argument to :class:`.IQMClient` constructor.


Code example
------------

The connection to an IQM Server instance is managed by an :class:`.IQMClient` instance.
Initialization is simple, and in case you perform the authentication
using the :envvar:`IQM_TOKEN` environment variable, it only requires the URL of IQM Server:

.. code-block:: python

    from iqm.iqm_client import IQMClient

    server_url = "https://<IQM_SERVER_URL>"
    iqm_client = IQMClient(server_url)

To submit a quantum circuit for execution, it has to be specified using the
:class:`iqm.pulse.Circuit` class. The available native instructions are documented in
:mod:`iqm.iqm_client.models` and in :class:`iqm.pulse.CircuitOperation`.

.. code-block:: python

    from math import pi

    from iqm.pulse.builder import CircuitOperation
    from iqm.pulse.circuit_operations import Circuit

    instructions = (
        CircuitOperation(
            name="prx", locus=("QB1",), args={"phase": 1.4 * pi, "angle": 0.5 * pi}
        ),
        CircuitOperation(name="cz", locus=("QB1", "QB2"), args={}),
        CircuitOperation(name="measure", locus=("QB2",), args={"key": "Qubit 2"}),
    )

    circuit = Circuit(name="quantum_circuit", instructions=instructions)

Then the circuit(s) can be submitted to the server using :meth:`.IQMClient.submit_circuits`.
Upon successful submission the method returns a :class:`.CircuitJob` object that can be used
to track the progress of the job. This is a convenience, the only thing that is really needed
to access the job on IQM Server is the unique job ID in :attr:`.CircuitJob.job_id`.
You can pass the job ID to :meth:`.IQMClient.get_job` to get a new :class:`.CircuitJob` object for the job.

To query the status of the job, use :meth:`.CircuitJob.update`. It will update the job object and
return its current status. The different job statuses are documented in :class:`.JobStatus`.

.. code-block:: python

    job = iqm_client.submit_circuits([circuit], shots=1000)
    print(job.job_id)

    job_status = job.update()
    print(job_status)


Eventually the job will end up in one of the terminal statuses:
``"completed"``, ``"failed"``, or ``"cancelled"``.
You can either periodically query the status, or use :meth:`.CircuitJob.wait_for_completion`
which will block and poll the job status until it hits a terminal status, which is then returned.

When the status is ``"completed"``, you can use :meth:`.CircuitJob.result` to get the job results:

.. code-block:: python

    job_status = job.wait_for_completion()
    print(job_status)
    job_result = job.result()

A job can be cancelled by calling :meth:`.CircuitJob.cancel`.


Job payload
-----------

A ``dict[str, Any]`` containing arbitrary metadata can be attached to :class:`iqm.pulse.Circuit`
before submitting it for execution.
The attached metadata should consist only of values of JSON serializable datatypes.
A utility function :func:`~.iqm_client.util.to_json_dict` can be used to convert supported datatypes,
e.g. :class:`numpy.ndarray`, to equivalent JSON serializable types.

The server stores the job payload (including the metadata), and it can be queried using
:meth:`.CircuitJob.payload`, which returns the submitted circuits (with their metadata), and
the various job parameters used.


Job metadata and errors
-----------------------

The server attaches its own metadata to the job, including details related to the compilation
and execution of the job. Important metadata items include

* :attr:`.CircuitJob.data.errors`: list of errors for a ``"failed"`` job
* :attr:`.CircuitJob.data.messages`: list of informational messages
* :attr:`.CircuitJob.data.timeline`: list of execution steps reached with their timestamps
* :attr:`.CircuitJob.data.compilation.calibration_set_id`: ID of the calibration set used in the execution


Job timeline
~~~~~~~~~~~~

Each item in :attr:`.CircuitJob.data.timeline` is a :class:`.TimelineEntry`,
containing an execution step reached, a timestamp, and the source of the entry.
The steps in the timeline are more detailed than the job statuses.
For example, when the job is accepted, IQM Server adds a timestamp with the status ``"created"``.

The timeline entries also contain information about the lower-level job processing steps by
Station Control. For example, before the circuits can be executed they are compiled into instruction
schedules, indicated by the ``"compilation_started"`` and ``"compilation_ended"`` timestamps.
The actual execution of the job on the quantum hardware is indicated by the
``"execution_started"`` and ``"execution_ended"`` timestamps.


Circuit transpilation
---------------------

IQM does not provide an open source circuit transpilation library, so this will have to be supplied
by the quantum computing framework or a third party library.  To obtain the necessary information
for circuit transpilation, :meth:`.IQMClient.get_dynamic_quantum_architecture` returns the names of the
QPU components (qubits and computational resonators), and the native operations available
in the given calibration set. This information should enable circuit transpilation for the
IQM Crystal quantum architectures.

The notable exception is the transpilation for the IQM Star quantum architectures, which have
computational resonators in addition to qubits. Some specialized transpilation logic involving
the MOVE gates specific to these architectures is provided, in the form of the functions
:func:`.transpile_insert_moves` and :func:`.transpile_remove_moves`.
See :mod:`iqm.iqm_client.transpile` for the details.

A typical Star architecture use case would look something like this:

.. code-block:: python

    from iqm.iqm_client import IQMClient, simplify_architecture, transpile_insert_moves, transpile_remove_moves
    from iqm.pulse.circuit_operations import Circuit

    client = IQMClient(URL_TO_STAR_SERVER)
    dqa = client.get_dynamic_quantum_architecture()
    simplified_dqa = simplify_architecture(dqa)

    # circuit valid for simplified_dqa
    circuit = Circuit(name="quantum_circuit", instructions=[...])

    # intended use
    circuit_with_moves = transpile_insert_moves(circuit, dqa)
    job = client.submit_circuits([circuit_with_moves])

    # back to simplified dqa
    circuit_without_moves = transpile_remove_moves(circuit_with_moves)
    # circuit_without_moves is equivalent to circuit


Note on qubit mapping
---------------------

We encourage to transpile circuits to use the physical IQM qubit names before submitting them to IQM
quantum computers.  In case the quantum computing framework does not allow for this, providing a
qubit mapping can do the translation from the framework qubit names to IQM qubit names.  Note that
qubit mapping is not supposed to be associated with individual circuits, but rather with the entire
job request to IQM Server.  Typically, you would have some local representation of the QPU and
transpile the circuits against that representation, then use qubit mapping along with the generated
circuits to map from the local representation to the IQM representation of qubit names.
We discourage exposing this feature to end users of the quantum computing framework.

Note on circuit duration check
------------------------------

Before performing circuit execution, IQM Server checks how long it would take to run each circuit.
If any circuit in a job would take too long to execute compared to the T2 time of the qubits,
the server will disqualify the job, not execute any circuits, and return a detailed error message.
In some special cases, it makes sense to adjust or disable this check using
the :attr:`max_circuit_duration_over_t2` attribute of :class:`.CircuitCompilationOptions`,
and then passing the options to :meth:`.IQMClient.submit_circuits`.

Note on environment variables
-----------------------------

Set :envvar:`IQM_CLIENT_REQUESTS_TIMEOUT` environment variable to override the network request
default timeout value (in seconds) for :class:`.IQMClient` methods.  The default value is 120
seconds and might not be sufficient e.g.  when fetching the results of a larger circuit job through
a slow network connection.

On Linux:

.. code-block:: bash

  $ export IQM_CLIENT_REQUESTS_TIMEOUT=300

On Windows:

.. code-block:: batch

  set IQM_CLIENT_REQUESTS_TIMEOUT=300

Set :envvar:`IQM_CLIENT_SECONDS_BETWEEN_CALLS` to control the polling interval (in seconds) when
waiting for a job to complete with :meth:`.CircuitJob.wait_for_completion`.
The default value is 1 second.

Set :envvar:`IQM_CLIENT_DEBUG=1` to print the run request when it is submitted for execution in
:meth:`.IQMClient.submit_circuits` or :meth:`.IQMClient.submit_run_request`. To inspect the run
request without sending it for execution, use :meth:`.IQMClient.create_run_request`.

Integration testing
-------------------

IQM provides a demo environment to test the integration against a mock quantum computer. If you'd
like to request access to that environment, please contact `IQM <info@meetiqm.com>`_.
