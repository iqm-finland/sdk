.. _User guide Qrisp:

Qrisp on IQM User Guide
=======================

This guide introduces the main features of the Qrisp adapter of IQM Client.
You are encouraged to run the demonstrated code snippets and check the output yourself.

.. note::

   IQM provides access to its quantum computers via IQM Resonance – IQM's quantum cloud service.
   Please head over `to our website <https://iqm.tech/products/iqm-resonance/>`_ to learn more.


.. toctree::
   :maxdepth: 2
   :hidden:

   plasma_sabre_tutorial
   detector_experiment_demo


Installation
------------

The recommended way is to install the optional ``qrisp`` feature of the ``iqm-client`` distribution package
directly from the Python Package Index (PyPI):

.. code-block:: bash

   $ uv pip install "iqm-client[qrisp]"

After installation, the Qrisp adapter can be imported in your Python code as follows:

.. code-block:: python

   from iqm import qrisp_iqm


Authentication
--------------

The ``server_url`` must be set either directly or via the :envvar:`IQM_SERVER_URL`
environment variable.  The ``device_instance`` can be set directly or via
:envvar:`IQM_QUANTUM_COMPUTER`.

For authentication, you can choose one of two options:

1. Set the :envvar:`IQM_TOKEN` environment variable to the API token obtained from the web dashboard.
2. Pass the ``token`` keyword argument when initializing :class:`~iqm.qrisp_iqm.backends.IQMBackend`.

You can obtain your personal API token from the `IQM Resonance <https://resonance.iqm.tech>`_ web dashboard.

The recommended setup is to configure all three environment variables::

    export IQM_SERVER_URL="https://resonance.iqm.tech/"
    export IQM_TOKEN="<YOUR_TOKEN>"
    export IQM_QUANTUM_COMPUTER="garnet"


Hello, world!
-------------

The :class:`~iqm.qrisp_iqm.backends.IQMBackend` class is the central entry point for running Qrisp circuits on IQM hardware.
Here is a minimal example that creates a Bell state circuit using Qrisp, transpiles it for an IQM device,
and runs it on the backend:

.. code-block:: python

    from qrisp import QuantumVariable, h, cx, measure
    from iqm.qrisp_iqm import IQMBackend

    os.environ["IQM_SERVER_URL"] = "https://resonance.iqm.tech/"
    os.environ["IQM_TOKEN"] = "<YOUR_TOKEN>"

    # 1. Build the circuit
    qv = QuantumVariable(2)
    h(qv[0])
    cx(qv[0], qv[1])
    measure(qv)

    qc = qv.qs.compile()

    # 2. Connect to the backend and run
    backend = IQMBackend(device_instance = "garnet")
    result = backend.run(qc, shots=1000)
    print("Result counts:", result)

Transpilation
-------------

Before a Qrisp circuit can be executed on an IQM device, it must be **transpiled**: qubits must be mapped to
physical device qubits (layout), SWAP gates must be inserted to satisfy connectivity constraints (routing), and
all gates must be decomposed into the device's native gate set (gate conversion).

The Qrisp adapter provides two levels of transpilation control: a **batteries-included** default pipeline
for common use cases, and a modular `PassManager <https://www.qrisp.eu/reference/Circuit%20Manipulation/Pass%20Management/PassManager.html>`_ system for advanced customisation.


Default transpilation with ``create_iqm_pass_manager`` and ``transpile_to_iqm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to transpile is :func:`.create_iqm_pass_manager`, a factory that builds a fully
configured `PassManager <https://www.qrisp.eu/reference/Circuit%20Manipulation/Pass%20Management/PassManager.html>`_ implementing the complete plasma-sabre pipeline: predicate-based
decomposition, layout, routing, SWAP optimization, and gate conversion to the native CZ + PRX gate set.
The companion convenience function :func:`.transpile_to_iqm` calls the factory and immediately runs the
resulting pipeline on your circuit:

.. code-block:: python

    from iqm.qrisp_iqm import create_iqm_pass_manager, transpile_to_iqm

    # Option A: one-liner
    iqm_ready_qc = transpile_to_iqm(qc, connectivity=backend.connectivity)

    # Option B: create the PassManager once, reuse it
    pm = create_iqm_pass_manager(
        connectivity=backend.connectivity,
        effort=40,
        depth_weight=-0.5,
    )
    iqm_ready_qc = pm.run(qc)

Both functions accept the following tuning parameters:

* ``effort`` — Controls how many random candidates the layout and routing optimisers explore.
  Higher values yield better results but increase compilation time (default: 30).
* ``depth_weight`` — Steers the optimization target: ``-1`` minimizes gate count,
  ``0`` balances gate count and depth, ``+1`` minimizes circuit depth (default: 0).


Custom transpilation with individual passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you need fine-grained control — for example, to experiment with specific layout strategies or
to integrate custom decomposition passes — you can build your own `PassManager <https://www.qrisp.eu/reference/Circuit%20Manipulation/Pass%20Management/PassManager.html>`_ from
individual passes. All layout and routing passes accept the device ``connectivity`` as a list of
``(u, v)`` edges.

**Layout passes**

* ``plasma_layout`` — Stochastic initial qubit placement. Internally tries a VF2++ subgraph
  isomorphism fast path first; falls back to heuristic search if the circuit's interaction graph
  does not embed perfectly into the topology.
* ``vf2pp_layout`` — Exact subgraph isomorphism. Raises an error if no perfect layout exists.
  Use this when you know your circuit's connectivity is compatible with the hardware.
* ``manual_layout`` — Explicit mapping of logical qubits to physical qubit indices,
  e.g. ``[1, 2, 4, 5]``.

**Routing passes**

* ``plasma_route`` — Inserts SWAP gates so that every two-qubit gate acts on physically adjacent
  qubits. Uses the same ``effort`` and ``depth_weight`` parameters as ``plasma_layout``.

**Gate conversion passes**

* ``convert_to_cz`` — Decomposes multi-qubit gates (CX, CY, SWAP, etc.) into CZ-based forms,
  which are native to IQM hardware.
* ``convert_to_prx`` — Converts single-qubit gates to PRX (phased rotation X) gates, the native
  single-qubit gate for IQM devices.

.. code-block:: python

    from qrisp import PassManager, convert_to_cz
    from iqm.qrisp_iqm import plasma_layout, plasma_route

    connectivity = [(0, 1), (1, 2), (2, 3), (3, 4)]

    pm = PassManager()
    pm += plasma_layout(connectivity, effort=40, depth_weight=-0.5)
    pm += plasma_route(connectivity, effort=40, depth_weight=-0.5)
    pm += convert_to_cz()
    pm += convert_to_prx

    transpiled_qc = pm.run(qc)


Submitting circuits for execution
---------------------------------

The :class:`~iqm.qrisp_iqm.backends.IQMBackend` supports two submission modes: **circuit submission** and **pulse submission**.
The backend automatically detects which path to use based on the circuit content.

Circuit submission (gate-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your circuit contains only standard Qrisp gates (no :class:`.IQMPulseOperation` instructions),
:meth:`~iqm.qrisp_iqm.backends.IQMBackend.run` transpiles it to the native gate set and submits it as an IQM circuit
via the IQM Client API.  Use :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async` to get back
an :class:`~iqm.qrisp_iqm.backends.IQMCircuitJob` that you can poll or cancel before retrieving results:

.. code-block:: python

    # Build a standard Qrisp circuit
    qv = QuantumVariable(3)
    h(qv[0])
    cx(qv[0], qv[1])
    cx(qv[0], qv[2])
    measure(qv)
    qc = qv.qs.compile()

    # Synchronous — blocks until done
    result = backend.run(qc, shots=1000)
    print(result)

    # Asynchronous — returns a job handle immediately
    job = backend.run_async(qc, shots=1000)
    print(job.status())          # QUEUED / RUNNING / DONE
    result = job.result()        # blocks until done
    print(result)


.. _pulse_submission:

Pulse submission (pulse-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your circuit contains :class:`.IQMPulseOperation` instructions (e.g. custom delay gates),
the backend routes the job through the Pulla pulse-level interface.  This gives you full control
over native IQM pulse schedules, including custom operations that have no gate-level equivalent.

In a pulse workflow you typically:

1. Use :func:`.extract_iqm_pulse` to compile a Qrisp function into an IQM :class:`~iqm.pulse.Circuit`.
2. Compile the circuit into a **playlist** (pulse schedule) via the Pulla compiler.  You can insert/modify
   custom pulse schedules in this step.
3. Submit the playlist as described in the `Pulla docs <https://docs.iqm.tech/iqm-pulla/>`_.

If you have already built a custom pulla compiler, you can specify it via the ``compiler`` keyword
within the :class:`~iqm.qrisp_iqm.backends.IQMBackend` constructor.

When a circuit contains :class:`.IQMPulseOperation` instructions, :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async`
automatically routes to the pulse path and returns an :class:`~iqm.qrisp_iqm.backends.IQMPulseJob`:

.. code-block:: python

    from iqm.qrisp_iqm import extract_iqm_pulse, quantum_op_to_qrisp_func

    # Get device architecture from the backend
    dqa = backend.iqm_client.get_dynamic_quantum_architecture()

    # Define a custom pulse operation
    delay = quantum_op_to_qrisp_func(QuantumOp(name="delay", params={"duration": (float,)}))

    @extract_iqm_pulse(dqa=dqa)
    def my_pulse_circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        delay(qv[0], duration=300e-9)
        cx(qv[0], qv[1])
        return measure(qv)

    meas_keys, iqm_pulse_qc = my_pulse_circuit()

    # Compile to a playlist
    compiler = backend._pulla.get_standard_compiler()

    job_definition, context = compiler.compile(circuits=[iqm_pulse_qc])
    job = backend.pulla.submit_playlist(job_definition, context=context)
    result = job.wait_for_completion().result()
    print(result)

Or simply use :meth:`~iqm.qrisp_iqm.backends.IQMBackend.run_async` — the backend detects the pulse
operation and handles compilation and submission automatically:

.. code-block:: python

    from iqm.qrisp_iqm import IQMPulseOperation, delay_quantum_op

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    qc.append(IQMPulseOperation(delay_quantum_op, {"duration": 100e-9}), [qc.qubits[0]])    
    qc.measure(qc.qubits)

    job = backend.run_async(qc, shots=1000)   # → IQMPulseJob
    result = job.result()
    print(result)

The key difference: circuit submission treats the job as a set of abstract gates and lets IQM's
server-side compilation handle scheduling; pulse submission gives you direct control over the
playlist so you can inspect, visualise, and customise the pulse schedule before execution.


Visualising the pulse playlist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have compiled a playlist, you can visualise it using IQM's playlist inspection tools.
This is useful for verifying waveforms and timing before submitting to hardware:

.. code-block:: python

    from iqm.pulse.playlist.visualisation.base import inspect_playlist
    from IPython.display import HTML, display

    playlist, context = compiler.compile([iqm_pulse_qc])

    # Generate and display an interactive HTML visualisation
    html_content = inspect_playlist(playlist, [0])
    display(HTML(html_content))


IQM Pulse integration
---------------------

The Qrisp adapter supports **pulse-level** operations, giving you full control over native IQM pulse
instructions such as delays, barriers, and custom gate implementations.

Using native IQM quantum operations in a circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`.quantum_op_to_qrisp_func` to expose an IQM :class:`~iqm.pulse.quantum_ops.QuantumOp`
as a Qrisp gate function that can be used alongside standard operations like ``h``, ``cx``, and
``measure``:

.. code-block:: python

    from iqm.pulse.quantum_ops import QuantumOp
    from iqm.qrisp_iqm import quantum_op_to_qrisp_func

    # Define a delay operation
    delay_quantum_op = QuantumOp(
        name="delay",
        params={"duration": (float,)},
    )

    # Convert to a Qrisp-callable function
    delay = quantum_op_to_qrisp_func(delay_quantum_op)

    # Now use it like any other gate:
    # delay(my_qubit, duration=300e-9)

For advanced use cases, you can also directly construct :class:`.IQMPulseOperation` instances
and append them to a circuit:

.. code-block:: python

    from iqm.qrisp_iqm import IQMPulseOperation

    pulse_op = IQMPulseOperation(
        delay_quantum_op,
        param_dict={"duration": 100e-9},
    )

    from qrisp import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.append(pulse_op, [qc.qubits[0]])

The delay operation already provides a predefined pulse schedule. To learn how to assign
and compile custom pulse-level gates, please consult the 
`Pulla docs <https://docs.iqm.tech/iqm-pulla/>`_.

Jasp tracing with ``extract_iqm_pulse``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`.extract_iqm_pulse` decorator is the primary entry point for pulse-level workflows. It traces
a Qrisp quantum function via Jasp (the JAX-based tracing layer), captures all quantum operations (including
custom :class:`.IQMPulseOperation` instances), transpiles the resulting circuit, and converts it to an IQM
:class:`~iqm.pulse.Circuit`:

.. code-block:: python

    from qrisp import QuantumVariable, h, cx, measure
    from iqm.qrisp_iqm import extract_iqm_pulse, quantum_op_to_qrisp_func

    # Get device architecture from the backend
    dqa = backend.iqm_client.get_dynamic_quantum_architecture()

    delay = quantum_op_to_qrisp_func(QuantumOp(name="delay", params={"duration": (float,)}))

    @extract_iqm_pulse(dqa=dqa)
    def my_circuit():
        qv = QuantumVariable(2)
        h(qv[0])
        delay(qv[0], duration=300e-9)
        cx(qv[0], qv[1])
        return measure(qv)

    meas_keys, iqm_pulse_qc = my_circuit()

The decorator returns measurement key strings alongside the compiled :class:`~iqm.pulse.Circuit`,
ready for playlist compilation and pulse-level execution (see :ref:`pulse submission <pulse_submission>` above).

Custom PassManagers with ``extract_iqm_pulse``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the decorator applies only ``convert_to_cz``. You can supply your own `PassManager <https://www.qrisp.eu/reference/Circuit%20Manipulation/Pass%20Management/PassManager.html>`_
for full control over layout, routing, and gate conversion:

.. code-block:: python

    from qrisp import PassManager
    from iqm.qrisp_iqm import plasma_layout, plasma_route

    custom_pm = PassManager()
    custom_pm += plasma_layout(connectivity, effort=40)
    custom_pm += plasma_route(connectivity, effort=40)
    custom_pm += convert_to_cz()
    custom_pm += convert_to_prx

    @extract_iqm_pulse(dqa=dqa, pass_manager=custom_pm)
    def my_routed_circuit():
        qv = QuantumVariable(5)
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[1], qv[2])
        cx(qv[2], qv[3])
        cx(qv[3], qv[4])
        return measure(qv)

    meas_keys, iqm_circuit = my_routed_circuit()

To skip transpilation entirely (e.g. when your circuit already contains only IQM-native operations),
set ``pass_manager=PassManager()``.


Quantum Error Correction
------------------------

The adapter integrates with the :class:`.DetectorExperiment` class to support quantum error
correction (QEC) workflows. This enables you to define parameterised QEC experiments, compute
logical error rates (LERs), and extract Stim circuits for classical validation.

Below is a sketch of a repetition code memory experiment (see the ``detector_experiment_demo``
tutorial for the complete, runnable implementation):

.. code-block:: python

    from qrisp import QuantumArray, QuantumBool, x, cx, reset, measure
    from qrisp.misc.stim_tools import stim_noise
    from plasma_sabre.qec import DetectorExperiment

    @DetectorExperiment
    def rep_code_experiment(delay_time):
        qubits = QuantumArray(shape=(7,), qtype=QuantumBool())
        data = qubits[::2]      # 4 data qubits
        ancilla = qubits[1::2]  # 3 ancilla qubits

        x(data)                 # initialise logical |1⟩

        # … syndrome extraction rounds: CNOTs, noise injections, measurements, detector parities …
        return detectors, [observable]

    # Compute LER with Stim simulation
    ler_stim = rep_code_experiment.compute_LER(0, shots=10_000, backend=StimBackend())

    # Compute LER on real IQM hardware
    ler_hw = rep_code_experiment.compute_LER(0, shots=10_000, backend=backend)

    # Batched sweep over idle delay times to study decoherence
    ler_sweep = rep_code_experiment.batched_compute_LER(
        [(0,), (500e-9,), (1e-6,)], shots=10_000, backend=backend
    )

The :class:`.DetectorExperiment` decorator gives you ``.compute_LER()``,
``.batched_compute_LER()``, ``.to_stim()``, and ``.to_iqm()`` methods for free, streamlining
the full QEC workflow from classical simulation to hardware execution.


Next steps
----------

- For a curated overview of all Qrisp adapter classes and functions, see :ref:`qrisp_iqm_api`.
- For the complete auto-generated API reference, see :doc:`../API`.
- For details on the IQM backend protocol, see :ref:`integration_guide`.
