.. _User guide:

Qubit Selector user guide
=========================

This guide shows how to choose the physical qubits a circuit runs on with IQM Qubit Selector,
:mod:`iqm.qubit_selector.core`. The core enumerates the placements of the couplings that the QPU allows for and ranks them with
the latest calibration metrics, without transpiling the circuit for any candidate. It works on an
:class:`iqm.pulse.Circuit <iqm.pulse.circuit_operations.Circuit>`, so it does not depend on the SDK the circuit was
written in: IQM Pulse, Qiskit, Cirq and Qrisp circuits are all placed with the same call.

For a notebook you can run and check the output of yourself, see the :doc:`examples <examples>`.

.. note::

   IQM provides access to its quantum computers via IQM Resonance – IQM's quantum cloud service.
   Please head over `to our website <https://iqm.tech/products/iqm-resonance/>`_ to learn more.

The steps are:

1. Connect to a QPU and fetch its architecture and calibration metrics.
2. Convert a circuit from your SDK to an :class:`iqm.pulse.Circuit <iqm.pulse.circuit_operations.Circuit>`.
3. Rank layouts with :func:`~iqm.qubit_selector.core.selector.select_layout` and
   :func:`~iqm.qubit_selector.core.selector.score_layouts`.
4. Tune the ranking: readout, excluded qubits, search budget, what-if metrics.
5. Run the circuit on the selected layout.


Installation
------------

Install the ``iqm-qubit-selector`` distribution package from the Python Package Index (PyPI). It brings IQM Client,
IQM Pulse and IQM Station Control Client along, which is all :mod:`iqm.qubit_selector.core` needs for circuits
written in IQM Pulse:

.. code-block:: bash

   $ uv pip install iqm-qubit-selector

Qiskit is not installed by default. To prepare circuits in Qiskit, Cirq or Qrisp, connect with
:class:`~iqm.qiskit_iqm.iqm_provider.IQMProvider`, or run the example notebooks, install the ``notebook`` feature as
well. It brings IQM Client with its Qiskit, Cirq and Qrisp adapters, and Jupyter:

.. code-block:: bash

   $ uv pip install "iqm-qubit-selector[notebook]"


Authentication
--------------

Set the ``IQM_TOKEN`` environment variable to the API token obtained from the IQM Server web dashboard.
:class:`~iqm.iqm_client.iqm_client.IQMClient` and :class:`~iqm.qiskit_iqm.iqm_provider.IQMProvider` read it from
there, so the token does not have to appear in your code or notebooks.


Connecting to a QPU
-------------------

Connect with :class:`~iqm.qiskit_iqm.iqm_provider.IQMProvider`. The backend carries both the client and the
architecture of the QPU:

.. code-block:: python

    from iqm.qiskit_iqm import IQMProvider

    iqm_server_url = "https://resonance.iqm.tech/"
    backend = IQMProvider(iqm_server_url, quantum_computer="garnet").get_backend()
    client = backend.client

Without Qiskit, connect with :meth:`~iqm.iqm_client.iqm_client.IQMClient.from_url` instead.


Architecture and calibration metrics
------------------------------------

The selector needs two inputs besides the circuit:

* the :class:`~iqm.station_control.interface.models.dynamic_quantum_architecture.DynamicQuantumArchitecture` of the
  QPU, which says which gates act on which qubits, and
* a :class:`~iqm.qubit_selector.core.metrics_data.MetricsData` with the gate and readout fidelities of the latest
  calibration.

:meth:`~iqm.qubit_selector.core.metrics_data.MetricsDataManager.get_metrics_data` fetches the metrics of a calibration
set and keeps what the cost model uses: CZ fidelities per qubit pair (``two_qubit``), PRX fidelities per qubit
(``single_qubit``) and readout metrics per qubit (``readout``, keyed by
:class:`~iqm.qubit_selector.core.metrics_data.ReadoutMode`). Pass the calibration set of the architecture, so that the
metrics match it; without one, the server's current default calibration set is used:

.. code-block:: python

    from iqm.qubit_selector.core import MetricsDataManager

    arch = backend.architecture  # or client.get_dynamic_quantum_architecture()
    manager = MetricsDataManager()
    metrics = manager.get_metrics_data(client, arch.calibration_set_id)

    print("Qubits:", len(arch.qubits))
    print("Two-qubit fidelity entries:", len(metrics.two_qubit))
    print("Readout metrics:", [str(mode) for mode in metrics.readout])

:meth:`~iqm.qubit_selector.core.metrics_data.MetricsDataManager.get_full_metrics_data` returns everything it parses
as a :class:`~iqm.qubit_selector.core.metrics_data.FullMetricsData`, including Clifford fidelities and ``T1``/``T2``
times. These are for reference only and are not used for ranking:

.. code-block:: python

    full_metrics = manager.get_full_metrics_data(client, arch.calibration_set_id)

    qubit = arch.qubits[0]
    print(f"{qubit}: T1 = {full_metrics.t1.get(qubit)} us, T2 = {full_metrics.t2.get(qubit)} us")


Preparing the circuit
---------------------

The selector only looks at which qubits a circuit uses and how many gates act on each qubit and qubit pair. Any
circuit that can be expressed as an :class:`iqm.pulse.Circuit <iqm.pulse.circuit_operations.Circuit>` can therefore
be placed. The circuit's qubit names only identify its qubits; the selector decides which physical qubits they go to.

The examples below all build a GHZ circuit on six qubits.

IQM Pulse
~~~~~~~~~

:class:`~iqm.pulse.circuit_operations.CircuitOperationList` builds a circuit from native IQM operations, on
placeholder qubits ``QB1`` to ``QBn``. A ``measure`` operation acts on any number of qubits, so it is added with
:meth:`~iqm.pulse.circuit_operations.CircuitOperationList.add_op`:

.. code-block:: python

    from math import pi

    from iqm.pulse import Circuit
    from iqm.pulse.circuit_operations import CircuitOperationList

    nqubits = 6

    builder = CircuitOperationList(num_qubits=nqubits)
    builder.prx(pi / 2, 0.0, 0)
    for i in range(1, nqubits):
        builder.prx(pi / 2, 0.0, i)
        builder.cz(i - 1, i)
    builder.add_op("measure", list(range(nqubits)), "m")

    circuit = Circuit(name="ghz", instructions=tuple(builder))

Qiskit
~~~~~~

Transpile the circuit to the backend with :func:`~iqm.qiskit_iqm.iqm_naive_move_pass.transpile_to_IQM`, then convert
it with :func:`~iqm.qiskit_iqm.pulse_level_utils.qiskit_to_iqm`, which returns the converted circuits and a compiler.
The transpiled circuit already sits on physical qubits that Qiskit picked; the selector treats those names as
placeholders and may move the circuit elsewhere:

.. code-block:: python

    from iqm.qiskit_iqm import transpile_to_IQM
    from iqm.qiskit_iqm.pulse_level_utils import qiskit_to_iqm
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(nqubits)
    qc.h(0)
    for i in range(1, nqubits):
        qc.cx(i - 1, i)
    qc.measure_all()

    qc_transpiled = transpile_to_IQM(qc, backend, optimization_level=3)
    converted, _ = qiskit_to_iqm(backend, qc_transpiled)
    circuit = converted[0]

Cirq
~~~~

:func:`~iqm.cirq_iqm.serialize.serialize_circuit` converts a Cirq circuit in native IQM gates (``PhasedXPowGate`` to
``prx``, ``CZ`` to ``cz``, measurements to ``measure``) on ``NamedQid`` qubits named like IQM qubits:

.. code-block:: python

    import cirq

    from iqm.cirq_iqm.serialize import serialize_circuit

    qubits = [cirq.NamedQid(f"QB{i + 1}", dimension=2) for i in range(nqubits)]
    cirq_circuit = cirq.Circuit()
    cirq_circuit.append(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.0).on(qubits[0]))
    for i in range(1, nqubits):
        cirq_circuit.append(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.0).on(qubits[i]))
        cirq_circuit.append(cirq.CZ(qubits[i - 1], qubits[i]))
    cirq_circuit.append(cirq.measure(*qubits, key="m"))

    circuit = serialize_circuit(cirq_circuit)

Qrisp
~~~~~

:func:`~iqm.qrisp_iqm.transpile_to_iqm` converts the circuit to native IQM gates and needs the connectivity of the
circuit's qubits as pairs of indices. Since the selector chooses the physical qubits afterwards, allow every pair to
interact so that the transpiler adds no SWAPs. :func:`~iqm.qrisp_iqm.qrisp_to_iqm_converter` then produces the
:class:`iqm.pulse.Circuit <iqm.pulse.circuit_operations.Circuit>`:

.. code-block:: python

    from itertools import combinations

    from iqm.qrisp_iqm import qrisp_to_iqm_converter, transpile_to_iqm
    from qrisp import QuantumCircuit as QrispQuantumCircuit

    qrisp_qc = QrispQuantumCircuit(nqubits)
    qrisp_qc.h(0)
    for i in range(1, nqubits):
        qrisp_qc.cx(i - 1, i)
    qrisp_qc.measure(list(range(nqubits)))

    qrisp_transpiled = transpile_to_iqm(qrisp_qc, list(combinations(range(nqubits), 2)))
    circuit = qrisp_to_iqm_converter(qrisp_transpiled, arch)


Selecting layouts
-----------------

:func:`~iqm.qubit_selector.core.selector.select_layout` returns the best ``num_layouts`` layouts as
``(layout, error)`` pairs, best first. The error is the estimated probability that at least one gate or readout
fails, so lower is better. The call is the same for every SDK; the results can differ slightly because each SDK
decomposes the GHZ circuit into a different number of gates:

.. code-block:: python

    from iqm.qubit_selector.core import select_layout

    for layout, error in select_layout(circuit, arch, metrics, num_layouts=5):
        print(f"{layout}  error {error:.3%}")

The QPU can also be given as a client, and the selector fetches the architecture and metrics of the default
calibration set itself:

.. code-block:: python

    best_layout, _ = select_layout(circuit, client=client)[0]

Give either a client alone, or ``arch`` and ``metrics`` without one. The selector fetches on every call, so when placing
several circuits, or scoring with freshly measured metrics, fetch ``arch`` once, as above, and pass it with the
metrics.

Reading a layout
~~~~~~~~~~~~~~~~

A layout lists one physical qubit per circuit qubit. Position ``i`` belongs to the ``i``-th circuit qubit in this
order: the qubits with two-qubit gates first, then the others, each group sorted by name. Names sort as strings, so
``QB10`` comes before ``QB2``. The helper below turns a layout into an explicit mapping; use it whenever you place
the circuit yourself:

.. code-block:: python

    from iqm.qubit_selector.core.ir import active_qubits, interactions_from_circuit


    def layout_order(circuit):
        """Return the circuit qubit that each position of a layout belongs to."""
        core = sorted({qubit for pair in interactions_from_circuit(circuit) for qubit in pair})
        return core + sorted(active_qubits(circuit) - set(core))


    def layout_mapping(circuit, layout):
        """Return which physical qubit each circuit qubit goes to."""
        return dict(zip(layout_order(circuit), layout, strict=True))


    best_layout, _ = select_layout(circuit, arch, metrics)[0]
    print(layout_mapping(circuit, best_layout))


Detailed scores
---------------

Each layout is scored as an error budget in nats: every gate or readout with fidelity ``F`` adds ``-ln(F)``, and
``total`` is the sum. A difference of ``d`` nats between two layouts means the better one is ``exp(d)`` times more
likely to run without error. The error of :func:`~iqm.qubit_selector.core.selector.select_layout` is
``1 - exp(-total)``; for deep circuits it approaches 100% for every candidate, while ``total`` still tells them apart.

:func:`~iqm.qubit_selector.core.selector.score_layouts` returns every candidate as a
:class:`~iqm.qubit_selector.core.cost.LayoutScore`, sorted by ``total``, with the two-qubit, single-qubit and readout
terms that make it up:

.. code-block:: python

    from iqm.qubit_selector.core import score_layouts

    scores = score_layouts(circuit, arch, metrics)
    for score in scores[:5]:
        print(
            f"{score.layout}  2Q {score.two_qubit:.4f}  1Q {score.single_qubit:.4f}  "
            f"RO {score.readout:.4f}  total {score.total:.4f}  error {score.error:.3%}"
        )


Readout
-------

``readout_mode`` selects which readout metric enters the score:

* ``ReadoutMode.NONE`` (default): readout is not scored.
* ``ReadoutMode.FIDELITY``: single-shot readout fidelity. Use it when you do not mitigate readout errors.
* ``ReadoutMode.QNDNESS``: how well the measured state survives the measurement. Use it when the circuit keeps using
  a qubit after measuring it.

.. code-block:: python

    from iqm.qubit_selector.core import ReadoutMode

    for mode in ReadoutMode:
        best = score_layouts(circuit, arch, metrics, readout_mode=mode)[0]
        print(f"{mode!s:>8}: {best.layout}  readout {best.readout:.4f} nats, error {best.error:.3%}")


Excluding qubits
----------------

``remove_qubits_list`` keeps qubits out of every layout, for example qubits you know to be faulty although their
metrics do not show it. They are removed from the coupling graph before the search, so they do not connect other
qubits either:

.. code-block:: python

    qubits_to_remove = ["QB1", "QB2"]
    results = select_layout(circuit, arch, metrics, num_layouts=5, remove_qubits_list=qubits_to_remove)

    for layout, _ in results:
        assert not set(layout) & set(qubits_to_remove)


Search budget
-------------

``max_num_layouts`` caps the number of candidate layouts that are scored. The default, 20000, spreads the candidates
over the whole QPU for chips of up to about 150 qubits. A larger budget finds more candidates in each region, at the
cost of a slower search:

.. code-block:: python

    scores = score_layouts(circuit, arch, metrics, max_num_layouts=200000)

When the budget is too small to cover every candidate, the enumerators sample with a fixed seed, so the same inputs
always give the same ranking. Pass ``seed`` to :func:`~iqm.qubit_selector.core.enumerator.make_enumerator` to draw a
different sample; see `Building blocks`_.


What-if metrics
---------------

:class:`~iqm.qubit_selector.core.metrics_data.MetricsData` is a plain dataclass, so you can edit the metrics to see
how the ranking reacts. Two-qubit fidelities are keyed by :func:`~iqm.qubit_selector.core.metrics_data.two_qubit_key`
in both orientations. Here the couplers the best layout uses are made much worse, and the selector moves the circuit
elsewhere:

.. code-block:: python

    from dataclasses import replace

    from iqm.qubit_selector.core import two_qubit_key

    best_layout, _ = select_layout(circuit, arch, metrics)[0]
    mapping = layout_mapping(circuit, best_layout)

    degraded = dict(metrics.two_qubit)
    for pair in interactions_from_circuit(circuit):
        a, b = (mapping[qubit] for qubit in pair)
        degraded[two_qubit_key(a, b)] = degraded[two_qubit_key(b, a)] = 0.5

    what_if = replace(metrics, two_qubit=degraded)
    print("With degraded couplers:", select_layout(circuit, arch, what_if)[0][0])

You can also build :class:`~iqm.qubit_selector.core.metrics_data.MetricsData` from scratch, for example from your own
characterisation data. :func:`~iqm.qubit_selector.core.metrics_data.two_qubit_fidelities` stores each pair in both
orientations. A readout mode you never score may be left out. A qubit or pair without a metric gives every layout
that uses it infinite error, so it is never selected, and a fidelity outside ``(0, 1]`` raises ``ValueError``:

.. code-block:: python

    from iqm.qubit_selector.core import MetricsData, two_qubit_fidelities

    custom_metrics = MetricsData(
        two_qubit=two_qubit_fidelities({("QB1", "QB2"): 0.995, ("QB2", "QB3"): 0.990}),
        single_qubit={"QB1": 0.999, "QB2": 0.998, "QB3": 0.997},
        readout={ReadoutMode.FIDELITY: {"QB1": 0.98, "QB2": 0.97, "QB3": 0.96}},
    )


Building blocks
---------------

:func:`~iqm.qubit_selector.core.selector.score_layouts` chains four steps, each available on its own:

1. :mod:`iqm.qubit_selector.core.ir` reads the circuit: two-qubit gate counts per pair
   (:func:`~iqm.qubit_selector.core.ir.interactions_from_circuit`), single-qubit gate counts per qubit and the
   measured qubits.
2. :func:`~iqm.qubit_selector.core.graph_builder.build_coupling_graph` turns the architecture into a ``networkx``
   graph of coupled qubits.
3. :func:`~iqm.qubit_selector.core.enumerator.make_enumerator` picks the enumerator for that graph:
   :class:`~iqm.qubit_selector.core.enumerator.VF2Enumerator` searches the graph for placements of the circuit's
   interactions, and :class:`~iqm.qubit_selector.core.enumerator.AllToAllEnumerator` picks qubit sets directly when
   every pair is coupled.
4. A cost model, here :class:`~iqm.qubit_selector.core.cost.FastEdgeCost`, scores the layouts and sorts them.

.. code-block:: python

    from iqm.qubit_selector.core import FastEdgeCost, make_enumerator
    from iqm.qubit_selector.core.graph_builder import build_coupling_graph
    from iqm.qubit_selector.core.ir import measured_qubits, single_qubit_gate_counts

    coupling_graph = build_coupling_graph(arch)
    interaction_graph = interactions_from_circuit(circuit)
    order = layout_order(circuit)

    enumerator = make_enumerator(coupling_graph, max_num_layouts=20000)
    layouts = enumerator.enumerate(interaction_graph, len(order))

    cost = FastEdgeCost(metrics, ReadoutMode.NONE)
    ranked = cost.sort_by_scores(
        layouts, interaction_graph, single_qubit_gate_counts(circuit), measured_qubits(circuit), order
    )
    print("Best:", ranked[0])

To rank with your own cost model, subclass :class:`~iqm.qubit_selector.core.cost.CostCalculatorBase` and implement
:meth:`~iqm.qubit_selector.core.cost.CostCalculatorBase.score_single_layout`. It returns a
:class:`~iqm.qubit_selector.core.cost.LayoutCost` (or a subclass), and the base class takes care of scoring and
sorting. This one ranks on the two-qubit gates only:

.. code-block:: python

    from iqm.qubit_selector.core import CostCalculatorBase, LayoutCost


    class TwoQubitOnlyCost(CostCalculatorBase[LayoutCost]):
        """Rank layouts by their two-qubit gate errors alone."""

        def __init__(self, metrics):
            self.fast_cost = FastEdgeCost(metrics)

        def score_single_layout(self, layout, interaction_graph, sqg_counts, measured, logical_qubits):
            score = self.fast_cost.score_single_layout(layout, interaction_graph, {}, set(), logical_qubits)
            return LayoutCost(layout=layout, total=score.two_qubit)


    ranked = TwoQubitOnlyCost(metrics).sort_by_scores(
        layouts, interaction_graph, single_qubit_gate_counts(circuit), measured_qubits(circuit), order
    )


Running on the selected layout
------------------------------

Layout selection does not depend on the SDK, but running the circuit does. With Qiskit, pin the circuit's virtual
qubits to the selected physical qubits with ``initial_layout`` when transpiling it.

For that, the selector should see the circuit's virtual qubits rather than the physical qubits of a default
placement. Convert the circuit to native IQM gates without placing it, with virtual qubit ``i`` named ``QB{i+1}``,
and turn the selected layout into the physical qubit index of each virtual qubit:

.. code-block:: python

    from qiskit import transpile

    qc_native = transpile(qc, basis_gates=["r", "cz"], optimization_level=3)
    circuit = backend.serialize_circuit(qc_native, {i: f"QB{i + 1}" for i in range(nqubits)})

    best_layout, _ = select_layout(circuit, arch, metrics)[0]
    mapping = layout_mapping(circuit, best_layout)
    best_layout_idx = [backend.qubit_name_to_index(mapping[f"QB{i + 1}"]) for i in range(nqubits)]

Then transpile with that layout pinned, and run.
:func:`~iqm.qiskit_iqm.iqm_naive_move_pass.transpile_to_IQM` works on crystal and Star QPUs alike; on a Star QPU it
also adds the ``MOVE`` gates:

.. code-block:: python

    qc_hw = transpile_to_IQM(qc, backend, initial_layout=best_layout_idx, optimization_level=3)
    counts = backend.run(qc_hw, shots=2000).result().get_counts()

The ``qubit_selector_core_qiskit_example`` notebook in the :doc:`examples <examples>` runs a GHZ circuit this way on a
crystal and a Star QPU, and compares it with the placement ``transpile_to_IQM`` picks without ``initial_layout``.


.. _star_architectures:

Star architectures
------------------

On a Star QPU, such as ``sirius``, the qubits do not couple to each other directly. Every two-qubit gate goes through
a computational resonator: a ``MOVE`` gate moves the state of one qubit into the resonator, a CZ acts between the
resonator and a second qubit, and another ``MOVE`` moves the state back. The calls stay the same:

.. code-block:: python

    star_backend = IQMProvider(iqm_server_url, quantum_computer="sirius").get_backend()
    star_arch = star_backend.architecture
    star_metrics = manager.get_metrics_data(star_backend.client, star_arch.calibration_set_id)

    print("Computational resonators:", star_arch.computational_resonators)
    results = select_layout(circuit, star_arch, star_metrics, num_layouts=3)

Three things happen behind the scenes:

* **Fictional CZ fidelities.** A logical ``CZ(q1, q2)`` runs as ``MOVE(q2, r) CZ(q1, r) MOVE(q2, r)`` through a
  shared resonator ``r``, so its fidelity is ``F_cz(q1, r) * F_move(q2, r) ** 2``, with the better choice of
  ``MOVE`` qubit kept.
  :meth:`~iqm.qubit_selector.core.metrics_data.MetricsDataManager.get_metrics_data` returns these qubit-qubit
  fidelities in ``two_qubit``, computed from the measured resonator fidelities by
  :func:`~iqm.qubit_selector.core.metrics_data.fictional_two_qubit_fidelities`.
* **A collapsed coupling graph.** :func:`~iqm.qubit_selector.core.graph_builder.build_coupling_graph` replaces the
  resonator loci with the qubit pairs that ``MOVE`` + CZ can couple. With a single resonator every pair can, so the
  graph is complete and :func:`~iqm.qubit_selector.core.enumerator.make_enumerator` picks
  :class:`~iqm.qubit_selector.core.enumerator.AllToAllEnumerator`, which ranks sets of qubits directly.
* **MOVE gates are removed from the circuit.** A Qiskit circuit transpiled with
  :func:`~iqm.qiskit_iqm.iqm_naive_move_pass.transpile_to_IQM` for a Star QPU contains ``MOVE`` gates. The selector
  removes them to recover the qubit-qubit interactions, so the circuit can be passed as it is.

For the same reason, a Qrisp circuit for a single-resonator Star QPU can be transpiled with all-to-all connectivity,
as in the Qrisp example above.


Next steps
----------

* The :doc:`API reference <API>` documents every function and class of :mod:`iqm.qubit_selector.core`.
* The :doc:`examples <examples>` include ``qubit_selector_core_qiskit_example``, a ready-to-run notebook that
  places a GHZ circuit on a crystal and a Star QPU and compares it with the default placement.
