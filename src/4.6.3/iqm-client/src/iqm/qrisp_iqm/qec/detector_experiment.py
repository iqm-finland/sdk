# Copyright 2026 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DetectorExperiment — zero-boilerplate QEC detector experiments.

Wraps a Jasp-traceable quantum function that returns detector and observable
parity checks, and automates the full LER workflow:

1. Trace the function → Stim circuit with detector error model (DEM)
2. Sample via Stim or any hardware backend
3. Decode with PyMatching
4. Compare predictions against observables → logical error rate

Also provides batched hardware submission and circuit extraction (to Stim /
to IQM) for inspection and visualisation.

Usage
-----
::

    @DetectorExperiment
    def my_experiment(*args):
        ...
        return detectors, observables

    # Pure Stim (fast)
    ler = my_experiment.compute_LER(arg, shots=10_000)

    # With a hardware backend
    ler = my_experiment.compute_LER(arg, shots=10_000, backend=backend)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from iqm.qrisp_iqm.iqm_converter import qrisp_to_iqm_converter
from jax import jit as jax_jit
import numpy as np
import pymatching
from qrisp import QuantumCircuit
from qrisp.interface import Backend
from qrisp.jasp import extract_stim, make_jaspr
import stim

if TYPE_CHECKING:
    from qrisp.circuit.pass_management import PassManager


class DetectorExperiment:
    r"""Decorator that turns a Jasp function into a detector-experiment handle.

    **What problem does this class solve?**

    A quantum error correction experiment involves many moving parts: you define
    a syndrome-extraction circuit with noise annotations via
    :func:`~qrisp.misc.stim_tools.stim_noise`, return detector and observable
    parity checks via :func:`~qrisp.parity`, then you must trace the circuit,
    extract a detector error model (DEM), sample the circuit, feed detection
    events and the DEM into a decoder (PyMatching), compare predictions against
    observed logical values, and tally the logical error rate (LER).
    Doing this by hand for every experiment — especially when sweeping
    parameters or switching between Stim and hardware backends — is tedious
    and error-prone.

    ``DetectorExperiment`` automates this entire pipeline. Decorate your
    Jasp-traceable experiment function with ``@DetectorExperiment``, and the
    class gives you:

    * **One-shot LER**: :meth:`compute_LER` traces the function, builds the
      DEM from your ``stim_noise`` annotations, samples the circuit (via
      Stim's built-in sampler or any hardware backend), decodes with
      PyMatching, and returns the logical error rate — all in a single call.
    * **Batched sweeps**: :meth:`batched_compute_LER` submits multiple
      parameter sets (e.g. a delay sweep) as a single hardware batch,
      eliminating per-job queue overhead.
    * **Circuit inspection**: :meth:`to_stim` and :meth:`to_iqm` let you
      extract and visualize the underlying Stim or IQM Pulse circuit before
      running on hardware, so you can verify detector / observable placement
      and gate decompositions at a glance.

    **How it works**

    When you call any of the public methods, the class internally:

    1. **Traces** your function via Jasp to obtain a quantum circuit with
       explicit detector and observable annotations.
    2. **Extracts** the Stim circuit (including the DEM built from your
       ``stim_noise`` calls) and, for hardware backends, a Qrisp circuit
       plus a post-processing function.
    3. **Samples** the circuit — either via Stim's fast built-in sampler
       (when no backend is given) or via the provided hardware / simulator
       backend.
    4. **Decodes** the detection events with PyMatching's minimum-weight
       perfect matching, using the DEM that was automatically constructed
       from your noise annotations.
    5. **Compares** the decoder's predictions against the observed logical
       values (your ``observable=True`` parities) and returns the fraction
       of shots where they disagree — the logical error rate.

    Because the DEM is derived directly from the ``stim_noise`` calls in
    your function, you never need to manually construct or synchronise error
    models. The same function works with Stim (for fast, noise-model-based
    simulation) and with real hardware (where physical noise replaces the
    annotated model).

    Parameters
    ----------
    func : callable
        A Jasp-traceable function that implements the detector experiment.
        Must return ``(list[Detector], list[Observable])`` — each detector
        and observable being the result of a :func:`~qrisp.parity` call.

    Examples
    --------
    A minimal repetition code memory experiment:

    ::

        from qrisp import (
            QuantumArray, QuantumBool,
            x, cx, measure, reset, parity,
        )
        from qrisp.misc.stim_tools import stim_noise
        from iqm.qrisp_iqm.qec import DetectorExperiment

        p = 0.01  # physical error strength

        @DetectorExperiment
        def rep_code(delay_time=0.0):
            # ── Allocate qubits ────────────────────────────────────────────
            qubits  = QuantumArray(shape=(7,), qtype=QuantumBool())
            data    = qubits[::2]      # 4 data qubits
            ancilla = qubits[1::2]     # 3 ancilla qubits

            # ── Prepare logical |1_L⟩ ──────────────────────────────────────
            x(data)

            # ── One syndrome round ─────────────────────────────────────────
            # Reset ancillas (with noise on data that idle during reset)
            reset(ancilla)
            stim_noise("X_ERROR", p, ancilla)
            stim_noise("DEPOLARIZE1", p, data)

            # CNOT layer 1:  data[i] → ancilla[i]
            for i in range(3):
                cx(data[i], ancilla[i])
                stim_noise("DEPOLARIZE2", p, data[i], ancilla[i])

            stim_noise("DEPOLARIZE1", p, data[3])   # untouched

            # CNOT layer 2:  data[i+1] → ancilla[i]
            for i in range(3):
                cx(data[i+1], ancilla[i])
                stim_noise("DEPOLARIZE2", p, data[i+1], ancilla[i])

            stim_noise("DEPOLARIZE1", p, data[0])   # untouched

            # Pre-measurement noise
            stim_noise("X_ERROR", p, ancilla)
            stim_noise("X_ERROR", p, data)

            # Measure ancillas → syndrome detectors
            anc_meas = measure(ancilla)
            # First (and only) round: compare to expected initial |0⟩
            synd_det = parity(anc_meas, expectation=0)

            # ── Final data measurement ─────────────────────────────────────
            data_meas = measure(data)

            # ── Measurement-round detectors ────────────────────────────────
            # Tie each ancilla back to its two neighboring data qubits
            det_0 = parity(data_meas[0], anc_meas[0], data_meas[1], expectation=0)
            det_1 = parity(data_meas[1], anc_meas[1], data_meas[2], expectation=0)
            det_2 = parity(data_meas[2], anc_meas[2], data_meas[3], expectation=0)

            # ── Observable ─────────────────────────────────────────────────
            # Logical |1_L⟩ → every data qubit should read 1
            obs = parity(data_meas[3], observable=True, expectation=1)

            return [synd_det, det_0, det_1, det_2], [obs]

    Once decorated, use the provided methods to analyze the experiment:

    ::

        # Compute logical error rate with Stim (fast simulation)
        ler = rep_code.compute_LER(0.0, shots=10_000)
        print(f"Logical error rate: {ler:.4f}")

        # Extract the Stim circuit for visualization
        stim_circuit = rep_code.to_stim(0.0)
        print(stim_circuit)

        # Run on hardware with an IQM backend
        from iqm.qrisp_iqm import IQMBackend, vf2pp_layout
        from qrisp import PassManager, convert_to_cz, convert_to_prx

        backend = IQMBackend(
            device_instance="emerald",
            api_token="YOUR_TOKEN",
            server_url="https://resonance.iqm.tech/",
            pass_manager=PassManager(),
        )
        backend.pm += vf2pp_layout(backend.connectivity)
        backend.pm += convert_to_cz()
        backend.pm += convert_to_prx

        ler_hw = rep_code.compute_LER(0.0, shots=10_000, backend=backend)
        print(f"Hardware LER: {ler_hw:.4f}")

    """

    # .. seealso::
    #
    #     :ref:`detector_experiment_demo`
    #     A step-by-step tutorial covering the repetition code, detectors,
    #     observables, the Stim timeline diagram, batched LER sweeps with
    #     pulse-level delays, and IQM Pulse playlist visualisation.

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_LER(
        self, *args: Any, shots: int = 10_000, backend: Backend | Callable[..., dict[str, int]] | None = None
    ) -> float:
        """Compute the logical error rate for the detector experiment.

        Parameters
        ----------
        *args
            Forwarded to the decorated Jasp function.
        shots : int
            Number of samples / shots.
        backend : callable or object, optional
            Backend to use for sampling. Can be either:
            - A callable with signature ``(qc: QuantumCircuit, shots: int) -> dict[str, int]``
            - An object with a ``run_func`` attribute containing such a callable
            If *None*, Stim's built-in detector sampler is used directly.

        Returns
        -------
        float
            Logical error rate  (``num_errors / shots``).

        """
        num_errors = self._count_logical_errors(*args, shots=shots, backend=backend)
        return num_errors / shots

    def batched_compute_LER(
        self,
        args_list: list[tuple[Any, ...]],
        shots: int = 10_000,
        backend: Backend | Callable[..., dict[str, int]] | None = None,
    ) -> list[float]:
        """Compute the logical error rate for multiple parameter sets in a single backend batch.

        All tracing-based operations (``make_jaspr``, ``Jaspr.to_qc``,
        ``extract_post_processing``, ``extract_stim``) are performed
        **sequentially on the calling thread** before any circuits are submitted,
        because these operations are **not thread-safe**.

        When *backend* is a :class:`~qrisp.interface.Backend` (e.g.
        ``IQMBackend``), all circuits are submitted in a single call to
        ``backend.batch_run_func``, eliminating the per-circuit overhead. For a
        plain callable backend the circuits are executed one-by-one as a
        fallback.

        Parameters
        ----------
        args_list : list[tuple]
            Each element is a tuple of arguments forwarded to the decorated Jasp
            function.  For experiments that take no arguments use ``[()]`` or
            ``[()] * n``.
        shots : int
            Number of samples per experiment.
        backend : Backend or callable, optional
            * A :class:`~qrisp.interface.Backend` — circuits are
              submitted as a single batch via ``batch_run_func``.
            * A plain callable ``(qc, shots) -> dict[str, int]`` — circuits
              are executed sequentially (no batching benefit, but API-compatible).
            * *None* — Stim's built-in detector sampler is used (no hardware
              backend, each experiment is sampled independently).

        Returns
        -------
        list[float]
            A list of logical error rates, one per entry in *args_list*.

        """
        # ── Pure Stim fast-path: no backend, no batching needed ─────────
        if backend is None:
            return [self.compute_LER(*args, shots=shots, backend=None) for args in args_list]
        if not isinstance(backend, Backend):
            raise ValueError("Plain Callables are not currently supported.")

        # ── 1. Prepare everything sequentially (tracing is not thread-safe)
        preparations = []
        for args in args_list:
            # Wrap in a closure so every argument is static (baked into the trace)
            def wrapped(_args: tuple = args) -> Any:
                return self._func(*_args)

            # Stim circuit (for the decoder)
            stim_res = extract_stim(wrapped, detector_order="return_order")()
            stim_circuit = stim_res[-1]

            # Qrisp circuit + post-processing
            jaspr = make_jaspr(wrapped)()
            qc_result = jaspr.to_qc()
            qrisp_circuit = qc_result[-1]
            post_processing = jax_jit(jaspr.extract_post_processing())

            preparations.append((stim_circuit, qrisp_circuit, post_processing))

        # ── 2. Submit all circuits to the backend in one batch ──────────
        batch = [qc for (_, qc, _) in preparations]
        all_counts = backend.run(batch, shots=shots)

        # ── 3. Post-process & decode each experiment ────────────────────
        ler_results = []
        for i, (stim_circuit, _, post_processing) in enumerate(preparations):
            counts = all_counts[i]

            detection_events_list: list = []
            observable_flips_list: list = []
            for bitstring, count in counts.items():
                bits = np.array([c == "1" for c in bitstring])
                post_processed = post_processing(bits)
                detection_events_list.extend([post_processed[:-1]] * count)
                observable_flips_list.extend([post_processed[-1]] * count)

            detection_events = np.array(detection_events_list)
            observable_flips = np.array(observable_flips_list).reshape((shots, -1))

            dem = stim_circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predictions = matcher.decode_batch(detection_events)

            num_errors = int(np.sum(np.any(observable_flips != predictions, axis=1)))
            ler_results.append(num_errors / shots)

        return ler_results

    def to_stim(self, *args: Any) -> stim.Circuit:
        """Extract the Stim circuit for the detector experiment.

        Parameters
        ----------
        *args
            Forwarded to the decorated Jasp function.

        Returns
        -------
        stim.Circuit
            The Stim circuit with detectors and observables.

        """

        # Wrap in a closure so every argument is static (baked into the trace)
        def wrapped() -> Any:
            return self._func(*args)

        # Extract the Stim circuit
        # detector_order must be "return_order" for the decode to work correctly
        res = extract_stim(wrapped, detector_order="return_order")()
        stim_circuit = res[-1]  # stim.Circuit is always last
        return stim_circuit

    def to_qc(self, *args: Any, pm: PassManager | None = None) -> QuantumCircuit:
        """Extract the Qrisp QuantumCircuit for the detector experiment.

        Parameters
        ----------
        *args
            Forwarded to the decorated Jasp function.
        pm : PassManager, optional
            A PassManager to apply transpilation to the circuit.
            If provided, the circuit will be transpiled before returning.

        Returns
        -------
        qrisp.QuantumCircuit
            The Qrisp quantum circuit (transpiled if pm was provided).

        """

        # Wrap in a closure so every argument is static (baked into the trace)
        def wrapped() -> Any:
            return self._func(*args)

        # Create Jaspr and extract QuantumCircuit
        jaspr = make_jaspr(wrapped)()
        qc_result = jaspr.to_qc()
        qrisp_circuit = qc_result[-1]  # QuantumCircuit is always last

        # Apply PassManager if provided
        if pm is not None:
            qrisp_circuit = pm.run(qrisp_circuit)

        return qrisp_circuit

    def to_iqm(self, *args: Any, topology: Any, pm: PassManager | None = None) -> Any:
        """Extract the IQM circuit for the detector experiment.

        Parameters
        ----------
        *args
            Forwarded to the decorated Jasp function.
        topology : ChipTopology or DynamicQuantumArchitecture
            IQM topology object. Can be obtained from:
            - ``pulla.get_chip_topology()`` (IQM Pulse)
            - ``iqm_client.get_dynamic_quantum_architecture()`` (IQM Client)
        pm : PassManager, optional
            A PassManager to apply transpilation before converting to IQM format.
            The circuit must be transpiled to native gates (CZ, U3) before conversion.
            If not provided, the circuit will be converted as-is (and may fail if
            it contains non-native gates).

        Returns
        -------
        iqm.pulse.Circuit
            The IQM circuit object ready for compilation.

        Notes
        -----
        You should provide a PassManager to transpile to native gates:

        ::

            from qrisp import PassManager, convert_to_cz
            from iqm.qrisp_iqm import plasma_layout, route
            from iqm.qrisp_iqm.iqm import get_coupling_map

            connectivity = get_coupling_map(topology)
            pm = PassManager()
            pm.add_pass(plasma_layout(connectivity=connectivity))
            pm.add_pass(route(connectivity=connectivity))
            pm.add_pass(convert_to_cz())

            iqm_circuit = experiment.to_iqm(args, topology=topology, pm=pm)

        """
        # Get the Qrisp circuit (with optional transpilation)
        qrisp_circuit = self.to_qc(*args, pm=pm)

        # Convert to IQM format
        iqm_circuit = qrisp_to_iqm_converter(qrisp_circuit, topology)
        return iqm_circuit

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _count_logical_errors(
        self, *args: Any, shots: int, backend: Backend | Callable[..., dict[str, int]] | None
    ) -> int:
        """Core routine — mirrors the tutorial's ``count_logical_errors``."""

        # Wrap in a closure so every argument is static (baked into the trace)
        def wrapped() -> Any:
            return self._func(*args)

        # 1. Extract the Stim circuit (with PyTree-aware return values)
        # detector_order must be "return_order" for the decode to work correctly
        res = extract_stim(wrapped, detector_order="return_order")()

        stim_circuit = res[-1]  # always last
        # With PyTree reconstruction the return is:
        #   (list[Detector], list[Observable], stim.Circuit)
        # Observables are at index -2; detectors at index -3 — but we only
        # need the stim_circuit and (optionally) the observable count for
        # the pure-Stim path.

        if backend is None:
            # ── Pure Stim path ──────────────────────────────────────────
            detection_events, observable_flips = stim_circuit.compile_detector_sampler().sample(
                shots, separate_observables=True
            )
        else:
            # ── Hardware / custom-backend path ──────────────────────────
            detection_events, observable_flips = self._sample_via_backend(*args, shots=shots, backend=backend)

        # 2. Decode with PyMatching
        dem = stim_circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        predictions = matcher.decode_batch(detection_events)

        # 3. Count mistakes
        num_errors = int(np.sum(np.any(observable_flips != predictions, axis=1)))
        return num_errors

    def _sample_via_backend(self, *args: Any, shots: int, backend: Backend) -> tuple[np.ndarray, np.ndarray]:
        """Run through an arbitrary backend that returns ``dict[str, int]``."""

        # Wrap in a closure so every argument is static (baked into the trace)
        def wrapped() -> Any:
            return self._func(*args)

        jaspr = make_jaspr(wrapped)()
        qc_result = jaspr.to_qc()
        qrisp_circuit: QuantumCircuit = qc_result[-1]

        # Extract run function from backend (support both callable and object with run_func)
        counts = backend.run(qrisp_circuit, shots)

        post_processing = jax_jit(jaspr.extract_post_processing())

        detection_events_list: list = []
        observable_flips_list: list = []

        for bitstring, count in counts.items():
            bits = np.array([c == "1" for c in bitstring])
            post_processed = post_processing(bits)
            detection_events_list.extend([post_processed[:-1]] * count)
            observable_flips_list.extend([post_processed[-1]] * count)

        detection_events = np.array(detection_events_list)
        observable_flips = np.array(observable_flips_list).reshape((shots, -1))

        return detection_events, observable_flips
