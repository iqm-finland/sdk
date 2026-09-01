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
"""IQM Backend implementation for Qrisp using plasma-sabre transpilation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import logging
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID
import warnings

from iqm.iqm_client import IQMClient
from iqm.iqm_client.models import (
    CircuitCompilationOptions,
    DDMode,
    HeraldingMode,
    MoveGateValidationMode,
)
from qrisp.circuit import CircuitPass, PassManager, QuantumCircuit
from qrisp.interface import Backend, MeasurementResult
from qrisp.misc.exceptions import QrispDeprecationWarning

from .iqm_circuit_job import IQMCircuitJob
from .iqm_job import IQMJob
from .iqm_pulse_job import IQMPulseJob
from ..iqm_converter import get_coupling_map, qrisp_to_iqm_converter
from ..pulse_operation import IQMPulseOperation

if TYPE_CHECKING:
    from qrisp.circuit import QuantumCircuit

    from exa.common.data.setting_node import SettingNode
    from iqm.cpc.compiler.compiler import Compiler
    from iqm.cpc.core.observation.observation_loading_rules import RuleType
    from iqm.pulla.pulla import Pulla
    from iqm.pulse import Circuit

from iqm.qrisp_iqm.passes.default_pass_manager import create_iqm_pass_manager

_DEFAULT_SHOTS: int = 1000
"""By default we will request this many shots of each circuit."""


class IQMBackend(Backend):
    """Qrisp backend for executing circuits on IQM quantum hardware.

    After transpilation, :meth:`run_async` automatically detects whether any of the circuits
    contains :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation` instructions, and routes the
    job to the appropriate submission path:

    - **No pulse ops** → submitted as a gate-level :class:`~.iqm_circuit_job.IQMCircuitJob`.
    - **Has pulse ops** → compiled locally down to pulse level, and submitted as a
      :class:`~.iqm_pulse_job.IQMPulseJob`.

    The ``server_url`` and ``device_instance`` parameters can alternatively be provided as
    the environment variables :envvar:`IQM_SERVER_URL` and :envvar:`IQM_QUANTUM_COMPUTER`.
    If the server requires user authentication, you can provide it either using environment
    variables or as keyword arguments. The user authentication kwargs are passed
    through to :class:`~iqm.iqm_client.iqm_client.IQMClient` as is, and are documented there.

    Args:
        server_url: URL for connecting to a quantum computer.
            If None, use environment variable :envvar:`IQM_SERVER_URL`.
        device_instance: IQM quantum computer identifier (e.g., ``"garnet"``).
            If None, use environment variable :envvar:`IQM_QUANTUM_COMPUTER`.
            If it is not defined, use the default quantum computer on the server.
        user_auth_args: User authentication information, e.g. an API token.
        transpiler:
            .. deprecated:: Use ``pass_manager`` instead.

            Transforms quantum circuits for hardware compatibility.  If provided, a
            :class:`~qrisp.DeprecationWarning` is raised and the callable is
            wrapped in a :class:`~qrisp.CircuitPass` and installed as the
            sole pass in the :class:`~qrisp.PassManager`.  Must not be used
            together with ``pass_manager``.
        calibration_set_id: ID of the calibration set to use.
            ``None`` means the server's current default calibration set is used.
        use_metrics:
            Iff ``True``, the backend will query the server for calibration
            data and related quality metrics, and pass these to the
            default pass manager.  **Currently a no-op** — the flag is
            stored and will be wired into both submission paths in a future
            release.
        pass_manager:
            PassManager for circuit transpilation. Accepted values:

            - ``None``: Builds a pre-configured PassManager that performs
              routing, layout and other optimization to bring good
              performance without configuration overhead.
            - A :class:`~qrisp.PassManager` instance: Used directly for
              transpilation, bypassing the default pipeline entirely.
        use_timeslot: If True, submits the job to the timeslot queue.
            If False, the job is submitted to the normal on-demand queue.
        compilation_options: Options for compiling circuits to pulse schedules.
            On the pulse-level execution path, the :attr:`CircuitCompilationOptions.max_circuit_duration_over_t2`
            attribute has no effect.
            ``None`` means use the default options.
        compiler:
            IQM circuit-to-pulse compiler to use for pulse-level submissions.
            ``None`` uses the standard compiler.

    Examples:
    **Automatic transpilation** (default PassManager)::

        >>> from iqm.qrisp_iqm import IQMBackend
        >>> backend = IQMBackend(
        ...     token="YOUR_API_TOKEN",
        ...     device_instance="garnet"
        ... )
        >>> from qrisp import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.measure(qc.qubits)
        >>> result = backend.run(qc, shots=1000)

    **Custom pass pipeline via :attr:`pm`**::

        >>> from qrisp import PassManager, convert_to_cz, convert_to_prx
        >>> from iqm.qrisp_iqm import vf2pp_layout
        >>> backend = IQMBackend(
        ...     token="YOUR_API_TOKEN",
        ...     device_instance="garnet",
        ...     pass_manager=PassManager()  # start with an empty manager
        ... )
        >>> coupling_map = [(0, 1), (1, 2), (2, 3)]
        >>> backend.pm += vf2pp_layout(coupling_map)
        >>> backend.pm += convert_to_cz()
        >>> backend.pm += convert_to_prx
        >>> result = backend.run(my_circuit, shots=1000)

    The ``pass_manager=PassManager()`` constructor argument gives you full
    control.  You can also start from ``None`` and extend the pipeline
    (or even insert passes at arbitrary positions via
    :meth:`~qrisp.PassManager.insert_before`).

    **Pulse-level execution** (automatic routing)::

        >>> from iqm.qrisp_iqm import IQMBackend
        >>> from iqm.qrisp_iqm.pulse_operation  import IQMPulseOperation
        >>> backend = IQMBackend(
        ...     token="YOUR_API_TOKEN",
        ...     device_instance="garnet"
        ... )
        >>> # Circuit with IQMPulseOperation is compiled locally to a pulse-level job.
        >>> job = backend.run_async(my_pulse_circuit, shots=1000)
        >>> result = job.result()

    """

    def __init__(  # noqa: PLR0913
        self,
        server_url: str | None = None,
        *,
        device_instance: str | None = None,
        transpiler: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
        calibration_set_id: UUID | str | None = None,
        use_metrics: bool = False,
        pass_manager: PassManager | None = None,
        use_timeslot: bool = False,
        compilation_options: CircuitCompilationOptions | None = None,
        compiler: Compiler | None = None,
        **user_auth_args,
    ) -> None:
        # api_token to token
        if token := user_auth_args.pop("api_token", None):
            if "token" in user_auth_args:
                raise ValueError('"api_token" is a deprecated alias for "token", do not provide both.')
            user_auth_args["token"] = token

        # Handle deprecated transpiler argument
        if transpiler is not None and pass_manager is not None:
            raise ValueError(
                "Cannot specify both 'transpiler' and 'pass_manager'. "
                "The 'transpiler' argument is deprecated; use 'pass_manager' instead."
            )

        if transpiler is not None:
            warnings.warn(
                "The 'transpiler' argument of IQMBackend is deprecated. Use 'pass_manager' with a CircuitPass instead.",
                QrispDeprecationWarning,
                stacklevel=2,
            )
            # Wrap the callable in a CircuitPass and build a PassManager
            circuit_pass = CircuitPass(transpiler)
            pass_manager = PassManager([circuit_pass])

        super().__init__(name=f"IQM-{device_instance or server_url}")

        self.iqm_server_url = server_url
        self.quantum_computer = device_instance
        self.user_auth_args = user_auth_args.copy()

        # defaults for options which are mostly job-specific, and can be overridden by the run_* methods
        self._compiler = compiler
        self.compilation_options = compilation_options
        self.use_timeslot = use_timeslot

        # Initialize IQM Client for API communication and DQA queries
        iqm_client = IQMClient(
            server_url,
            quantum_computer=device_instance,
            **user_auth_args,
        )
        self.iqm_client = iqm_client

        # calibration set to UUID | None
        if calibration_set_id is not None and not isinstance(calibration_set_id, UUID):
            calibration_set_id = UUID(calibration_set_id)

        # get the DQA, build the coupling map
        dqa = iqm_client.get_dynamic_quantum_architecture(calibration_set_id)
        self._dqa = dqa
        self._connectivity = get_coupling_map(dqa)
        self._calibration_set_id: UUID = dqa.calibration_set_id  # current calibration set

        # we will need to check if the default calset has been changed before using it
        self._use_default_calibration_set = calibration_set_id is None

        # Eagerly resolve the pass manager so users can inspect/modify self.pm
        # before the first backend call.
        if pass_manager is None:
            self.pm: PassManager = create_iqm_pass_manager(self._connectivity)
        elif isinstance(pass_manager, PassManager):
            self.pm = pass_manager
        else:
            raise TypeError(f"Expected a PassManager instance or None, got {pass_manager!r}")
        """The public pass manager for circuit transpilation. Set via the
        ``pass_manager`` :meth:`__init__` parameter. Can be a pre-built
        :class:`~qrisp.PassManager` instance or ``None``
        (the default pipeline). After
        construction, passes can be added incrementally via ``+=``.
        """

        self.use_metrics = use_metrics  # stored; implementation planned
        # metrics = client._get_calibration_quality_metrics(architecture.calibration_set_id) if use_metrics else None

        # Pulla is initialized lazily on first pulse submission
        # to avoid unnecessary import / connection overhead
        self._pulla_client: Pulla | None = None

    def _check_default_calset(self) -> None:
        """Warn if we are using the default calset and it has changed on the server."""
        if self._use_default_calibration_set:
            # check if the default calset has changed
            default_calset_id = self.iqm_client.get_dynamic_quantum_architecture(None).calibration_set_id
            if self._calibration_set_id != default_calset_id:
                # just warn, keep using the old default calset
                warnings.warn(
                    f"Server default calibration set has changed from {self._calibration_set_id} "
                    f"to {default_calset_id}. Create a new IQMBackend if you wish to transpile the "
                    "circuits using the new calibration set."
                )

    @property
    def _pulla(self) -> Pulla:
        """The Pulla client instance, lazily initialized.

        On first access, creates a :class:`.Pulla` connection to the
        same IQM Server URL as the one used by IQMClient.

        Returns:
            The pulse-level IQM client.

        """
        if self._pulla_client is None:
            from iqm.pulla.pulla import Pulla  # noqa: PLC0415

            self._pulla_client = Pulla(
                self.iqm_server_url,
                quantum_computer=self.quantum_computer,
                **self.user_auth_args,
            )
        return self._pulla_client

    @staticmethod
    def _apply_compilation_options_to_settings(
        settings: SettingNode,
        options: CircuitCompilationOptions | None = None,
    ) -> dict[str, Any]:
        """Apply circuit compilation options to the compiler settings tree.

        The IQM compiler exposes compilation and execution options (heralding, dynamical decoupling,
        active reset etc.)  through the settings tree.  This method translates
        :class:`.CircuitCompilationOptions` fields into the corresponding settings tree nodes.

        Args:
            settings: Settings tree obtained from :meth:`Compiler.get_settings`.
            options: Options for circuit compilation. ``None`` means use the default options.

        Returns:
            Initial compiler context.

        """
        # TODO this belongs in iqm.cpc, it's not specific to Qrisp
        # also done on server-side!
        if options is None:
            return {}

        stages = settings.stages
        stages.timebox_stage.prepend_reset.active_reset_cycles = options.active_reset_cycles
        stages.timebox_stage.prepend_heralding.add_heralding = options.heralding_mode == HeraldingMode.ZEROS

        move_val = options.move_gate_validation
        if move_val == MoveGateValidationMode.NONE:
            stages.circuit_stage.validate_circuits.move_gate_validation = False
            stages.circuit_stage.validate_circuits.validate_prx = False
        else:
            stages.circuit_stage.validate_circuits.move_gate_validation = True
            stages.circuit_stage.validate_circuits.validate_prx = move_val == MoveGateValidationMode.ALLOW_PRX

        move_tracking = options.move_gate_frame_tracking
        stages.schedule_stage.apply_move_gate_phase_corrections.move_gate_frame_tracking = move_tracking.value

        stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = options.dd_mode == DDMode.DISABLED
        # NOTE providing a DD strategy object cannot be done using Settings, it goes into the compiler context
        return {"DDStrategy": options.dd_strategy} if options.dd_strategy else {}

    @staticmethod
    def _circuits_have_pulse_ops(
        circuits: Iterable[QuantumCircuit],
    ) -> bool:
        """Check whether any circuit in the batch contains operations that need to be compiled locally.

        Args:
            circuits: The circuit batch to check.


        Returns:
            ``True`` iff at least one :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation`
            is found in the circuit batch.

        """
        for qc in circuits:
            for instr in qc.data:
                if isinstance(instr.op, IQMPulseOperation):
                    return True
        return False

    @classmethod
    def _default_options(cls) -> Mapping[str, Any]:
        """Default runtime options for the IQM backend."""
        return {"shots": _DEFAULT_SHOTS}

    # ------------------------------------------------------------------
    # Execution entry points
    # ------------------------------------------------------------------

    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
        *,
        use_timeslot: bool | None = None,
        compilation_options: CircuitCompilationOptions | None = None,
    ) -> MeasurementResult | list[MeasurementResult]:
        """Transpile and submit circuits for execution, return measurement result(s).

        Blocks until the execution is finished, which may take awhile if there are other jobs in the
        queue.  Consider using :meth:`run_async` instead.

        Takes the same args as :meth:`run_async`.

        Returns:
            A pre-populated :class:`~qrisp.interface.MeasurementResult` when
            one circuit is submitted, or a list of them for multiple circuits.

        """
        self._validate_shots(shots)
        self._check_circuit_limit(circuits)
        batch = not isinstance(circuits, QuantumCircuit)
        all_counts = (
            self.run_async(
                circuits,
                shots,
                use_timeslot=use_timeslot,
                compilation_options=compilation_options,
            )
            .result()
            .all_counts
        )

        results = []
        for counts in all_counts:
            raw = MeasurementResult()
            raw._inject(counts)
            results.append(raw)

        return results if batch else results[0]

    def run_async(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        shots: int | None = None,
        *,
        use_timeslot: bool | None = None,
        compilation_options: CircuitCompilationOptions | None = None,
    ) -> IQMJob:
        """Transpile and submit circuits for execution on the IQM device, return immediately.

        Automatically detects whether any of ``circuits`` contain
        :class:`~iqm.qrisp_iqm.pulse_operation.IQMPulseOperation` instances:

        - **No pulse ops** → submitted as a gate-level :class:`~.iqm_circuit_job.IQMCircuitJob`.
        - **Has pulse ops** → compiled locally down to pulse level, and submitted as a
          :class:`~.iqm_pulse_job.IQMPulseJob`.

        Args:
            circuits: The circuit(s) to execute.
            shots: Number of shots per circuit. If ``None``, use the backend default.
            use_timeslot: If True, submits the job to the timeslot queue.
                If False, the job is submitted to the normal on-demand queue.
                ``None`` means use the default set in :meth:`__init__`.
            compilation_options: Options for compiling circuits to pulse schedules.
                On the pulse-level execution path, the :attr:`CircuitCompilationOptions.max_circuit_duration_over_t2`
                attribute has no effect.
                ``None`` means use the default set in :meth:`__init__`.

        Returns:
            A job handle for the submitted execution.

        """
        # Normalize to sequence
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        self._validate_shots(shots)
        self._check_circuit_limit(circuits)
        n_shots = shots if shots is not None else self._options.get("shots", _DEFAULT_SHOTS)
        use_timeslot = self.use_timeslot if use_timeslot is None else use_timeslot
        compilation_options = self.compilation_options if compilation_options is None else compilation_options

        logging.getLogger("iqm").setLevel(logging.WARNING)
        self._check_default_calset()

        # transpile to IQM format
        pm = self.pm
        circuit_batch = []
        for qc in circuits:
            transpiled_qc = pm.run(qc)
            iqm_qc = qrisp_to_iqm_converter(transpiled_qc, self._dqa)
            circuit_batch.append(iqm_qc)

        if self._circuits_have_pulse_ops(circuits):
            return self._run_async_pulse(
                circuit_batch, n_shots, use_timeslot=use_timeslot, compilation_options=compilation_options
            )
        return self._run_async_client(
            circuit_batch, n_shots, use_timeslot=use_timeslot, compilation_options=compilation_options
        )

    def _run_async_client(
        self,
        circuit_batch: Sequence[Circuit],
        shots: int,
        *,
        use_timeslot: bool,
        compilation_options: CircuitCompilationOptions | None,
    ) -> IQMCircuitJob:
        """Transpile, convert, and submit circuits via IQMClient.

        See :meth:`run_async`.
        """
        job = self.iqm_client.submit_circuits(
            list(circuit_batch),
            shots=shots,
            options=compilation_options,
            calibration_set_id=self._calibration_set_id,
            use_timeslot=use_timeslot,
        )
        return IQMCircuitJob(job, backend=self)

    def _run_async_pulse(
        self,
        circuit_batch: Sequence[Circuit],
        shots: int,
        *,
        use_timeslot: bool,
        compilation_options: CircuitCompilationOptions | None,
    ) -> IQMPulseJob:
        """Transpile, convert, compile, and submit circuits via Pulla.

        See :meth:`run_async`.
        """
        # Resolve compiler, forwarding calibration_set_id via loading_rules.
        compiler = self._compiler
        if compiler is None:
            from iqm.cpc.core.observation.observation_loading_rules import LatestFromStash  # noqa: PLC0415

            stash = self._pulla.get_calibration_stash(self._calibration_set_id)
            loading_rules: list[RuleType] = [LatestFromStash(stash)]
            compiler = self._pulla.get_standard_compiler(loading_rules=loading_rules)

        # Build settings, inject compilation options and shot count, then compile.
        settings = compiler.get_settings(circuits=circuit_batch)
        settings.set_shots(shots)
        context = self._apply_compilation_options_to_settings(settings, compilation_options)
        run_definition, context = compiler.compile(circuit_batch, settings=settings, context=context)

        job = self._pulla.submit_playlist(
            run_definition,
            context=context,
            use_timeslot=use_timeslot,
        )
        return IQMPulseJob(job, backend=self)

    # ------------------------------------------------------------------
    # Transpilation metadata
    # ------------------------------------------------------------------

    @property
    def connectivity(self) -> list[tuple[int, int]]:
        """Currently executable qubit connectivity for this IQM backend.

        Cached once at construction time from the DynamicQuantumArchitecture.

        Returns:
            The two-qubit gate coupling map as a list of qubit-index pairs.

        """
        return self._connectivity

    # ------------------------------------------------------------------
    # Job retrieval
    # ------------------------------------------------------------------

    @overload
    def retrieve_job(self, job_id: str, *, pulse: Literal[False]) -> IQMCircuitJob: ...

    @overload
    def retrieve_job(self, job_id: str, *, pulse: Literal[True]) -> IQMPulseJob: ...

    def retrieve_job(self, job_id: str, *, pulse: bool = False) -> IQMJob:
        """Retrieve a previously submitted job.

        Args:
            job_id: The IQM job ID (UUID as string) of the job to retrieve.
            pulse: Whether the job was submitted at the pulse level.

        Returns:
            Handle for the previously submitted job.

        """
        if pulse:
            from iqm.pulla.pulla import PullaJob  # noqa: PLC0415

            job_data = self._pulla.get_job(UUID(job_id))
            pulla_job = PullaJob(
                data=job_data,
                _pulla=self._pulla,
                _context={},
            )
            return IQMPulseJob(pulla_job, backend=self)

        job = self.iqm_client.get_job(UUID(job_id))
        return IQMCircuitJob(job, backend=self)
