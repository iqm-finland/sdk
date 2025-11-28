# Copyright 2022 Qiskit on IQM developers
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
"""Qiskit backend provider for IQM backends."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import UUID
import warnings

from iqm.iqm_client import CircuitCompilationOptions, CircuitValidationError, IQMClient
from iqm.iqm_client.util import to_json_dict
from iqm.qiskit_iqm import IQMFakeAphrodite, IQMFakeApollo, IQMFakeBackend, IQMFakeDeneb
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus, JobV1, Options

from iqm.pulse import Circuit
from iqm.station_control.interface.models import CircuitBatch, RunRequest

try:
    __version__ = version("qiskit-iqm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


class IQMBackend(IQMBackendBase):
    """Backend for executing quantum circuits on IQM quantum computers.

    Args:
        client: Client instance for communicating with an IQM Server.
        calibration_set_id: ID of the calibration set the backend will use.
            ``None`` means the IQM Server will be queried for the current default
            calibration set.
        use_metrics: If True, the backend will query the server for calibration data and related
            quality metrics, and pass these to the transpilation target(s). The default value is set
            to False until quality metrics become available on the Resonance API.
        kwargs: Optional arguments to be passed to the parent Backend initializer.

    """

    def __init__(
        self,
        client: IQMClient,
        *,
        calibration_set_id: str | UUID | None = None,
        use_metrics: bool = False,
        **kwargs,
    ):
        if calibration_set_id is not None and not isinstance(calibration_set_id, UUID):
            calibration_set_id = UUID(calibration_set_id)

        self._use_default_calibration_set = calibration_set_id is None
        architecture = client.get_dynamic_quantum_architecture(calibration_set_id)
        metrics = client._get_calibration_quality_metrics(architecture.calibration_set_id) if use_metrics else None
        super().__init__(architecture, metrics=metrics, **kwargs)
        self.client: IQMClient = client
        self._max_circuits: int | None = None
        self._calibration_set_id = architecture.calibration_set_id

    @classmethod
    def _default_options(cls) -> Options:
        """Qiskit method for defining the default options for running the backend. We don't use them since they would
        not be documented here. Instead, we use the keyword arguments of the run method to pass options.
        """
        return Options()

    @property
    def max_circuits(self) -> int | None:
        """Maximum number of circuits that should be run in a single batch.

        Currently there is no hard limit on the number of circuits that can be executed in a single batch/job.
        However, some libraries like Qiskit Experiments use this property to split multi-circuit computational
        tasks into multiple baches/jobs.

        The default value is ``None``, meaning there is no limit. You can set it to a specific integer
        value to force these libraries to run at most that many circuits in a single batch.
        """
        return self._max_circuits

    @max_circuits.setter
    def max_circuits(self, value: int | None) -> None:
        self._max_circuits = value

    def run(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        *,
        use_timeslot: bool = False,
        **options,
    ) -> IQMJob:
        """Run a quantum circuit or a list of quantum circuits on the IQM quantum computer represented by this backend.

        Args:
            run_input: The circuits to run.
            use_timeslot: Submits the job to the timeslot queue if set to ``True``. If set to ``False``,
                the job is submitted to the normal on-demand queue.
            options: Keyword arguments passed on to :meth:`create_run_request`, and documented there.

        Returns:
            Job object from which the results can be obtained once the execution has finished.

        """
        run_request = self.create_run_request(run_input, **options)
        circuit_job = self.client.submit_run_request(run_request, use_timeslot=use_timeslot)
        job = IQMJob(self, circuit_job)
        job.circuit_metadata = [c.metadata if isinstance(c, Circuit) else {} for c in run_request.circuits]
        return job

    def create_run_request(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        shots: int = 1024,
        circuit_compilation_options: CircuitCompilationOptions | None = None,
        circuit_callback: Callable[[list[QuantumCircuit]], Any] | None = None,
        qubit_index_to_name: dict[int, str] | None = None,
        **unknown_options,
    ) -> RunRequest:
        """Creates a run request without submitting it for execution.

        This can be used to check what would be submitted for execution by an equivalent call to :meth:`run`.

        Args:
            run_input: Same as in :meth:`run`.

        Args:
            shots: Number of repetitions of each circuit, for sampling.
            circuit_compilation_options:
                Compilation options for the circuits, passed on to :class:`~iqm.iqm_client.iqm_client.IQMClient`.
                If ``None``, the defaults of the :class:`~iqm.iqm_client.models.CircuitCompilationOptions`
                class are used.
            circuit_callback:
                Callback function that, if provided, will be called for the circuits before sending
                them to the device.  This may be useful in situations when you do not have explicit
                control over transpilation, but need some information on how it was done. This can
                happen, for example, when you use pre-implemented algorithms and experiments in
                Qiskit, where the implementation of the said algorithm or experiment takes care of
                delivering correctly transpiled circuits to the backend. This callback method gives
                you a chance to look into those transpiled circuits, and extract any info you need.
                As a side effect, you can also use this callback to modify the transpiled circuits
                in-place, just before execution; however, we do not recommend to use it for this
                purpose.
            qubit_index_to_name: Mapping from qubit indices in the circuit to qubit names on the device.
                If ``None``, :attr:`.IQMBackendBase.index_to_qubit_name` will be used.

        Returns:
            The created run request object

        """
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        if len(circuits) == 0:
            raise ValueError("Empty list of circuits submitted for execution.")

        # Catch old iqm-client options
        if "max_circuit_duration_over_t2" in unknown_options or "heralding_mode" in unknown_options:
            warnings.warn(
                DeprecationWarning(
                    "max_circuit_duration_over_t2 and heralding_mode are deprecated, please use "
                    + "circuit_compilation_options instead."
                )
            )
        if circuit_compilation_options is None:
            cc_options_kwargs = {}
            if "max_circuit_duration_over_t2" in unknown_options:
                cc_options_kwargs["max_circuit_duration_over_t2"] = unknown_options.pop("max_circuit_duration_over_t2")
            if "heralding_mode" in unknown_options:
                cc_options_kwargs["heralding_mode"] = unknown_options.pop("heralding_mode")
            circuit_compilation_options = CircuitCompilationOptions(**cc_options_kwargs)

        if unknown_options:
            warnings.warn(f"Unknown backend option(s): {unknown_options}")

        if circuit_callback:
            circuit_callback(circuits)

        circuits_serialized: CircuitBatch = [
            self.serialize_circuit(circuit, qubit_index_to_name) for circuit in circuits
        ]

        if self._use_default_calibration_set:
            default_calset_id = self.client.get_dynamic_quantum_architecture(None).calibration_set_id
            if self._calibration_set_id != default_calset_id:
                warnings.warn(
                    f"Server default calibration set has changed from {self._calibration_set_id} "
                    f"to {default_calset_id}. Create a new IQMBackend if you wish to transpile the "
                    "circuits using the new calibration set."
                )
        try:
            run_request = self.client.create_run_request(
                circuits_serialized,
                calibration_set_id=self._calibration_set_id,
                shots=shots,
                options=circuit_compilation_options,
            )
        except CircuitValidationError as e:
            raise CircuitValidationError(
                f"{e}\nMake sure the circuits have been transpiled using the same backend that you used to submit "
                f"the circuits."
            ) from e

        return run_request

    def retrieve_job(self, job_id: str) -> IQMJob:
        """Create and return an IQMJob instance associated with this backend with given job id.

        Args:
            job_id: ID of the job to retrieve.

        Returns:
            corresponding job

        """
        circuit_job = self.client.get_job(UUID(job_id))
        return IQMJob(self, circuit_job)

    def serialize_circuit(self, circuit: QuantumCircuit, qubit_index_to_name: dict[int, str] | None = None) -> Circuit:
        """Serialize a quantum circuit into the IQM data transfer format.

        Serializing is not strictly bound to the native gateset, i.e. some gates that are not explicitly mentioned in
        the native gateset of the backend can still be serialized. For example, the native single qubit gate for IQM
        backend is the 'r' gate, however 'x', 'rx', 'y' and 'ry' gates can also be serialized since they are just
        particular cases of the 'r' gate. If the circuit was transpiled against a backend using Qiskit's transpiler
        machinery, these gates are not supposed to be present. However, when constructing circuits manually and
        submitting directly to the backend, it is sometimes more explicit and understandable to use these concrete
        gates rather than 'r'. Serializing them explicitly makes it possible for the backend to accept such circuits.

        Qiskit uses one measurement instruction per qubit (i.e. there is no measurement grouping concept). While
        serializing we do not group any measurements together but rather associate a unique measurement key with each
        measurement instruction, so that the results can later be reconstructed correctly (see :class:`.MeasurementKey`
        documentation for more details).

        Args:
            circuit: quantum circuit to serialize
            qubit_index_to_name: Mapping from qubit indices in the circuit to qubit names on the device.
                If ``None``, :attr:`.IQMBackendBase.index_to_qubit_name` will be used.

        Returns:
            data transfer object representing the circuit

        Raises:
            ValueError: circuit contains an unsupported instruction or is not transpiled in general

        """
        if qubit_index_to_name is None:
            qubit_index_to_name = self._idx_to_qb
        instructions = tuple(serialize_instructions(circuit, qubit_index_to_name=qubit_index_to_name))

        try:
            metadata = to_json_dict(circuit.metadata)
        except ValueError:
            warnings.warn(
                f"Metadata of circuit {circuit.name} was dropped because it could not be serialised to JSON.",
            )
            metadata = None

        return Circuit(name=circuit.name, instructions=instructions, metadata=metadata)


facade_names: dict[str, IQMFakeBackend] = {
    "facade_adonis": IQMFakeAdonis(),
    "facade_apollo": IQMFakeApollo(),
    "facade_aphrodite": IQMFakeAphrodite(),
    "facade_deneb": IQMFakeDeneb(),
    "facade_garnet": IQMFakeGarnet(),
}


class IQMFacadeBackend(IQMBackend):
    """Simulates locally the execution of quantum circuits on a remote mock IQM quantum computer.

    This backend is meant to be used to run circuits on a mock IQM Server that has no real quantum hardware,
    and if the mock execution is successful, simulate the circuits locally using an error model that
    is broadly representative of the mocked QPU. Finally it returns the *simulated results*.

    If you just want to run a local simulation, use :class:`.IQMFakeBackend` directly.

    .. important::

       When using a facade backend, the IQM Server URL of :class:`IQMProvider` should always point to a mock environment
       rather than a real quantum computer, as the execution results from the server will be discarded and replaced by
       a locally simulated result generated by Qiskit Aer. If you use a real quantum computer with a facade backend,
       you will just waste your credits and/or computation time.

    Args:
        client: Client instance for communicating with an IQM Server.
        name: Name of the fake backend (simulator instance) to use. If None, will be determined automatically based
            on the static quantum architecture of the server.
        kwargs: Optional arguments to be passed to the parent class.

    """

    def __init__(self, client: IQMClient, *, name: str | None = None, **kwargs):
        super().__init__(client, **kwargs)
        sqa = self.client.get_static_quantum_architecture()
        if name is None:
            # use a fake backend (local simulator) that matches the server
            for backend in facade_names.values():
                if backend.validate_compatible_architecture(sqa):
                    self._fake_backend = backend
                    return
            raise ValueError("Quantum architecture of the server does not match any known IQMFakeBackend.")
        else:
            if name not in facade_names:
                raise ValueError(f"Unknown facade backend: {name}")
            backend = facade_names[name]
            if not backend.validate_compatible_architecture(sqa):
                raise ValueError("Quantum architecture of the server does not match the requested IQMFakeBackend.")
            self._fake_backend = backend

    def _validate_no_empty_cregs(self, circuit: QuantumCircuit) -> bool:
        """Returns True if given circuit has no empty (unused) classical registers, False otherwise."""
        cregs_utilization = dict.fromkeys(circuit.cregs, 0)
        used_cregs = [circuit.find_bit(i.clbits[0]).registers[0][0] for i in circuit.data if len(i.clbits) > 0]
        for creg in used_cregs:
            cregs_utilization[creg] += 1

        if 0 in cregs_utilization.values():
            return False
        return True

    def run(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        *,
        use_timeslot: bool = False,
        **options,
    ) -> JobV1:
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input
        circuits_validated_cregs: list[bool] = [self._validate_no_empty_cregs(circuit) for circuit in circuits]
        if not all(circuits_validated_cregs):
            raise ValueError(
                "One or more circuits contain unused classical registers. This is not allowed for facade simulation, "
                "see the user guide."
            )

        iqm_backend_job = super().run(run_input, use_timeslot=use_timeslot, **options)
        iqm_backend_job.result()  # get and discard results
        if iqm_backend_job.status() == JobStatus.ERROR:
            raise RuntimeError("Remote execution did not succeed.")
        return self._fake_backend.run(run_input, **options)


class IQMProvider:
    """Provider for IQM backends.

    IQMProvider connects to a quantum computer through an IQM Server.
    If the server requires user authentication, you can provide it either using environment
    variables, or as keyword arguments to IQMProvider. The user authentication kwargs are passed
    through to :class:`~iqm.iqm_client.iqm_client.IQMClient` as is, and are documented there.

    Args:
        url: URL of the IQM Server (e.g. "https://resonance.meetiqm.com/").
        quantum_computer: ID or alias of the quantum computer to connect to, if the IQM Server
            instance controls more than one (e.g. "garnet"). ``None`` means connect to the
            default one.

    """

    def __init__(
        self,
        url: str,
        *,
        quantum_computer: str | None = None,
        **user_auth_args,  # contains keyword args token or tokens_file
    ):
        self.url = url
        self.quantum_computer = quantum_computer
        self.user_auth_args = user_auth_args

    def get_backend(
        self,
        name: str | None = None,
        calibration_set_id: UUID | None = None,
        *,
        use_metrics: bool = False,
    ) -> IQMBackend:
        """IQMBackend instance associated with this provider.

        Args:
            name: Optional name of a facade backend to request, see :class:`IQMFacadeBackend`.
            calibration_set_id: ID of the calibration set to be used with the backend.
                Affects both the transpilation target and the circuit execution.
                If None, the server default calibration set will be used.
            use_metrics: If True, the backend will provide calibration data and related quality metrics
                to the transpilation target to improve the transpilation. The default value is set to False
                until quality metrics become available on the Resonance API.

        Returns:
            Backend instance for connecting to a quantum computer.

        """
        client = IQMClient(self.url, quantum_computer=self.quantum_computer, **self.user_auth_args)

        if name and name.startswith("facade_"):
            return IQMFacadeBackend(client, name=name, calibration_set_id=calibration_set_id, use_metrics=use_metrics)
        return IQMBackend(client, calibration_set_id=calibration_set_id, use_metrics=use_metrics)
