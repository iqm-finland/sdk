# Copyright 2021-2024 IQM client developers
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
# pylint: disable=too-many-lines
r"""
Client for connecting to the IQM quantum computer server interface.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from importlib.metadata import version
import itertools
import json
import os
import platform
import time
from typing import Any, Callable, Optional
from uuid import UUID
import warnings

from packaging.version import parse
import requests
from requests import HTTPError

from iqm.iqm_client.api import APIConfig, APIEndpoint, APIVariant
from iqm.iqm_client.authentication import TokenManager
from iqm.iqm_client.errors import (
    APITimeoutError,
    ArchitectureRetrievalError,
    CircuitExecutionError,
    CircuitValidationError,
    ClientAuthenticationError,
    ClientConfigurationError,
    JobAbortionError,
)
from iqm.iqm_client.models import (
    _SUPPORTED_OPERATIONS,
    Circuit,
    CircuitBatch,
    CircuitCompilationOptions,
    ClientLibrary,
    DynamicQuantumArchitecture,
    Instruction,
    MoveGateValidationMode,
    QuantumArchitecture,
    QuantumArchitectureSpecification,
    RunCounts,
    RunRequest,
    RunResult,
    RunStatus,
    Status,
    serialize_qubit_mapping,
    validate_circuit,
)

REQUESTS_TIMEOUT = float(os.environ.get('IQM_CLIENT_REQUESTS_TIMEOUT', 60.0))
DEFAULT_TIMEOUT_SECONDS = 900
SECONDS_BETWEEN_CALLS = float(os.environ.get('IQM_CLIENT_SECONDS_BETWEEN_CALLS', 1.0))


class IQMClient:
    """Provides access to IQM quantum computers.

    Args:
        url: Endpoint for accessing the server. Has to start with http or https.
        client_signature: String that IQMClient adds to User-Agent header of requests
            it sends to the server. The signature is appended to IQMClients own version
            information and is intended to carry additional version information,
            for example the version information of the caller.
        token: Long-lived authentication token in plain text format. Used by IQM Resonance.
            If ``token`` is given no other user authentication parameters should be given.
        tokens_file: Path to a tokens file used for authentication.
            If ``tokens_file`` is given no other user authentication parameters should be given.
        auth_server_url: Base URL of the authentication server.
            If ``auth_server_url`` is given also ``username`` and ``password`` must be given.
        username: Username to log in to authentication server.
        password: Password to log in to authentication server.
        api_variant: API variant to use. Default is ``APIVariant.V1``.
            Configurable also by environment variable ``IQM_CLIENT_API_VARIANT``.

    Alternatively, the user authentication related keyword arguments can also be given in
    environment variables :envvar:`IQM_TOKEN`, :envvar:`IQM_TOKENS_FILE`, :envvar:`IQM_AUTH_SERVER`,
    :envvar:`IQM_AUTH_USERNAME` and :envvar:`IQM_AUTH_PASSWORD`. All parameters must be given either
    as keyword arguments or as environment variables. Same combination restrictions apply
    for values given as environment variables as for keyword arguments.
    """

    def __init__(
        self,
        url: str,
        *,
        client_signature: Optional[str] = None,
        token: Optional[str] = None,
        tokens_file: Optional[str] = None,
        auth_server_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_variant: Optional[APIVariant] = None,
    ):
        if not url.startswith(('http:', 'https:')):
            raise ClientConfigurationError(f'The URL schema has to be http or https. Incorrect schema in URL: {url}')
        self._token_manager: Optional[TokenManager] = TokenManager(
            token,
            tokens_file,
            auth_server_url,
            username,
            password,
        )
        version_string = 'iqm-client'
        self._signature = f'{platform.platform(terse=True)}'
        self._signature += f', python {platform.python_version()}'
        self._signature += f', iqm-client {version(version_string)}'
        if client_signature:
            self._signature += f', {client_signature}'
        self._architecture: QuantumArchitectureSpecification | None = None
        self._dynamic_architectures: dict[UUID, DynamicQuantumArchitecture] = {}

        if api_variant is None:
            env_var = os.environ.get('IQM_CLIENT_API_VARIANT')
            api_variant = APIVariant(env_var) if env_var else APIVariant.V1
        self._api = APIConfig(api_variant, url)
        if (version_incompatibility_msg := self._check_versions()) is not None:
            warnings.warn(version_incompatibility_msg)

    def __del__(self):
        try:
            # try our best to close the auth session, doesn't matter if it fails,
            self.close_auth_session()
        except Exception:  # pylint: disable=broad-except
            pass

    def _retry_request_on_error(self, request: Callable[[], requests.Response]) -> requests.Response:
        """This is a temporary workaround for 502 errors.
        The current implementation of the server side can run out of network connections
        and silently drop incoming connections making IQM Client to fail with 502 errors."""

        while True:
            result = request()
            if result.status_code == 502:
                time.sleep(SECONDS_BETWEEN_CALLS)
                continue
            break

        return result

    # pylint: disable=too-many-locals
    def submit_circuits(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: Optional[dict[str, str]] = None,
        custom_settings: Optional[dict[str, Any]] = None,
        calibration_set_id: Optional[UUID] = None,
        shots: int = 1,
        options: Optional[CircuitCompilationOptions] = None,
    ) -> UUID:
        """Submits a batch of quantum circuits for execution on a quantum computer.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            custom_settings: Custom settings to override default settings and calibration data.
                Note: This field should always be ``None`` in normal use.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the latest one
            shots: Number of times ``circuits`` are executed. Must be greater than zero
            options: Various discrete options for compiling quantum circuits to instruction schedules.
        Returns:
            ID for the created job. This ID is needed to query the job status and the execution results.
        """
        run_request = self.create_run_request(
            circuits=circuits,
            qubit_mapping=qubit_mapping,
            custom_settings=custom_settings,
            calibration_set_id=calibration_set_id,
            shots=shots,
            options=options,
        )
        job_id = self.submit_run_request(run_request)
        return job_id

    def create_run_request(
        self,
        circuits: CircuitBatch,
        *,
        qubit_mapping: Optional[dict[str, str]] = None,
        custom_settings: Optional[dict[str, Any]] = None,
        calibration_set_id: Optional[UUID] = None,
        shots: int = 1,
        options: Optional[CircuitCompilationOptions] = None,
    ) -> RunRequest:
        """Creates a run request for executing circuits without sending it to the server.

        This is called inside :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Can be used to inspect the run request that would be submitted by :meth:`submit_circuits`, without actually
        submitting it for execution.

        Args:
            circuits: Circuits to be executed.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if all ``circuits`` already use physical qubit names.
                Note that the ``qubit_mapping`` is used for all ``circuits``.
            custom_settings: Custom settings to override default settings and calibration data.
                Note: This field should always be ``None`` in normal use.
            calibration_set_id: ID of the calibration set to use, or ``None`` to use the latest one
            shots: Number of times ``circuits`` are executed. Must be greater than zero.
            options: Various discrete options for compiling quantum circuits to instruction schedules.
        Returns:
            RunRequest that would be submitted by equivalent call to :meth:`submit_circuits`.
        """
        if shots < 1:
            raise ValueError('Number of shots must be greater than zero.')
        if options is None:
            options = CircuitCompilationOptions()

        for i, circuit in enumerate(circuits):
            try:
                # validate the circuit against the static information in iqm.iqm_client.models._SUPPORTED_OPERATIONS
                validate_circuit(circuit)
            except ValueError as e:
                raise CircuitValidationError(f'The circuit at index {i} failed the validation').with_traceback(
                    e.__traceback__
                )

        architecture = self.get_dynamic_quantum_architecture(calibration_set_id)

        self._validate_qubit_mapping(architecture, circuits, qubit_mapping)
        serialized_qubit_mapping = serialize_qubit_mapping(qubit_mapping) if qubit_mapping else None

        # validate the circuit against the calibration-dependent dynamic quantum architecture
        self._validate_circuit_instructions(
            architecture,
            circuits,
            qubit_mapping,
            validate_moves=options.move_gate_validation,
            must_close_sandwiches=False,
        )
        return RunRequest(
            qubit_mapping=serialized_qubit_mapping,
            circuits=circuits,
            custom_settings=custom_settings,
            calibration_set_id=calibration_set_id,
            shots=shots,
            max_circuit_duration_over_t2=options.max_circuit_duration_over_t2,
            heralding_mode=options.heralding_mode,
            move_validation_mode=options.move_gate_validation,
            move_gate_frame_tracking_mode=options.move_gate_frame_tracking,
            active_reset_cycles=options.active_reset_cycles,
            dd_mode=options.dd_mode,
            dd_strategy=options.dd_strategy,
        )

    def submit_run_request(self, run_request: RunRequest) -> UUID:
        """Submits a run request for execution on a quantum computer.

        This is called inside :meth:`submit_circuits` and does not need to be called separately in normal usage.

        Args:
            run_request: the run request to be submitted for execution.
        Returns:
            ID for the created job. This ID is needed to query the job status and the execution results.
        """
        headers = {'Expect': '100-Continue', **self._default_headers()}
        try:
            # check if someone is trying to profile us with OpenTelemetry
            # pylint: disable=import-outside-toplevel
            # pylint: disable=import-error
            from opentelemetry import propagate

            propagate.inject(headers)
        except ImportError as _:
            # no OpenTelemetry, no problem
            pass

        if os.environ.get('IQM_CLIENT_DEBUG') == '1':
            print(f'\nIQM CLIENT DEBUGGING ENABLED\nSUBMITTING RUN REQUEST:\n{run_request}\n')

        result = self._retry_request_on_error(
            lambda: requests.post(
                self._api.url(APIEndpoint.SUBMIT_JOB),
                json=json.loads(run_request.model_dump_json(exclude_none=True)),
                headers=headers,
                timeout=REQUESTS_TIMEOUT,
            )
        )

        self._check_not_found_error(result)

        if result.status_code == 401:
            raise ClientAuthenticationError(f'Authentication failed: {result.text}')

        if 400 <= result.status_code < 500:
            raise ClientConfigurationError(f'Client configuration error: {result.text}')

        result.raise_for_status()

        try:
            job_id = UUID(result.json()['id'])
            return job_id
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise CircuitExecutionError(f'Invalid response: {result.text}, {e}') from e

    @staticmethod
    def _validate_qubit_mapping(
        architecture: DynamicQuantumArchitecture,
        circuits: CircuitBatch,
        qubit_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Validates the given qubit mapping, if defined.

        Args:
          architecture: Quantum architecture to check against.
          circuits: Circuits to be checked.
          qubit_mapping: Mapping of logical qubit names to physical qubit names.
              Can be set to ``None`` if all ``circuits`` already use physical qubit names.
              Note that the ``qubit_mapping`` is used for all ``circuits``.

        Raises:
            CircuitValidationError: There was something wrong with ``circuits``.
        """
        if qubit_mapping is None:
            return

        # check if qubit mapping is injective
        target_qubits = set(qubit_mapping.values())
        if not len(target_qubits) == len(qubit_mapping):
            raise CircuitValidationError('Multiple logical qubits map to the same physical qubit.')

        # check if qubit mapping covers all qubits in the circuits
        for i, circuit in enumerate(circuits):
            diff = circuit.all_qubits() - set(qubit_mapping)
            if diff:
                raise CircuitValidationError(
                    f"The qubits {diff} in circuit '{circuit.name}' at index {i} "
                    f'are not found in the provided qubit mapping.'
                )

        # check that each mapped qubit is defined in the quantum architecture
        for _logical, physical in qubit_mapping.items():
            if physical not in architecture.components:
                raise CircuitValidationError(f'Component {physical} not present in dynamic quantum architecture')

    @staticmethod
    def _validate_circuit_instructions(
        architecture: DynamicQuantumArchitecture,
        circuits: CircuitBatch,
        qubit_mapping: Optional[dict[str, str]] = None,
        validate_moves: MoveGateValidationMode = MoveGateValidationMode.STRICT,
        *,
        must_close_sandwiches: bool = True,
    ) -> None:
        """Validate the given circuits against the given quantum architecture.

        Args:
          architecture: quantum architecture to check against
          circuits: circuits to be checked
          qubit_mapping: Mapping of logical qubit names to physical qubit names.
              Can be set to ``None`` if all ``circuits`` already use physical qubit names.
              Note that the ``qubit_mapping`` is used for all ``circuits``.
          validate_moves: determines how MOVE gate validation works
          must_close_sandwiches: Iff True, MOVE sandwiches cannot be left open when the circuit ends.

        Raises:
            CircuitValidationError: validation failed
        """
        for index, circuit in enumerate(circuits):
            measurement_keys: set[str] = set()
            for instr in circuit.instructions:
                IQMClient._validate_instruction(architecture, instr, qubit_mapping)
                # check measurement key uniqueness
                if instr.name in {'measure', 'measurement'}:
                    key = instr.args['key']
                    if key in measurement_keys:
                        raise CircuitValidationError(f'Circuit {index}: {instr!r} has a non-unique measurement key.')
                    measurement_keys.add(key)
            IQMClient._validate_circuit_moves(
                architecture,
                circuit,
                qubit_mapping,
                validate_moves=validate_moves,
                must_close_sandwiches=must_close_sandwiches,
            )

    @staticmethod
    def _validate_instruction(
        architecture: DynamicQuantumArchitecture,
        instruction: Instruction,
        qubit_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Validate an instruction against the dynamic quantum quantum architecture.

        Checks that the instruction uses a valid implementation, and targets a valid locus.

        Args:
          architecture: quantum architecture to check against
          instruction: instruction to check
          qubit_mapping: Mapping of logical qubit names to physical qubit names.
              Can be set to ``None`` if ``instruction`` already uses physical qubit names.

        Raises:
            CircuitValidationError: validation failed
        """
        op_info = _SUPPORTED_OPERATIONS.get(instruction.name)
        if op_info is None:
            raise CircuitValidationError(f"Unknown quantum operation '{instruction.name}'.")

        # apply the qubit mapping if any
        mapped_qubits = tuple(qubit_mapping[q] for q in instruction.qubits) if qubit_mapping else instruction.qubits

        def check_locus_components(allowed_components: Iterable[str], msg: str) -> None:
            """Checks that the instruction locus consists of the allowed components only."""
            for q, mapped_q in zip(instruction.qubits, mapped_qubits):
                if mapped_q not in allowed_components:
                    raise CircuitValidationError(
                        f'{instruction!r}: Component {q} = {mapped_q} {msg}.'
                        if qubit_mapping
                        else f'{instruction!r}: Component {q} {msg}.'
                    )

        if op_info.no_calibration_needed:
            # all QPU loci are allowed
            check_locus_components(architecture.components, msg='does not exist on the QPU')
            return

        gate_info = architecture.gates.get(instruction.name)
        if gate_info is None:
            raise CircuitValidationError(
                f"Operation '{instruction.name}' is not supported by the dynamic quantum architecture."
            )

        if instruction.implementation is not None:
            # specific implementation requested
            impl_info = gate_info.implementations.get(instruction.implementation)
            if impl_info is None:
                raise CircuitValidationError(
                    f"Operation '{instruction.name}' implementation '{instruction.implementation}' "
                    f'is not supported by the dynamic quantum architecture.'
                )
            allowed_loci = impl_info.loci
            instruction_name = f'{instruction.name}.{instruction.implementation}'
        else:
            # any implementation is fine
            allowed_loci = gate_info.loci
            instruction_name = f'{instruction.name}'

        if op_info.factorizable:
            # Check that all the locus components are allowed by the architecture
            check_locus_components(
                set(q for locus in allowed_loci for q in locus), msg=f"is not allowed as locus for '{instruction_name}'"
            )
            return

        # Check that locus matches one of the allowed loci
        all_loci = (
            tuple(tuple(x) for locus in allowed_loci for x in itertools.permutations(locus))
            if op_info.symmetric
            else allowed_loci
        )
        if mapped_qubits not in all_loci:
            raise CircuitValidationError(
                f"{instruction.qubits} = {tuple(mapped_qubits)} is not allowed as locus for '{instruction_name}'"
                if qubit_mapping
                else f"{instruction.qubits} is not allowed as locus for '{instruction_name}'"
            )

    @staticmethod
    def _validate_circuit_moves(
        architecture: DynamicQuantumArchitecture,
        circuit: Circuit,
        qubit_mapping: Optional[dict[str, str]] = None,
        validate_moves: MoveGateValidationMode = MoveGateValidationMode.STRICT,
        *,
        must_close_sandwiches: bool = True,
    ) -> None:
        """Raises an error if the MOVE gates in the circuit are not valid in the given architecture.

        Args:
            architecture: Quantum architecture to check against.
            circuit: Quantum circuit to validate.
            qubit_mapping: Mapping of logical qubit names to physical qubit names.
                Can be set to ``None`` if the ``circuit`` already uses physical qubit names.
            validate_moves: Option for bypassing full or partial MOVE gate validation.
            must_close_sandwiches: Iff True, MOVE sandwiches cannot be left open when the circuit ends.
        Raises:
            CircuitValidationError: validation failed
        """
        # pylint: disable=too-many-branches
        if validate_moves == MoveGateValidationMode.NONE:
            return
        move_gate = 'move'
        # Check if MOVE gates are allowed on this architecture
        if move_gate not in architecture.gates:
            if any(i.name == move_gate for i in circuit.instructions):
                raise CircuitValidationError('MOVE instruction is not supported by the given device architecture.')
            return

        # some gates are allowed in MOVE sandwiches
        allowed_gates = {'barrier'}
        if validate_moves == MoveGateValidationMode.ALLOW_PRX:
            allowed_gates.add('prx')

        all_resonators = set(architecture.computational_resonators)
        all_qubits = set(architecture.qubits)
        if qubit_mapping:
            reverse_mapping = {phys: log for log, phys in qubit_mapping.items()}
            all_resonators = {reverse_mapping[q] if q in reverse_mapping else q for q in all_resonators}
            all_qubits = {reverse_mapping[q] if q in reverse_mapping else q for q in all_qubits}

        # Mapping from resonator to the qubit whose state it holds. Resonators not in the map hold no qubit state.
        resonator_occupations: dict[str, str] = {}
        # Qubits whose states are currently moved to a resonator
        moved_qubits: set[str] = set()

        for inst in circuit.instructions:
            if inst.name == 'move':
                qubit, resonator = inst.qubits
                if not (qubit in all_qubits and resonator in all_resonators):
                    raise CircuitValidationError(
                        f'MOVE instructions are only allowed between qubit and resonator, not {inst.qubits}.'
                    )

                if (resonator_qubit := resonator_occupations.get(resonator)) is None:
                    # Beginning MOVE: check that the qubit hasn't been moved to another resonator
                    if qubit in moved_qubits:
                        raise CircuitValidationError(
                            f'MOVE instruction {inst.qubits}: state of {qubit} is '
                            f'in another resonator: {resonator_occupations}.'
                        )
                    resonator_occupations[resonator] = qubit
                    moved_qubits.add(qubit)
                else:
                    # Ending MOVE: check that the qubit matches to the qubit that was moved to the resonator
                    if resonator_qubit != qubit:
                        raise CircuitValidationError(
                            f'MOVE instruction {inst.qubits} to an already occupied resonator: {resonator_occupations}.'
                        )
                    del resonator_occupations[resonator]
                    moved_qubits.remove(qubit)
            elif moved_qubits:
                # Validate that moved qubits are not used during MOVE operations
                if inst.name not in allowed_gates:
                    if overlap := set(inst.qubits) & moved_qubits:
                        raise CircuitValidationError(
                            f'Instruction {inst.name} acts on {inst.qubits} while the state(s) of {overlap} '
                            f'are in a resonator. Current resonator occupation: {resonator_occupations}.'
                        )

        # Finally validate that all MOVE sandwiches have been ended before the circuit ends
        if must_close_sandwiches and resonator_occupations:
            raise CircuitValidationError(
                f'Circuit ends while qubit state(s) are still in a resonator: {resonator_occupations}.'
            )

    def _get_run_v1(self, job_id: UUID, timeout_secs: float = REQUESTS_TIMEOUT) -> RunResult:
        """
        V1 API (Cocos circuits execution and Resonance) has an inefficient `GET /jobs/<id>` endpoint
        that returns the full job status, including the result and the original request, in a single call.
        """
        result = self._retry_request_on_error(
            lambda: requests.get(
                self._api.url(APIEndpoint.GET_JOB_RESULT, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            )
        )
        self._check_not_found_error(result)
        result.raise_for_status()
        return RunResult.from_dict(result.json())

    def _get_run_v2(self, job_id: UUID, timeout_secs: float = REQUESTS_TIMEOUT) -> RunResult:
        """
        V2 API (Station-based circuits execution) has granular endpoints for job status and result.
        """
        status_response = requests.get(
            self._api.url(APIEndpoint.GET_JOB_STATUS, str(job_id)),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        self._check_not_found_error(status_response)
        status_response.raise_for_status()
        status = status_response.json()
        if Status(status['status']) not in [Status.READY, Status.ABORTED, Status.DELETED, Status.FAILED]:
            return RunResult.from_dict(
                {
                    'measurements': [],
                    'status': status['status'],
                    'message': '',
                    'metadata': {
                        'calibration_set_id': None,
                        'circuits_batch': [],
                        'parameters': None,
                        'timestamps': {},
                    },
                }
            )

        result = self._retry_request_on_error(
            lambda: requests.get(
                self._api.url(APIEndpoint.GET_JOB_RESULT, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            )
        )
        if result.status_code != 404:
            result.raise_for_status()
        measurements = [] if result.status_code == 404 else result.json()
        request_parameters = (
            {}
            if result.status_code == 404
            else requests.get(
                self._api.url(APIEndpoint.GET_JOB_REQUEST_PARAMETERS, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            ).json()
        )
        calibration_set_id = (
            None
            if result.status_code == 404
            else requests.get(
                self._api.url(APIEndpoint.GET_JOB_CALIBRATION_SET_ID, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            ).json()
        )
        circuits_batch = (
            []
            if result.status_code == 404
            else requests.get(
                self._api.url(APIEndpoint.GET_JOB_CIRCUITS_BATCH, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            ).json()
        )
        timeline = (
            []
            if result.status_code == 404
            else requests.get(
                self._api.url(APIEndpoint.GET_JOB_TIMELINE, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            ).json()
        )
        error_message_response = requests.get(
            self._api.url(APIEndpoint.GET_JOB_ERROR_LOG, str(job_id)),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        error_message = error_message_response.text if error_message_response.status_code == 200 else None
        return RunResult.from_dict(
            {
                'measurements': measurements,
                'status': status['status'],
                'message': error_message,
                'metadata': {
                    '`': calibration_set_id,
                    'circuits_batch': circuits_batch,
                    'parameters': (
                        None
                        if result.status_code == 404
                        else {
                            'shots': request_parameters['shots'],
                            'max_circuit_duration_over_t2': request_parameters['max_circuit_duration_over_t2'],
                            'heralding_mode': request_parameters['heralding_mode'],
                            'move_validation_mode': request_parameters['move_validation_mode'],
                            'move_gate_frame_tracking_mode': request_parameters['move_gate_frame_tracking_mode'],
                        }
                    ),
                    'timestamps': {datapoint['stage']: datapoint['timestamp'] for datapoint in timeline},
                },
            }
        )

    def get_run(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunResult:
        """Query the status and results of a submitted job.

        Args:
            job_id: id of the job to query
            timeout_secs: network request timeout

        Returns:
            result of the job (can be pending)

        Raises:
            CircuitExecutionError: IQM server specific exceptions
            HTTPException: HTTP exceptions
        """
        try:
            if self._api.variant == APIVariant.V2:
                run_result = self._get_run_v2(job_id, timeout_secs)
            else:
                run_result = self._get_run_v1(job_id, timeout_secs)
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise CircuitExecutionError(f'Invalid response: {e}') from e

        if run_result.warnings:
            for warning in run_result.warnings:
                warnings.warn(warning)
        if run_result.status == Status.FAILED:
            raise CircuitExecutionError(run_result.message)
        return run_result

    def get_run_status(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunStatus:
        """Query the status of a submitted job.

        Args:
            job_id: id of the job to query
            timeout_secs: network request timeout

        Returns:
            status of the job

        Raises:
            CircuitExecutionError: IQM server specific exceptions
            HTTPException: HTTP exceptions
        """
        result = self._retry_request_on_error(
            lambda: requests.get(
                self._api.url(APIEndpoint.GET_JOB_STATUS, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            )
        )

        self._check_not_found_error(result)
        result.raise_for_status()
        try:
            run_result = RunStatus.from_dict(result.json())
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise CircuitExecutionError(f'Invalid response: {result.text}, {e}') from e

        if run_result.warnings:
            for warning in run_result.warnings:
                warnings.warn(warning)
        return run_result

    def wait_for_compilation(self, job_id: UUID, timeout_secs: float = DEFAULT_TIMEOUT_SECONDS) -> RunResult:
        """Poll results until a job is either pending execution, ready, failed, aborted, or timed out.

        Args:
            job_id: id of the job to wait for
            timeout_secs: how long to wait for a response before raising an APITimeoutError

        Returns:
            job result

        Raises:
            APITimeoutError: time exceeded the set timeout
        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_secs:
            status = self.get_run_status(job_id).status
            if status not in (Status.PENDING_COMPILATION, Status.ACCEPTED, Status.PROCESSING, Status.RECEIVED):
                return self.get_run(job_id)
            time.sleep(SECONDS_BETWEEN_CALLS)
        raise APITimeoutError(f"The job compilation didn't finish in {timeout_secs} seconds.")

    def wait_for_results(self, job_id: UUID, timeout_secs: float = DEFAULT_TIMEOUT_SECONDS) -> RunResult:
        """Poll results until a job is either ready, failed, aborted, or timed out.
           Note, that jobs handling on the server side is async and if we try to request the results
           right after submitting the job (which is usually the case)
           we will find the job is still pending at least for the first query.

        Args:
            job_id: id of the job to wait for
            timeout_secs: how long to wait for a response before raising an APITimeoutError

        Returns:
            job result

        Raises:
            APITimeoutError: time exceeded the set timeout
        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_secs:
            status = self.get_run_status(job_id).status
            if status not in (
                Status.PENDING_COMPILATION,
                Status.PENDING_EXECUTION,
                Status.ACCEPTED,
                Status.PROCESSING,
                Status.RECEIVED,
            ):
                return self.get_run(job_id)
            time.sleep(SECONDS_BETWEEN_CALLS)
        raise APITimeoutError(f"The job didn't finish in {timeout_secs} seconds.")

    def abort_job(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> None:
        """Abort a job that was submitted for execution.

        Args:
            job_id: id of the job to be aborted
            timeout_secs: network request timeout

        Raises:
            JobAbortionError: if aborting the job failed
        """
        result = requests.post(
            self._api.url(APIEndpoint.ABORT_JOB, str(job_id)),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        if result.status_code != 200:
            raise JobAbortionError(result.text)

    def get_quantum_architecture(self, *, timeout_secs: float = REQUESTS_TIMEOUT) -> QuantumArchitectureSpecification:
        """Retrieve quantum architecture from server.
        Caches the result and returns the same result on later invocations.

        Args:
            timeout_secs: network request timeout

        Returns:
            quantum architecture

        Raises:
            ArchitectureRetrievalError: IQM server specific exceptions
            ClientAuthenticationError: if no valid authentication is provided
            HTTPException: HTTP exceptions
        """
        if self._architecture:
            return self._architecture

        result = requests.get(
            self._api.url(APIEndpoint.QUANTUM_ARCHITECTURE),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )

        self._check_not_found_error(result)
        self._check_authentication_errors(result)
        result.raise_for_status()

        try:
            qa = QuantumArchitecture(**result.json()).quantum_architecture
        except (json.decoder.JSONDecodeError, KeyError) as e:
            # show the error class also
            raise ArchitectureRetrievalError(f'Invalid response: {result.text}, {e!r}') from e

        # Cache architecture so that later invocations do not need to query it again
        self._architecture = qa
        return qa

    def get_dynamic_quantum_architecture(
        self, calibration_set_id: Optional[UUID] = None, *, timeout_secs: float = REQUESTS_TIMEOUT
    ) -> DynamicQuantumArchitecture:
        """Retrieve dynamic quantum architecture (DQA) for the given calibration set from server.

        Caches the result and returns the same result on later invocations, unless ``calibration_set_id`` is ``None``.
        If ``calibration_set_id`` is ``None``, always retrieves the result from the server because the "default"
        calibration set may have changed.

        Args:
            calibration_set_id: ID of the calibration set for which the DQA is returned.
                If ``None``, use current default calibration set on the server.
            timeout_secs: network request timeout

        Returns:
            dynamic quantum architecture corresponding to the given calibration set

        Raises:
            ArchitectureRetrievalError: IQM server specific exceptions
            ClientAuthenticationError: if no valid authentication is provided
            HTTPException: HTTP exceptions
        """
        if calibration_set_id is None:
            calibration_set_id_str = 'default'
        elif calibration_set_id in self._dynamic_architectures:
            return self._dynamic_architectures[calibration_set_id]
        else:
            calibration_set_id_str = str(calibration_set_id)

        result = requests.get(
            self._api.url(APIEndpoint.CALIBRATED_GATES, calibration_set_id_str),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )

        self._check_not_found_error(result)
        self._check_authentication_errors(result)
        result.raise_for_status()

        try:
            dqa = DynamicQuantumArchitecture(**result.json())
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise ArchitectureRetrievalError(f'Invalid response: {result.text}, {e}') from e

        # Cache architecture so that later invocations do not need to query it again
        self._dynamic_architectures[dqa.calibration_set_id] = dqa

        return dqa

    def get_feedback_groups(self, *, timeout_secs: float = REQUESTS_TIMEOUT) -> tuple[frozenset[str], ...]:
        """Retrieve groups of qubits that can receive real-time feedback signals from each other.

        Real-time feedback enables conditional gates such as `cc_prx`.
        Some hardware configurations support routing real-time feedback only between certain qubits.

        This method is only supported for the API variant V2.

        Returns:
            Feedback groups. Within a group, any qubit can receive real-time feedback from any other qubit in
                the same group. A qubit can belong to multiple groups.
                If there is only one group, there are no restrictions regarding feedback routing.

        Raises:
            ClientAuthenticationError: if no valid authentication is provided
            HTTPException: HTTP exceptions
        """
        result = requests.get(
            self._api.url(APIEndpoint.CHANNEL_PROPERTIES),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )
        self._check_not_found_error(result)
        self._check_authentication_errors(result)
        result.raise_for_status()
        try:
            channel_properties = result.json()
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise HTTPError(f'Invalid response: {result.text}, {e}') from e

        all_qubits = self.get_quantum_architecture().qubits
        groups: dict[str, set[str]] = {}
        # All qubits that can read from the same source belong to the same group.
        # A qubit may belong to multiple groups.
        for channel_name, properties in channel_properties.items():
            # Relying on naming convention because we don't have proper mapping available:
            qubit = channel_name.split('__')[0]
            if qubit not in all_qubits:
                continue
            for source in properties.get('fast_feedback_sources', ()):
                groups.setdefault(source, set()).add(qubit)
        # Merge identical groups
        unique_groups: set[frozenset[str]] = {frozenset(group) for group in groups.values()}
        # Sort by group size
        return tuple(sorted(unique_groups, key=len, reverse=True))

    def close_auth_session(self) -> bool:
        """Terminate session with authentication server if there was one created.

        Returns:
            True iff session was successfully closed

        Raises:
            ClientAuthenticationError: if logout failed
            ClientAuthenticationError: if asked to close externally managed authentication session
        """
        if self._token_manager is None:
            return False
        return self._token_manager.close()

    def _default_headers(self) -> dict[str, str]:
        headers = {'User-Agent': self._signature}
        if self._token_manager is not None:
            bearer_token = self._token_manager.get_bearer_token()
            if bearer_token:
                headers['Authorization'] = bearer_token
        return headers

    @staticmethod
    def _check_authentication_errors(result: requests.Response) -> None:
        # for not strictly authenticated endpoints,
        # we need to handle 302 redirects to the auth server login page
        if result.history and any(
            response.status_code == 302 for response in result.history
        ):  # pragma: no cover (generators are broken in coverage)
            raise ClientAuthenticationError('Authentication is required.')
        if result.status_code == 401:
            raise ClientAuthenticationError(f'Authentication failed: {result.text}')

    def _check_not_found_error(self, response: requests.Response) -> None:
        """Raises HTTPError with appropriate message if ``response.status_code == 404``."""
        if response.status_code == 404:
            version_message = ''
            if (version_incompatibility_msg := self._check_versions()) is not None:
                version_message = (
                    f' This may be caused by the server version not supporting this endpoint. '
                    f'{version_incompatibility_msg}'
                )
            raise HTTPError(f'{response.url} not found.{version_message}', response=response)

    def _check_versions(self) -> Optional[str]:
        """Checks the client version against compatible client versions reported by server.

        Returns:
            A message about client incompatibility with the server if the versions are incompatible or if the
            compatibility could not be confirmed, None if they are compatible.
        """
        try:
            libraries = self.get_supported_client_libraries()
            compatible_iqm_client = libraries.get(
                'iqm-client',
                libraries.get('iqm_client'),
            )
            if compatible_iqm_client is None:
                return 'Could not verify IQM Client compatibility with the server. You might encounter issues.'
            min_version = parse(compatible_iqm_client.min)
            max_version = parse(compatible_iqm_client.max)
            client_version = parse(version('iqm-client'))
            if client_version < min_version or client_version >= max_version:
                return (
                    f'Your IQM Client version {client_version} was built for a different version of IQM Server. '
                    f'You might encounter issues. For the best experience, consider using a version '
                    f'of IQM Client that satisfies {min_version} <= iqm-client < {max_version}.'
                )
            return None
        except Exception as e:  # pylint: disable=broad-except
            # we don't want the version check to prevent usage of IQMClient in any situation
            check_error = e
        return f'Could not verify IQM Client compatibility with the server. You might encounter issues. {check_error}'

    def get_run_counts(self, job_id: UUID, *, timeout_secs: float = REQUESTS_TIMEOUT) -> RunCounts:
        """Query the counts of an executed job.

        Args:
            job_id: id of the job to query
            timeout_secs: network request timeout

        Returns:
            counts strings dictionary

        Raises:
            CircuitExecutionError: IQM server specific exceptions
            HTTPException: HTTP exceptions
        """
        result = self._retry_request_on_error(
            lambda: requests.get(
                self._api.url(APIEndpoint.GET_JOB_COUNTS, str(job_id)),
                headers=self._default_headers(),
                timeout=timeout_secs,
            )
        )

        self._check_not_found_error(result)
        result.raise_for_status()
        try:
            run_counts = RunCounts.from_dict(result.json())
        except (json.decoder.JSONDecodeError, KeyError) as e:
            raise CircuitExecutionError(f'Invalid response: {result.text}, {e}') from e

        return run_counts

    def get_supported_client_libraries(self, timeout_secs: float = REQUESTS_TIMEOUT) -> dict[str, ClientLibrary]:
        """Retrieves information about supported client libraries from the server.

        Args:
            timeout_secs: network request timeout in seconds

        Returns:
            Dictionary mapping library identifiers to their metadata

        Raises:
            HTTPError: HTTP exceptions
            ClientAuthenticationError: if authentication fails
        """
        result = requests.get(
            self._api.url(APIEndpoint.CLIENT_LIBRARIES),
            headers=self._default_headers(),
            timeout=timeout_secs,
        )

        self._check_not_found_error(result)
        self._check_authentication_errors(result)
        result.raise_for_status()

        return {key: ClientLibrary.model_validate(value) for key, value in result.json().items()}
