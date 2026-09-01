# Copyright 2022-2026 IQM
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

"""High-level API for readout error characterization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Any, Self

import numpy as np

from iqm.pulla.pulla import Pulla, PullaJob

from .data_collection import CalibrationJobInfo, retrieve_calibration_results, run_calibration_circuits
from .data_processing import (
    CalibrationData,
    ErrorProbabilities,
    compute_error_probabilities,
)

__all__ = ["RECConfiguration", "ReadoutErrorCharacterization"]


@dataclass
class RECConfiguration:
    """Configuration for a readout error characterization (REC) experiment.

    All parameters have sensible defaults so ``RECConfiguration()`` is valid
    and immediately usable.
    """

    num_circuits: int = 50
    """Number of calibration circuits to generate"""

    shots: int = 10_000
    """Total measurement shots distributed across all circuits. Each circuit receives ``shots // num_circuits``
    shots."""

    qubits: list[str] | None = None
    """Qubit names to characterize.

    Non-operational qubits (those without a calibrated ``measure`` gate) are always excluded:
    they cannot be characterized or have readout error mitigation applied.

    * ``None`` (default): qubits are not pinned. When used standalone, this
      characterizes every operational qubit on the QPU; in the
      :class:`~iqm.error_reduction_tools.rem.rem_workflow.REMWorkflow` it lets the
      workflow infer the qubits from the measured qubits of the input circuits.
    * ``list[str]``: characterize exactly these qubit names, minus any that are not operational.
    """

    seed: int | None = None
    """Random seed for reproducible circuit generation. ``None`` produces non-deterministic results."""

    symmetrize: bool = True
    """When ``True``, generates complementary preparation pairs so that each qubit is prepared in |0⟩ and |1⟩ equally
    often."""

    equatorial_randomization: bool = True
    """When ``True``, randomizes the phase angle of every X gate to average out coherent phase errors."""


class ReadoutErrorCharacterization:
    """High-level interface for readout error characterization.

    Wraps the lower-level functions in
    :mod:`iqm.error_reduction_tools.readout_characterization.data_collection` and
    :mod:`iqm.error_reduction_tools.readout_characterization.data_processing` behind a clean,
    lifecycle-oriented API.

    **Typical online usage**::

        config = RECConfiguration(shots=20_000, qubits=["QB1", "QB2", "QB5"])
        rec = ReadoutErrorCharacterization(client)
        rec.submit_job(config)
        rec.retrieve_results()
        probs = rec.get_readout_error_probabilities()

    **Chained usage**::

        probs = (
            ReadoutErrorCharacterization(client)
            .submit_job(config)
            .retrieve_results()
            .get_readout_error_probabilities()
        )

    **Offline usage (reload saved results)**::

        rec = ReadoutErrorCharacterization.load("charact_garnet_20260311.json")
        probs = rec.get_readout_error_probabilities()
    """

    def __init__(self, client: Pulla | None) -> None:
        """Initialise the characterization object.

        No circuits are submitted at initialization time. Internal state is
        populated progressively by :meth:`submit_job`, :meth:`retrieve_results`,
        and :meth:`get_readout_error_probabilities`.

        Args:
            client: Client for connecting to an IQM quantum computer. Pass ``None`` when
                reconstructing an object offline via :meth:`from_dict` or
                :meth:`load`.

        """
        self._client: Pulla | None = client
        self._config: RECConfiguration | None = None
        self._job: PullaJob | None = None
        self._job_info: CalibrationJobInfo | None = None
        self._raw_results: CalibrationData | None = None
        self._error_probabilities: ErrorProbabilities | None = None
        self._timestamp: str | None = None

    def __repr__(self) -> str:
        """Printable representation."""
        parts: list[str] = []
        if self._client is not None:
            parts.append("client=<connected>")
        if self._config is not None:
            parts.append(f"config={self._config!r}")
        if self._job is not None:
            parts.append("job=<submitted>")
        if self._raw_results is not None:
            n = len(self._raw_results["measured_qubits"])
            parts.append(f"results_available(n_qubits={n})")
        if self._error_probabilities is not None:
            parts.append("probabilities_computed")
        if self._timestamp is not None:
            parts.append(f"timestamp={self._timestamp!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def submit_job(self, config: RECConfiguration | None = None) -> Self:
        """Submit calibration circuits to the quantum computer.

        Stores the returned :class:`~iqm.pulla.pulla.PullaJob` and job
        metadata internally.

        Args:
            config: Characterization configuration. Defaults to
                :class:`RECConfiguration` with all default values when ``None``.

        Returns:
            The current instance for method chaining.

        Raises:
            RuntimeError: If no client was provided at construction time.

        """
        if self._client is None:
            raise RuntimeError(
                "No client available for connecting to a quantum computer. Provide one at construction time."
            )

        if config is None:
            config = RECConfiguration()

        self._config = config
        self._job, self._job_info = run_calibration_circuits(
            client=self._client,
            qubits=config.qubits,
            number_of_circuits=config.num_circuits,
            shots=config.shots,
            symmetrize=config.symmetrize,
            seed=config.seed,
            equatorial_randomization=config.equatorial_randomization,
        )
        return self

    def retrieve_results(
        self,
        job: PullaJob | None = None,
        job_info: CalibrationJobInfo | None = None,
    ) -> Self:
        """Retrieve measurement results from the calibration job.

        Wraps :func:`~iqm.error_reduction_tools.readout_characterization.data_collection.retrieve_calibration_results`
        and stores the raw counts internally.

        ``job`` and ``job_info`` are always treated as a matched pair from the same
        calibration run.  Either supply both to use an externally obtained job,
        or omit both to use the job stored internally by :meth:`submit_job`.
        Passing only one of the two is an error.

        Args:
            job: Job to fetch results from.
            job_info: :class:`~iqm.error_reduction_tools.readout_characterization.data_collection.CalibrationJobInfo`
                that accompanies ``job`` (preparation strings, qubit list, etc.).

        Returns:
            The current instance for method chaining.

        Raises:
            ValueError: If exactly one of ``job`` / ``job_info`` is provided.
            RuntimeError: If neither explicit arguments nor internal state are
                available.

        """
        if job is not None and job_info is not None:
            active_job, active_job_info = job, job_info

        elif job is None and job_info is None:  # Fall back to internal state set by ``submit_job()``.
            if self._job is None:
                raise RuntimeError(
                    "No job available. Call ``submit_job()`` first or pass ``job`` and ``job_info`` explicitly."
                )
            if self._job_info is None:
                raise RuntimeError(
                    "No job metadata (``job_info``) available. Call ``submit_job()`` first or pass ``job`` and "
                    "``job_info`` explicitly."
                )
            active_job, active_job_info = self._job, self._job_info

        else:
            raise ValueError(
                "``job`` and ``job_info`` must be provided together — they are a matched "
                "pair from the same calibration run. "
                f"Got job={'<set>' if job else 'None'}, "
                f"job_info={'<set>' if job_info else 'None'}."
            )

        self._raw_results = retrieve_calibration_results(active_job, active_job_info)
        return self

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_readout_error_probabilities(self) -> ErrorProbabilities:
        """Return per-qubit readout error probabilities.

        Computes the probabilities on the first call (lazy evaluation) and
        caches the result internally. Subsequent calls return the cached value
        without recomputation.

        Returns:
            Per-qubit readout error probabilities. See
            :class:`~iqm.error_reduction_tools.readout_characterization.data_processing.ErrorProbabilities`
            for the structure.

        Raises:
            RuntimeError: If :meth:`retrieve_results` has not been called.

        """
        if self._raw_results is None:
            raise RuntimeError("No results available. Call ``retrieve_results()`` first.")
        if self._error_probabilities is None:
            self._error_probabilities = compute_error_probabilities(self._raw_results)
        return self._error_probabilities

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the characterization state to a JSON-compatible dictionary.

        The serialized form contains everything required to reconstruct a
        post-retrieval object offline via :meth:`from_dict`:

        +------------------------+------------------------------------------+
        | Key                    | Contents                                 |
        +========================+==========================================+
        | ``counts_by_prep``     | Raw calibration counts                   |
        +------------------------+------------------------------------------+
        | ``measured_qubits``    | Qubit labels                             |
        +------------------------+------------------------------------------+
        | ``config``             | :class:`RECConfiguration` used           |
        +------------------------+------------------------------------------+
        | ``charact_data``       | Cached assignment matrices (*optional*,  |
        |                        | present only if already computed)        |
        +------------------------+------------------------------------------+
        | ``charact_data_std``   | Matching standard deviations (*optional*,|
        |                        | present only if already computed)        |
        +------------------------+------------------------------------------+
        | ``timestamp``          | ISO-8601 creation timestamp, in UTC      |
        +------------------------+------------------------------------------+

        Returns:
            JSON-serializable dictionary.

        Raises:
            RuntimeError: If :meth:`retrieve_results` has not been called.

        """
        if self._raw_results is None:
            raise RuntimeError("No results to serialize. Call ``retrieve_results()`` before ``to_dict()``.")
        self._timestamp = datetime.now(tz=UTC).isoformat()
        data: dict[str, Any] = {
            "counts_by_prep": self._raw_results["counts_by_prep"],
            "measured_qubits": self._raw_results["measured_qubits"],
            "config": (asdict(self._config) if self._config is not None else None),
            "timestamp": self._timestamp,
        }
        if self._error_probabilities is not None:
            data["charact_data"] = {k: v.tolist() for k, v in self._error_probabilities["charact_data"].items()}
            data["charact_data_std"] = {k: v.tolist() for k, v in self._error_probabilities["charact_data_std"].items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct a :class:`ReadoutErrorCharacterization` from a dictionary.

        The reconstructed object holds raw results and cached error probabilities
        (when available). It has no client or job handle, but all analysis and
        visualization methods work normally.

        Args:
            data: Dictionary produced by :meth:`to_dict` or loaded via :meth:`load`.

        Returns:
            New :class:`ReadoutErrorCharacterization` instance ready for offline
            analysis.

        """
        rec = cls(client=None)
        rec._raw_results = {
            "counts_by_prep": data["counts_by_prep"],
            "measured_qubits": data["measured_qubits"],
        }
        rec._timestamp = data.get("timestamp")
        if data.get("config") is not None:
            rec._config = RECConfiguration(**data["config"])
        if "charact_data" in data:
            rec._error_probabilities = {
                "charact_data": {k: np.array(v) for k, v in data["charact_data"].items()},
                "charact_data_std": {k: np.array(v) for k, v in data.get("charact_data_std", {}).items()},
            }
        return rec

    def save(self, path: str) -> None:
        """Serialize the characterization state to a JSON file.

        Args:
            path: Destination file path (file will be overwritten if it exists).

        Raises:
            RuntimeError: If :meth:`retrieve_results` has not been called yet.

        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> Self:
        """Reconstruct a :class:`ReadoutErrorCharacterization` from a JSON file.

        Args:
            path: Path to a file created by :meth:`save`.

        Returns:
            New :class:`ReadoutErrorCharacterization` instance ready for offline
            analysis.

        """
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_error_probabilities(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Plot per-qubit readout error probabilities.

        Thin convenience wrapper around
        :func:`~iqm.error_reduction_tools.readout_characterization.visualization.plot_error_probabilities`.
        All keyword arguments are forwarded verbatim (e.g. ``title``,
        ``show_plot``).

        The visualization module is imported lazily to avoid pulling in
        ``matplotlib`` at package import time.

        Raises:
            RuntimeError: If :meth:`retrieve_results` has not been called yet.

        """
        probs = self.get_readout_error_probabilities()  # raises early if no results
        from .visualization import plot_error_probabilities  # noqa: PLC0415

        plot_error_probabilities(probs, **kwargs)
