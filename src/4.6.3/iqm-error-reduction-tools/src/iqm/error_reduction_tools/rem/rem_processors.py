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

"""IQM's Readout Error Mitigation (REM) module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypedDict, cast
import warnings

from mthree import M3Mitigation
from mthree.classes import QuasiDistribution
import numpy as np

from iqm.pulla.pulla import Pulla

from ..readout_characterization.data_collection import (
    retrieve_calibration_results,
    run_calibration_circuits,
)
from ..readout_characterization.data_processing import AssignmentMatrix, compute_error_probabilities
from ..utils.circuit_utils import TwirledCircuit
from ..utils.general_utils import expectation_from_counts, marginalize_counts

SampledDistribution: TypeAlias = dict[str, int]


class MitigationResult(TypedDict):
    """Return type of :meth:`ReadoutErrorMitigation.mitigate_counts`."""

    mitigated_counts: list[list[dict[str, float]]]
    """``result["mitigated_counts"][i][j]`` is the mitigated quasi-probability distribution
    for circuit *i* under observable *j*."""
    expectation_values: list[list[float]]
    """``result["expectation_values"][i][j]`` is the ZZ\u2026Z expectation value
    for circuit *i* under observable *j*."""
    characterization_performed: bool
    """True if auto-characterization ran during the call."""


def estimate_counts_entropy(counts: SampledDistribution) -> float:
    r"""Estimate the Shannon entropy of the measurement probability distribution.

    Computes :math:`H = -\\sum_i p_i \\log_2(p_i)` from the observed counts
    which quantifies the spread of the distribution.  Higher entropy means
    more unique outcomes with similar probabilities, making mitigation
    more expensive.

    Args:
        counts: Dictionary mapping bitstrings to their measured counts.

    Returns:
        Shannon entropy in bits.  Ranges from 0 (single outcome) to
        :math:`\\log_2(N)` where *N* is the number of unique bitstrings.

    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probabilities = np.array(list(counts.values()), dtype=float) / total
    # Filter out zero-probability entries to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def should_apply_mitigation(
    counts: SampledDistribution,
    max_entropy: float = 28.0,
    force_mitigation: bool = False,
) -> tuple[bool, float]:
    """Determine whether to apply mitigation based on distribution entropy.

    Args:
        counts: Dictionary mapping bitstrings to their measured counts.
        max_entropy: Maximum allowed Shannon entropy (in bits) of the counts
            distribution.  Distributions with entropy above this threshold
            are considered too expensive to mitigate.
        force_mitigation: If True, always apply mitigation regardless of entropy.

    Returns:
        Whether we should mitigate, estimated entropy of ``counts``.

    """
    entropy = estimate_counts_entropy(counts)

    if force_mitigation:
        return True, entropy

    return entropy <= max_entropy, entropy


class ReadoutErrorMitigation:
    """Apply readout error mitigation to measurement counts, with optional twirling support.

    This class performs tensor-product inversion of readout confusion matrices
    to correct measurement errors. Supports both standard and symmetrized
    (twirled) characterization data.
    """

    def __init__(
        self,
        readout_errors: dict[str, AssignmentMatrix] | dict[str, float],
        max_entropy: float = 28.0,
    ):
        """Initialize the readout error mitigator.

        Args:
            readout_errors: Mapping qubit names to their readout error,
                expressed as either a scalar (symmetrized) value or a full 2x2 assignment matrix.
                Can be empty, in which case auto-characterization will be used.
            max_entropy: Maximum allowed Shannon entropy (in bits) of the counts
                distribution before mitigation is skipped.  See
                :func:`estimate_counts_entropy` for details.

        """
        self.charact_data = readout_errors
        self._data_type = self._validate_charact_data() if readout_errors else None
        if readout_errors:
            self._warn_high_error_rates(readout_errors)
        self.max_entropy = max_entropy

    @classmethod
    def from_client(
        cls,
        client: Pulla,
        qubits: list[str] | None = None,
        number_of_circuits: int = 100,
        shots: int = 10000,
        symmetrize: bool = True,
        seed: int | None = None,
        equatorial_randomization: bool = True,
        max_entropy: float = 28.0,
    ) -> ReadoutErrorMitigation:
        """Create ReadoutErrorMitigation by running characterization on a client.

        Runs the full characterization workflow on the quantum computer and then
        returns an initialized ReadoutErrorMitigation instance.

        Args:
            client: Client for connecting to an IQM quantum computer.
            qubits: List of qubit names to characterize (e.g., ["QB1", "QB2"]). If None,
                    characterizes all qubits available on the client.
            number_of_circuits: Number of calibration circuits to generate.
            shots: Total shots for characterization, distributed across all circuits.
            symmetrize: If True, generates complementary I/X preparation pairs.
            seed: Random seed for reproducible characterization.
            equatorial_randomization: If True, randomizes X gate phases.
            max_entropy: Maximum allowed Shannon entropy (in bits) threshold.

        Returns:
            ReadoutErrorMitigation instance with characterization data.

        Example:
            >>> from iqm.pulla import Pulla
            >>> client = Pulla(url="https://example.iqm.fi")
            >>> mitigator = ReadoutErrorMitigation.from_client(
            ...     client=client,
            ...     qubits=["QB1", "QB2"],
            ...     number_of_circuits=30,
            ...     shots=10000,
            ... )
            >>> mitigated = mitigator.mitigate(counts, cl_index_to_qubit_name=["QB1", "QB2"])

        """
        # Run characterization
        job, job_info = run_calibration_circuits(
            client=client,
            qubits=qubits,
            number_of_circuits=number_of_circuits,
            shots=shots,
            symmetrize=symmetrize,
            seed=seed,
            equatorial_randomization=equatorial_randomization,
        )

        # Retrieve and process results
        calibration_raw_data = retrieve_calibration_results(job, job_info)
        charact_result = compute_error_probabilities(calibration_raw_data)

        # Create and return instance
        return cls(readout_errors=charact_result["charact_data"], max_entropy=max_entropy)

    def _validate_charact_data(self) -> str | None:
        """Validate the characterization data format.

        Returns:
            Data type identifier: either "matrix" or "scalar".

        Raises:
            ValueError: If characterization data has invalid format,
                or contains mixed data types.

        """
        if not self.charact_data:
            return None

        data_types = set()
        for qubit_name, data in self.charact_data.items():
            if isinstance(data, np.ndarray) and data.shape == (2, 2):
                data_types.add("matrix")
                if not np.all(np.isclose(np.sum(data, axis=0), 1.0)):
                    raise ValueError(
                        f"Characterization matrix for qubit {qubit_name} must be stochastic (columns sum to 1)."
                    )
            elif isinstance(data, float):
                data_types.add("scalar")
            else:
                raise ValueError(
                    f"Characterization data for qubit {qubit_name} must be either a 2x2 numpy array or a scalar."
                )

        if len(data_types) > 1:
            raise ValueError(
                "All characterization data values must be of the same type (either all 2x2 matrices or all scalars)."
            )

        return data_types.pop()

    @staticmethod
    def _warn_high_error_rates(charact_data: Mapping[str, np.ndarray | float]) -> None:
        """Emit warnings for qubits whose readout error rate is >= 20%.

        For matrix data, checks both P(1|0) (false-1 rate) and P(0|1) (false-0 rate).
        For scalar (symmetrized) data, checks the single error rate.

        Args:
            charact_data: Characterization data dict (qubit_name → matrix or scalar).

        """
        for qubit_name, data in charact_data.items():
            if isinstance(data, np.ndarray) and data.shape == (2, 2):
                p1_given_0 = data[1, 0]  # P(1|0) — false-1 rate
                p0_given_1 = data[0, 1]  # P(0|1) — false-0 rate
                if p1_given_0 >= 0.2 or p0_given_1 >= 0.2:  # noqa: PLR2004
                    warnings.warn(
                        f"High readout error on {qubit_name}: "
                        f"P(1|0)={p1_given_0:.1%}, P(0|1)={p0_given_1:.1%}. "
                        "Mitigation quality may be poor.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                error = float(data)
                if error >= 0.2:  # noqa: PLR2004
                    warnings.warn(
                        f"High readout error on {qubit_name}: error rate={error:.1%}. "
                        f"Mitigation quality may be poor. Consider not using qubit {qubit_name}.",
                        UserWarning,
                        stacklevel=3,
                    )

    def _validate_counts_and_mapping(self, counts: SampledDistribution, cl_index_to_qubit_name: dict[int, str]) -> None:
        """Validate the counts format and qubit mapping.

        Args:
            cl_index_to_qubit_name: A dict mapping classical bit indices to qubit names
            counts: Dictionary mapping bitstrings to their measured counts.

        Raises:
            ValueError: If counts are empty, invalid format, or mapping doesn't
                match the measured qubits.

        """
        if not counts:
            raise ValueError("Counts dictionary cannot be empty.")

        if not isinstance(counts, dict):
            raise ValueError("Counts must be provided as a dictionary.")

        # Determine number of measured qubits from the first bitstring
        first_bitstring = next(iter(counts.keys()))
        num_meas_qubits = len(first_bitstring)

        if not isinstance(cl_index_to_qubit_name, dict):
            raise ValueError("cl_index_to_qubit_mapping must be provided as a dictionary.")

        if set(range(num_meas_qubits)) != set(cl_index_to_qubit_name.keys()):
            raise ValueError(
                f"Counts bitstring length {num_meas_qubits} does not match keys in cl_index_to_qubit_name."
            )

    def _mitigate(
        self,
        counts: SampledDistribution,
        cl_index_to_qubit_name: dict[int, str],
        force_mitigation: bool = False,
        charact_data: dict[str, np.ndarray] | dict[str, float] | None = None,
        data_type: str | None = None,
    ) -> QuasiDistribution | SampledDistribution:
        """Apply readout error mitigation to the given counts.

        Args:
            counts: Dictionary mapping bitstrings to their measured counts.
            cl_index_to_qubit_name: Dict mapping indices to qubit names.
            force_mitigation: If True, apply mitigation regardless of complexity.
            charact_data: Override characterization data for this call. If ``None``,
                ``self.charact_data`` is used. Used internally by ``mitigate_counts``
                when ``twirled=True`` to pass a symmetrized copy.
            data_type: Override data type identifier (``"matrix"`` or ``"scalar"``).
                If ``None``, ``self._data_type`` is used.

        """
        self._validate_counts_and_mapping(counts, cl_index_to_qubit_name)

        effective_data = charact_data if charact_data is not None else self.charact_data
        effective_type = data_type if data_type is not None else self._data_type

        should_mitigate, entropy = should_apply_mitigation(counts, self.max_entropy, force_mitigation)

        if not should_mitigate:
            warnings.warn(
                f"Estimated counts distribution entropy ({entropy:.2f} bits) exceeds "
                f"maximum threshold ({self.max_entropy:.2f} bits). Skipping mitigation. "
                f"Use force_mitigation=True to override."
            )
            return counts

        # Prepare calibration matrices
        if effective_type == "matrix":
            cal_matrices = np.stack(
                [np.array(effective_data[qubit_name]) for qubit_name in cl_index_to_qubit_name.values()]
            )
        else:
            cal_matrices_list = []
            for qubit_name in cl_index_to_qubit_name.values():
                symm_error: float = float(effective_data[qubit_name])
                cal_matrices_list.append(np.array([[1 - symm_error, symm_error], [symm_error, 1 - symm_error]]))
            cal_matrices = np.stack(cal_matrices_list)

        # Perform the actual mitigation (using mthree functionalities)
        m3_mit = M3Mitigation()

        m3_mit.cals_from_matrices(cal_matrices)
        quasi_probability = m3_mit.apply_correction(counts, qubits=list(cl_index_to_qubit_name.keys()))
        return quasi_probability

    def get_calibration_data(self) -> dict[str, np.ndarray]:
        """Get the calibration data for all qubits.

        Returns:
            Dictionary mapping qubit names to their calibration matrices (2x2 numpy arrays).

        """
        cal_data = {}
        for qubit_name, data in self.charact_data.items():
            if self._data_type == "matrix":
                cal_data[qubit_name] = np.array(data)
            else:
                symm_error = float(data)
                cal_data[qubit_name] = np.array([[1 - symm_error, symm_error], [symm_error, 1 - symm_error]])
        return cal_data

    def symmetrize_charact_data(self) -> None:
        """Symmetrize the characterization data by averaging error rates.

        Converts matrices to symmetrized scalar error rates by averaging P(0|1) and P(1|0).
        If data is already scalar, this method has no effect.
        """
        if self._data_type == "scalar":
            return

        symmetrized_data: dict[str, float] = {}
        for qubit_name, data in self.charact_data.items():
            if not isinstance(data, np.ndarray) or data.shape != (2, 2):
                raise ValueError(
                    f"Characterization data for qubit {qubit_name} must be a 2x2 NumPy array, got {type(data)}."
                )
            symmetrized_data[qubit_name] = (data[0, 1] + data[1, 0]) / 2.0

        self.charact_data = symmetrized_data
        self._data_type = "scalar"

    @staticmethod
    def _symmetrize_copy(
        charact_data: dict[str, np.ndarray] | dict[str, float],
    ) -> dict[str, float]:
        """Return a symmetrized copy of characterization data without modifying the original.

        Converts 2×2 matrices to scalar error rates by averaging P(0|1) and P(1|0).
        Already-scalar data is returned as-is in a new dict.

        Args:
            charact_data: Characterization data (per-qubit matrices or scalars).

        Returns:
            New dict mapping qubit names to symmetrized scalar error rates.

        """
        # Check if all values are already scalar (no matrices to symmetrize)
        if all(not isinstance(v, np.ndarray) for v in charact_data.values()):
            return {k: float(v) for k, v in charact_data.items()}

        # Otherwise, process each value (convert matrices, keep scalars)
        symmetrized: dict[str, float] = {}
        for qubit_name, data in charact_data.items():
            if isinstance(data, np.ndarray) and data.shape == (2, 2):
                symmetrized[qubit_name] = float((data[0, 1] + data[1, 0]) / 2.0)
            else:
                symmetrized[qubit_name] = float(data)
        return symmetrized

    @staticmethod
    def _normalize_observables(  # noqa: PLR0912
        observables: list[list[str] | str | list[int]] | None,
        qubit_to_bit_mapping: dict[str, int],
    ) -> list[list[str]]:
        """Normalize observables to the canonical qubit-name list format.

        Supports three input formats, auto-detected per observable element:

        1. **Qubit names** (current): ``[["QB3", "QB5"], ["QB1"]]``
        2. **Pauli strings** (Z/I only): ``["IIIZZIZI", "ZIIIIIIII"]``
        3. **Classical bit indices**: ``[[2, 4], [0]]``

        All formats are converted to lists of qubit names using ``qubit_to_bit_mapping``.

        Args:
            observables: List of observables in any supported format, or ``None`` for
                the full register as a single observable.
            qubit_to_bit_mapping: Canonical mapping from qubit names to bit indices.

        Returns:
            Observables in canonical ``list[list[str]]`` format.

        Raises:
            ValueError: If an observable format is unrecognized or references
                invalid bit indices / qubit names.

        """
        if observables is None:
            return [list(qubit_to_bit_mapping.keys())]

        if not observables:
            raise ValueError("observables list cannot be empty.")

        bit_to_qubit = {v: k for k, v in qubit_to_bit_mapping.items()}
        num_bits = len(qubit_to_bit_mapping)
        normalized: list[list[str]] = []

        for obs in observables:
            if isinstance(obs, str):
                # Pauli string format: must contain only 'I' and 'Z'
                if not all(c in ("I", "Z") for c in obs):
                    raise ValueError(f"Pauli string observable '{obs}' contains characters other than 'I' and 'Z'.")
                if len(obs) != num_bits:
                    raise ValueError(
                        f"Pauli string length ({len(obs)}) does not match the number of measured qubits ({num_bits})."
                    )
                # Pauli string position i → bit index (n-1-i) in little-endian convention
                qubit_names: list[str] = []
                for i, c in enumerate(obs):
                    if c == "Z":
                        bit_idx = num_bits - 1 - i
                        if bit_idx not in bit_to_qubit:
                            raise ValueError(
                                f"Bit index {bit_idx} (from Pauli string position {i}) "
                                f"not found in qubit_to_bit_mapping."
                            )
                        qubit_names.append(bit_to_qubit[bit_idx])
                if not qubit_names:
                    raise ValueError(f"Pauli string '{obs}' has no Z operators; at least one Z is required.")
                normalized.append(qubit_names)
            elif isinstance(obs, list):
                if not obs:
                    raise ValueError("Observable list cannot be empty.")
                first = obs[0]
                if isinstance(first, int):
                    # Bit index format
                    qubit_names_from_idx: list[str] = []
                    for idx in obs:
                        if not isinstance(idx, int):
                            raise ValueError(
                                f"Mixed types in observable list: expected all int, got {type(idx).__name__}."
                            )
                        if idx not in bit_to_qubit:
                            raise ValueError(
                                f"Bit index {idx} not found in qubit_to_bit_mapping. "
                                f"Available indices: {sorted(bit_to_qubit.keys())}."
                            )
                        qubit_names_from_idx.append(bit_to_qubit[idx])
                    normalized.append(qubit_names_from_idx)
                elif isinstance(first, str):
                    # Qubit name format (current / canonical)
                    normalized.append(cast(list[str], obs))
                else:
                    raise ValueError(
                        f"Unrecognized observable element type: {type(first).__name__}. "
                        f"Expected str (qubit name) or int (bit index)."
                    )
            else:
                raise ValueError(
                    f"Unrecognized observable type: {type(obs).__name__}. "
                    f"Expected list[str] (qubit names), str (Pauli string), or list[int] (bit indices)."
                )

        return normalized

    def mitigate_counts(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        experiment_counts: list[SampledDistribution],
        qubit_to_bit_mapping: (dict[str, int] | TwirledCircuit | list[TwirledCircuit]),
        observables: list[list[str] | str | list[int]] | None = None,
        twirled: bool = False,
        force_mitigation: bool = False,
        nearest_probability: bool = True,
        client: Pulla | None = None,
        auto_characterize_shots: int = 10000,
        auto_characterize_circuits: int = 20,
    ) -> MitigationResult:
        """Mitigate a list of measurement count dictionaries for one or more observables.

        Bitstrings and bit indices follow **little-endian** (Qiskit) convention:
        bit index 0 is the **rightmost** character of the bitstring.

        Args:
            experiment_counts: Measurement count dictionaries.
                Each dict maps bitstrings (e.g., ``"0101"``) to shot counts.
                Bitstrings follow **little-endian** convention
                (rightmost character = bit index 0).
            qubit_to_bit_mapping: A property of the transpiled circuit: it maps every measured
                qubit name to its classical bit index and is fixed once the circuit has been run.

                Accepts multiple formats (auto-detected):

                - **dict** (canonical): ``{"QB5": 0, "QB12": 1, "QB3": 2, "QB7": 3}``
                - **TwirledCircuit**: extracts mapping from ``measured_qubits``
                - **list[TwirledCircuit]**: extracts from the first circuit

            observables: Observables to mitigate and compute ZZ…Z expectation values for.
                ``observables`` is a list of qubit-name subsets; for
                each subset the raw counts are first marginalized to the corresponding bit indices
                (looked up from ``qubit_to_bit_mapping``), then readout-error-mitigated, and finally a
                ZZ…Z expectation value is computed.

                Supports three formats per element (auto-detected):

                - **Qubit names** (current): ``[["QB3", "QB5"], ["QB1"]]``
                - **Pauli strings** (Z/I only): ``["IIIZZIZI", "ZIIIIIIII"]``
                - **Bit indices**: ``[[2, 4], [0]]``

                If ``None``, a single observable covering all qubits
                in ``qubit_to_bit_mapping`` is used.  Formats can be mixed
                within a single call.
            twirled: If ``True``, symmetrize the characterization data before mitigation
                by averaging P(0|1) and P(1|0) into a single error rate per qubit.
                This is appropriate when the counts were obtained using readout
                twirling.  The original ``charact_data`` on the instance is **not**
                modified; a temporary symmetrized copy is used.  Default is ``False``.
            force_mitigation: If True, apply mitigation even if complexity exceeds threshold.
            nearest_probability: If True, project each quasi-probability distribution onto the
                nearest valid probability distribution using ``mthree``'s
                ``nearest_probability_distribution`` method. Eliminates negative values
                at the cost of a small bias. Default is True.
            client: Optional client for automatic characterization of missing qubits on the quantum computer.
            auto_characterize_shots: Shots to use when auto-characterizing (default: 10000).
            auto_characterize_circuits: Number of circuits for auto-characterization (default: 20).

        Returns:
            Dictionary containing:

            - ``"mitigated_counts"``: ``list[list[dict[str, float]]]`` —
              ``result["mitigated_counts"][i][j]`` is the mitigated quasi-probability
              distribution for circuit ``i`` under observable ``j``.
            - ``"expectation_values"``: ``list[list[float]]`` —
              ``result["expectation_values"][i][j]`` is the ZZ…Z expectation value
              for circuit ``i`` under observable ``j``.
            - ``"characterization_performed"``: ``bool`` — True if auto-characterization ran.

        Raises:
            ValueError: If ``experiment_counts`` or ``qubit_to_bit_mapping`` are empty, if any
                       observable qubit is not in ``qubit_to_bit_mapping``, or characterization
                       data is missing and no ``client`` is provided.

        Example — full-distribution mitigation::

            >>> mitigator = ReadoutErrorMitigation(charact_data)
            >>> result = mitigator.mitigate_counts(
            ...     experiment_counts=[{"00": 450, "11": 450}],
            ...     qubit_to_bit_mapping={"QB1": 0, "QB2": 1},
            ... )
            >>> result["mitigated_counts"][0][0]   # circuit 0, observable 0
            >>> result["expectation_values"][0][0]  # <ZZ> for circuit 0

        Example — multiple observables in one call::

            >>> result = mitigator.mitigate_counts(
            ...     experiment_counts=[{"0000": 450, "1111": 450}],
            ...     qubit_to_bit_mapping={"QB1": 0, "QB2": 1, "QB3": 2, "QB4": 3},
            ...     observables=[
            ...         ["QB1", "QB2", "QB3", "QB4"],  # full ZZZZ
            ...         ["QB1", "QB2"],                 # ZZ on first two qubits
            ...         ["QB3", "QB4"],                 # ZZ on last two qubits
            ...     ],
            ... )
            >>> result["expectation_values"][0]  # [<ZZZZ>, <Z0Z1>, <Z2Z3>]

        Example — auto-characterization::

            >>> mitigator = ReadoutErrorMitigation(charact_data={})
            >>> result = mitigator.mitigate_counts(
            ...     experiment_counts=counts_list,
            ...     qubit_to_bit_mapping={"QB1": 0, "QB2": 1},
            ...     client=client,
            ... )

        Example — twirled mitigation (same instance reusable for both)::

            >>> rem = ReadoutErrorMitigation(charact_data=probs["charact_data"])
            >>> result_twirled = rem.mitigate_counts(
            ...     [twirled_counts], mapping, twirled=True,
            ... )
            >>> result_standard = rem.mitigate_counts(
            ...     [standard_counts], mapping, twirled=False,
            ... )

        Example — Pauli-string observables (natural for Qiskit users)::

            >>> result = rem.mitigate_counts(
            ...     experiment_counts=[counts],
            ...     qubit_to_bit_mapping={"QB1": 0, "QB2": 1, "QB3": 2, "QB4": 3},
            ...     observables=["IZIZ", "ZZII"],
            ...     twirled=True,
            ... )

        Example — bit-index observables (framework-agnostic)::

            >>> result = rem.mitigate_counts(
            ...     experiment_counts=[counts],
            ...     qubit_to_bit_mapping={"QB1": 0, "QB2": 1, "QB3": 2, "QB4": 3},
            ...     observables=[[0, 2], [2, 3]],
            ... )

        Example — TwirledCircuit as mapping::

            >>> result = rem.mitigate_counts(
            ...     experiment_counts=[counts],
            ...     qubit_to_bit_mapping=twirled_circuit,
            ...     twirled=True,
            ... )

        """
        if not experiment_counts:
            raise ValueError("experiment_counts list cannot be empty.")

        # Normalize qubit_to_bit_mapping to canonical dict[str, int] format
        # Accepts dict, TwirledCircuit, or list[TwirledCircuit] for convenience
        mapping_dict: dict[str, int]
        if isinstance(qubit_to_bit_mapping, dict):
            mapping_dict = qubit_to_bit_mapping
        elif isinstance(qubit_to_bit_mapping, TwirledCircuit):
            # Extract from TwirledCircuit.measured_qubits
            meas_mapping = dict(enumerate(qubit_to_bit_mapping.measured_qubits))
            mapping_dict = {v: k for k, v in meas_mapping.items()}
        elif isinstance(qubit_to_bit_mapping, list):
            # Extract from first circuit in list
            if not qubit_to_bit_mapping:
                raise ValueError("qubit_to_bit_mapping list cannot be empty.")
            first = qubit_to_bit_mapping[0]
            if not isinstance(first, TwirledCircuit):
                raise TypeError(f"Expected list of TwirledCircuit, got list of {type(first).__name__}.")
            meas_mapping = dict(enumerate(first.measured_qubits))
            mapping_dict = {v: k for k, v in meas_mapping.items()}
        else:
            raise TypeError(
                f"qubit_to_bit_mapping must be a dict, TwirledCircuit, "
                f"or list[TwirledCircuit]. Got {type(qubit_to_bit_mapping).__name__}."
            )

        if not mapping_dict:
            raise ValueError("qubit_to_bit_mapping must be non-empty after normalization.")

        # From here on, use only the canonical dict format
        qubit_to_bit_mapping = mapping_dict

        # Normalize observables to canonical list[list[str]] format
        normalized_observables = self._normalize_observables(observables, qubit_to_bit_mapping)

        # Validate that every observable qubit exists in qubit_to_bit_mapping
        for obs in normalized_observables:
            for q in obs:
                if q not in qubit_to_bit_mapping:
                    raise ValueError(
                        f"Qubit '{q}' in observables is not in qubit_to_bit_mapping. "
                        f"Available qubits: {list(qubit_to_bit_mapping.keys())}."
                    )

        # Collect unique qubit names that need characterization data
        all_qubit_names: set[str] = {q for obs in normalized_observables for q in obs}

        # Check if characterization is needed
        missing_qubits = [q for q in all_qubit_names if q not in self.charact_data]
        characterization_performed = False

        if missing_qubits:
            if client is None:
                raise ValueError(
                    f"Characterization data missing for qubits {missing_qubits}. "
                    f"Available qubits: {list(self.charact_data.keys())}. "
                    f"Provide a client instance to auto-characterize missing qubits."
                )

            warnings.warn(
                f"Auto-characterizing missing qubits {missing_qubits} on the quantum computer. "
                f"This will run {auto_characterize_circuits} circuits with {auto_characterize_shots} shots."
            )

            job, job_info = run_calibration_circuits(
                client=client,
                qubits=missing_qubits,
                number_of_circuits=auto_characterize_circuits,
                shots=auto_characterize_shots,
            )

            counts_list = retrieve_calibration_results(job, job_info)
            charact_result = compute_error_probabilities(counts_list)
            new_charact_data = charact_result["charact_data"]

            if self._data_type == "scalar":
                scalar_updates: dict[str, float] = {
                    qubit_name: float((matrix[0, 1] + matrix[1, 0]) / 2.0)
                    for qubit_name, matrix in new_charact_data.items()
                }
                scalar_charact_data = cast(dict[str, float], self.charact_data)
                scalar_charact_data.update(scalar_updates)
                self._warn_high_error_rates(scalar_updates)
            else:
                matrix_charact_data = cast(dict[str, np.ndarray], self.charact_data)
                matrix_charact_data.update(new_charact_data)
                self._warn_high_error_rates(new_charact_data)

            if self._data_type is None:
                self._data_type = self._validate_charact_data()

            characterization_performed = True

        # Prepare effective characterization data for mitigation
        effective_charact_data = self.charact_data
        effective_data_type = self._data_type
        if twirled:
            effective_charact_data = self._symmetrize_copy(self.charact_data)
            effective_data_type = "scalar"

        # Mitigate all experiment counts for each observable
        all_mitigated_counts = []
        all_expectation_values = []

        for counts in experiment_counts:
            mitigated_per_obs = []
            exp_vals_per_obs = []

            for obs_qubits in normalized_observables:
                # Sort by bit index for a consistent marginalization order
                obs_qubits_sorted = sorted(obs_qubits, key=lambda q: qubit_to_bit_mapping[q])
                support = [qubit_to_bit_mapping[q] for q in obs_qubits_sorted]

                # Marginalize raw counts to this subset of bit indices
                marginalized = marginalize_counts(counts, support)
                cl_map = {i: obs_qubits_sorted[i] for i in range(len(obs_qubits_sorted))}

                quasi_mitigated = self._mitigate(
                    marginalized,
                    cl_map,
                    force_mitigation,
                    charact_data=effective_charact_data,
                    data_type=effective_data_type,
                )
                exp_vals_per_obs.append(expectation_from_counts(dict(quasi_mitigated)))
                mitigated_per_obs.append(
                    dict(quasi_mitigated.nearest_probability_distribution())
                    if nearest_probability and isinstance(quasi_mitigated, QuasiDistribution)
                    else dict(quasi_mitigated)
                )

            all_mitigated_counts.append(mitigated_per_obs)
            all_expectation_values.append(exp_vals_per_obs)

        return {
            "mitigated_counts": cast(list[list[dict[str, float]]], all_mitigated_counts),
            "expectation_values": all_expectation_values,
            "characterization_performed": characterization_performed,
        }
