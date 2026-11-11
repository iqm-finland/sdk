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

"""Tests for pulse-level global folding used by ZNE."""

import cmath
from collections import Counter
import math

from iqm.error_reduction_tools.twirling.twirling_api import rot_string_for
from iqm.error_reduction_tools.utils.circuit_utils import TwirledCircuit
from iqm.error_reduction_tools.zne.zne_modifiers import (
    as_fold_count,
    compact_factor,
    fold_global_pulse,
    invert_pulse_operation,
    invert_pulse_operations,
    noise_factor_for,
)
import numpy as np
import pytest

from iqm.pulse import Circuit as PulseCircuit
from iqm.pulse import CircuitOperation


def prx(qubit, angle, phase=0.0):
    return CircuitOperation(name="prx", locus=(qubit,), args={"angle": angle, "phase": phase})


def cz(qubit_a, qubit_b):
    return CircuitOperation(name="cz", locus=(qubit_a, qubit_b), args={})


def measure(qubit, key):
    return CircuitOperation(name="measure", locus=(qubit,), args={"key": key})


def bell_operations():
    """A small circuit with 4 prx, 2 cz and 2 terminal measurements."""
    return [
        prx("QB1", math.pi / 2, math.pi / 2),
        prx("QB2", math.pi, 0.0),
        cz("QB1", "QB2"),
        prx("QB1", math.pi / 4, 0.3),
        cz("QB1", "QB2"),
        prx("QB2", math.pi / 2, 0.7),
        measure("QB1", "m_0"),
        measure("QB2", "m_1"),
    ]


def bell_circuit():
    return PulseCircuit(name="bell", instructions=tuple(bell_operations()))


def chain_operations():
    """Three cz gates on three distinct loci, so what each fold reaches is identifiable per locus."""
    return [
        prx("QB1", math.pi / 2, 0.0),
        cz("QB1", "QB2"),
        prx("QB2", math.pi / 3, 0.2),
        cz("QB2", "QB3"),
        prx("QB3", math.pi / 4, 0.5),
        cz("QB3", "QB4"),
        prx("QB4", math.pi / 5, 0.8),
        measure("QB1", "m_0"),
    ]


CHAIN_QUBITS = ("QB1", "QB2", "QB3", "QB4")


def chain():
    return PulseCircuit(name="chain", instructions=tuple(chain_operations()))


def circuit_of(operations):
    return PulseCircuit(name="u", instructions=tuple(operations))


def ops_of(circuit):
    return circuit.operations if isinstance(circuit, TwirledCircuit) else list(circuit.instructions)


def op_counts(circuit):
    return Counter(operation.name for operation in ops_of(circuit))


def cz_loci(circuit):
    return Counter(operation.locus for operation in ops_of(circuit) if operation.name == "cz")


def _prx_matrix(angle, phase):
    """The native prx: a rotation by ``angle`` about an equatorial axis at ``phase``."""
    cos, sin = math.cos(angle / 2), math.sin(angle / 2)
    return np.array(
        [
            [cos, -1j * cmath.exp(-1j * phase) * sin],
            [-1j * cmath.exp(1j * phase) * sin, cos],
        ]
    )


def unitary_of(circuit, qubits):
    """Dense unitary of a native pulse circuit, with ``qubits[0]`` the most significant bit.

    Barriers and measurements carry no unitary and are skipped. Skipping measurements is safe
    because ``fold_global_pulse`` refuses a circuit whose measurements are not terminal, which
    ``test_mid_circuit_measurement_raises`` pins.
    """
    position = {qubit: index for index, qubit in enumerate(qubits)}
    size = len(qubits)
    unitary = np.eye(2**size, dtype=complex)
    for operation in ops_of(circuit):
        if operation.name in ("barrier", "measure"):
            continue
        if operation.name == "prx":
            index = position[operation.locus[0]]
            full = np.kron(
                np.kron(np.eye(2**index), _prx_matrix(operation.args["angle"], operation.args["phase"])),
                np.eye(2 ** (size - index - 1)),
            )
        elif operation.name == "cz":
            first, second = (position[qubit] for qubit in operation.locus)
            full = np.diag(
                [
                    -1.0 + 0j if (state >> (size - 1 - first)) & 1 and (state >> (size - 1 - second)) & 1 else 1.0 + 0j
                    for state in range(2**size)
                ]
            )
        else:
            raise AssertionError(f"unitary_of does not handle {operation.name!r}")
        unitary = full @ unitary
    return unitary


def signature(operation):
    """A hashable, comparable summary of an operation: name, locus and rounded prx args."""
    angle = operation.args.get("angle")
    phase = operation.args.get("phase")
    return (
        operation.name,
        operation.locus,
        None if angle is None else round(angle, 12),
        None if phase is None else round(phase, 12),
    )


def signatures(operations):
    return [signature(operation) for operation in operations]


def test_invert_prx_negates_angle_and_keeps_phase():
    """Test that inverting a prx gate negates its angle and keeps its phase."""
    inverted = invert_pulse_operation(prx("QB1", 0.4, 1.2))

    assert inverted.name == "prx"
    assert inverted.locus == ("QB1",)
    assert inverted.args["angle"] == pytest.approx(-0.4)
    assert inverted.args["phase"] == pytest.approx(1.2)


def test_invert_cz_is_self_inverse():
    """Test that a cz gate is its own inverse."""
    original = cz("QB1", "QB2")
    inverted = invert_pulse_operation(original)

    assert inverted.name == "cz"
    assert inverted.locus == original.locus


def test_invert_does_not_mutate_input():
    """Test that inverting an operation does not change the input."""
    original = prx("QB1", 0.4, 1.2)
    invert_pulse_operation(original)

    assert original.args["angle"] == pytest.approx(0.4)


def test_invert_measure_raises():
    """Test that inverting a measurement raises."""
    with pytest.raises(ValueError, match="no well-defined inverse"):
        invert_pulse_operation(measure("QB1", "m_0"))


def test_invert_sequence_reverses_and_inverts():
    """Test that inverting a sequence reverses it and inverts every operation."""
    operations = [prx("QB1", 0.4), cz("QB1", "QB2"), prx("QB2", 0.9)]

    inverted = invert_pulse_operations(operations)

    assert [operation.name for operation in inverted] == ["prx", "cz", "prx"]
    assert inverted[0].locus == ("QB2",)
    assert inverted[0].args["angle"] == pytest.approx(-0.9)
    assert inverted[2].locus == ("QB1",)
    assert inverted[2].args["angle"] == pytest.approx(-0.4)


def test_zero_folds_is_a_copy():
    """Test that zero folds return the same operations."""
    circuit = bell_circuit()

    folded = fold_global_pulse(circuit, 0)

    assert op_counts(folded) == op_counts(circuit)
    assert folded.metadata["achieved_folds"] == 0


@pytest.mark.parametrize("folds", [1, 2, 3])
def test_folding_is_exact(folds):
    """Test that s folds repeat every gate 1 + 2s times and keep the measurements."""
    circuit = bell_circuit()
    base = op_counts(circuit)

    folded = fold_global_pulse(circuit, folds)
    counts = op_counts(folded)

    factor = 1 + 2 * folds
    assert counts["cz"] == factor * base["cz"]
    assert counts["prx"] == factor * base["prx"]
    # Measurements are never folded.
    assert counts["measure"] == base["measure"]
    assert folded.metadata["achieved_folds"] == folds


def test_one_fold_produces_the_exact_U_Udag_U_sequence():
    """Test that one fold gives U U^dagger U followed by the measurements."""
    body = bell_operations()[:-2]  # drop the two terminal measurements
    measurements = bell_operations()[-2:]
    expected = body + invert_pulse_operations(body) + body + measurements

    folded = fold_global_pulse(bell_circuit(), 1)

    assert signatures(ops_of(folded)) == signatures(expected)


@pytest.mark.parametrize("folds", [1.5, 5 / 3, 0.5, 1.01, -1])
def test_anything_but_a_whole_count_is_refused(folds):
    """Test that a fractional or negative number of folds is rejected."""
    with pytest.raises(ValueError, match="at least 0"):
        fold_global_pulse(chain(), folds)


@pytest.mark.parametrize("folds", [0, 1, 2])
def test_a_fold_amplifies_every_gate_equally(folds):
    """Test that folding repeats every cz gate the same number of times."""
    loci = (("QB1", "QB2"), ("QB2", "QB3"), ("QB3", "QB4"))

    counts = dict(cz_loci(fold_global_pulse(chain(), folds)))

    assert counts == dict.fromkeys(loci, 1 + 2 * folds)


@pytest.mark.parametrize("folds", [0, 1, 2, 3])
def test_folding_leaves_the_unitary_unchanged(folds):
    """Test that folding does not change the unitary of the circuit."""
    circuit = chain()

    folded = fold_global_pulse(circuit, folds)

    assert np.allclose(unitary_of(folded, CHAIN_QUBITS), unitary_of(circuit, CHAIN_QUBITS), atol=1e-10)


def test_the_unitary_check_has_teeth():
    """Test that the unitary check detects an inverse that also negates the phase."""
    original = prx("QB1", 0.7, 1.3)
    identity = np.eye(2, dtype=complex)

    correct = circuit_of([original, invert_pulse_operation(original)])
    wrong = circuit_of([original, prx("QB1", -0.7, -1.3)])

    assert np.allclose(unitary_of(correct, ("QB1",)), identity, atol=1e-10)
    assert not np.allclose(unitary_of(wrong, ("QB1",)), identity, atol=1e-10)


def test_accepts_twirled_circuit_and_keeps_that_representation():
    """Test that folding a TwirledCircuit returns a TwirledCircuit."""
    folded = fold_global_pulse(TwirledCircuit(bell_operations()), 1)

    assert isinstance(folded, TwirledCircuit)
    assert op_counts(folded)["cz"] == 6
    assert folded.metadata["achieved_folds"] == 1


def test_folding_a_randomized_circuit_preserves_the_readout_twirling_basis():
    """Test that folding a randomized circuit keeps its readout twirling basis."""
    circuit = TwirledCircuit(bell_operations())
    circuit.rot_dict = {"QB1": "X", "QB2": "I"}

    folded = fold_global_pulse(circuit, 3)

    assert folded.rot_dict == {"QB1": "X", "QB2": "I"}
    assert folded.measured_qubits == circuit.measured_qubits
    assert rot_string_for(folded) == rot_string_for(circuit)


def test_folding_a_randomized_circuit_does_not_mutate_the_original():
    """Test that folding a randomized circuit does not change the original."""
    circuit = TwirledCircuit(bell_operations())
    circuit.rot_dict = {"QB1": "X", "QB2": "I"}
    original_operations = len(circuit.operations)

    folded = fold_global_pulse(circuit, 3)
    folded.rot_dict["QB1"] = "I"

    assert len(circuit.operations) == original_operations
    assert circuit.rot_dict == {"QB1": "X", "QB2": "I"}


def test_folding_keeps_metadata_the_input_already_carried():
    """Test that folding keeps the metadata of the input and adds the number of folds."""
    circuit = PulseCircuit(name="bell", instructions=tuple(bell_operations()), metadata={"origin": "test"})

    folded = fold_global_pulse(circuit, 3)

    assert folded.metadata["origin"] == "test"
    assert folded.metadata["achieved_folds"] == 3


def test_barriers_are_folded_with_the_body():
    """Test that barriers are folded together with the gates."""
    operations = [
        prx("QB1", 0.4),
        cz("QB1", "QB2"),
        CircuitOperation(name="barrier", locus=("QB1", "QB2"), args={}),
        cz("QB1", "QB2"),
        measure("QB1", "m_0"),
    ]

    folded = fold_global_pulse(PulseCircuit(name="c", instructions=tuple(operations)), 1)

    assert op_counts(folded)["cz"] == 6
    assert folded.metadata["achieved_folds"] == 1


def test_a_circuit_of_single_qubit_gates_still_folds():
    """Test that a circuit with only single-qubit gates is folded."""
    operations = [prx("QB1", 0.4), prx("QB1", 0.8), measure("QB1", "m_0")]

    folded = fold_global_pulse(PulseCircuit(name="c", instructions=tuple(operations)), 1)

    assert op_counts(folded)["prx"] == 6


def test_original_circuit_is_not_modified():
    """Test that folding does not change the original circuit."""
    circuit = bell_circuit()
    original_angles = [operation.args.get("angle") for operation in circuit.instructions]

    fold_global_pulse(circuit, 3)

    assert [operation.args.get("angle") for operation in circuit.instructions] == original_angles


def test_gates_after_measuring_another_qubit_are_folded_and_measurements_moved_last():
    """Test that gates following the measurement of another qubit are folded, with the measurements last."""
    operations = [prx("QB1", 0.4), measure("QB1", "m_0"), prx("QB2", 0.9), measure("QB2", "m_1")]
    folded = fold_global_pulse(PulseCircuit(name="c", instructions=tuple(operations)), 1)
    assert [op.name for op in folded.instructions] == ["prx"] * 6 + ["measure"] * 2


def test_mid_circuit_measurement_raises():
    """Test that a gate after a measurement raises."""
    operations = [prx("QB1", 0.4), measure("QB1", "m_0"), cz("QB1", "QB2")]

    with pytest.raises(ValueError, match="terminal"):
        fold_global_pulse(PulseCircuit(name="c", instructions=tuple(operations)), 3)


def test_circuit_without_gates_raises():
    """Test that a circuit without gates raises."""
    operations = [measure("QB1", "m_0")]

    with pytest.raises(ValueError, match="no foldable gates"):
        fold_global_pulse(PulseCircuit(name="c", instructions=tuple(operations)), 3)


def test_invalid_circuit_type_raises():
    """Test that an input that is not a circuit raises."""
    with pytest.raises(TypeError, match="iqm.pulse.Circuit"):
        fold_global_pulse("not a circuit", 3)


class TestFoldCountNormalization:
    """Folds are whole counts of ``U^dagger U`` insertions, so anything else is refused."""

    @pytest.mark.parametrize("value,expected", [(0, 0), (1.0, 1), (7, 7)])
    def test_a_whole_number_is_narrowed_to_an_int(self, value, expected):
        """Test that a whole fold count is returned as an int."""
        result = as_fold_count(value)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("value", [1.5, 0.5, -1, -2.0])
    def test_a_fraction_or_a_negative_count_is_refused(self, value):
        """Test that a fractional or negative fold count is rejected."""
        with pytest.raises(ValueError, match="at least 0"):
            as_fold_count(value)

    @pytest.mark.parametrize("value", [True, "2", None])
    def test_something_that_is_not_a_number_is_refused(self, value):
        """Test that a fold count that is not a number is rejected."""
        with pytest.raises(TypeError, match="must be a real number"):
            as_fold_count(value)

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_a_non_finite_count_is_refused(self, value):
        """Test that a non-finite fold count is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            as_fold_count(value)

    @pytest.mark.parametrize("folds,factor", [(0, 1), (1, 3), (2, 5), (7, 15)])
    def test_s_folds_amplify_every_gate_by_1_plus_2s(self, folds, factor):
        """Test that s folds give the noise factor 1 + 2s."""
        assert noise_factor_for(folds) == factor


class TestNoiseFactorNormalization:
    """Post-processing reads factors that were already executed, so it only narrows the spelling."""

    @pytest.mark.parametrize("value,expected", [(1.0, 1), (2.0, 2), (5, 5)])
    def test_a_whole_number_is_narrowed_to_an_int(self, value, expected):
        """Test that a whole noise factor is returned as an int."""
        result = compact_factor(value)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("value", [1.5, 1.01, 2.25])
    def test_a_fraction_stays_a_float(self, value):
        """Test that a fractional noise factor stays a float."""
        result = compact_factor(value)
        assert result == value
        assert isinstance(result, float)
