# Copyright 2024-2025 IQM
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

from iqm.pulla.utils import iqm_circuit_to_gate_implementation
from iqm.pulse import Circuit
from iqm.pulse import CircuitOperation as I
from iqm.pulse.quantum_ops import QuantumOp


def test_iqm_circuit_to_gate_implementation(pulla_on_spark):
    qc = Circuit(
        name="very_custom",
        instructions=[
            I(name="prx", locus=("0",), args={"angle": 0.5, "phase": 0}),
            I(name="prx", locus=("0",), args={"angle": 0.3, "phase": 0}),
            I(name="prx", locus=("1",), args={"angle": 0.1, "phase": 0.2}),
            I(name="barrier", locus=("0", "1"), args={}),
            I(name="cz", locus=("0", "1"), args={}),
            I(name="measure", locus=("0", "1"), args={"key": "m"}),
        ],
    )

    qubit_mapping = {"0": "QB3", "1": "QB4", "2": "QB1"}  # '2' unused!
    impl_class = iqm_circuit_to_gate_implementation(qc, qubit_mapping)

    compiler = pulla_on_spark.get_standard_compiler()
    compiler.add_implementation("very_custom", "CustomImpl", impl_class, quantum_op=QuantumOp("very_custom", arity=3))

    impl = compiler.builder.get_implementation("very_custom", ("QB3", "QB4", "QB1"))
    assert isinstance(impl, impl_class)

    timebox = impl()
    assert "PRX" in timebox[0].label
    assert "PRX" in timebox[1].label
    assert "PRX" in timebox[2].label
    assert "Barrier" in timebox[3].label
    assert "CZ" in timebox[4].label
    assert "Readout" in timebox[-1].label
