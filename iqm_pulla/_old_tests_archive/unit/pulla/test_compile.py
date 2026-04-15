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
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from iqm.pulla.utils_qiskit import qiskit_to_pulla


def test_compile_simple_qiskit_circuit(pulla_on_spark, qiskit_backend_spark):
    """
    Test compiling a simple qiskit circuit and building settings using the standard compiler constructed by Pulla.
    """
    qc = QuantumCircuit(5, 5)
    qc.x(0)
    qc.rx(0.1, 0)
    qc.r(0.1, 0.2, 1)
    qc.barrier(0, 1, 2, 3)
    qc.cz(0, 2)
    qc.measure_all()

    qreg = qc.qregs[0]
    layout = {qreg[k]: k for k in range(5)}
    qc = transpile(qc, backend=qiskit_backend_spark, initial_layout=layout)
    circuits, compiler = qiskit_to_pulla(pulla_on_spark, qiskit_backend_spark, qc)

    _, context = compiler.compile(circuits)
    compiler.build_settings(context, shots=20)


def test_compile_idempotency_data(pulla_on_spark, qiskit_backend_spark):
    """
    Test that the compilation stages do not change the data.
    """
    qc = QuantumCircuit(5, 5)
    qc.x(0)
    qc.rx(0.1, 0)
    qc.r(0.1, 0.2, 1)
    qc.barrier(0, 1, 2, 3)
    qc.cz(0, 2)
    qc.measure_all()

    qreg = qc.qregs[0]
    layout = {qreg[k]: k for k in range(5)}
    qc = transpile(qc, backend=qiskit_backend_spark, initial_layout=layout)
    data, compiler = qiskit_to_pulla(pulla_on_spark, qiskit_backend_spark, qc)

    # First, run two full compilations.
    # Context is handled by the `compile()` method, so we only check whether data is unchanged
    data_after_compile_1, _ = compiler.compile(data)
    # If compilation stages are idempotent, repeated compilation should succeed
    data_after_compile_2, _ = compiler.compile(data)
    # and original data, and two compiled data objects should all be different
    data_object_ids = [id(obj) for obj in [data, data_after_compile_1, data_after_compile_2]]
    assert len(data_object_ids) == len(set(data_object_ids))


def test_compile_idempotency_context(pulla_on_spark, qiskit_backend_spark):
    """
    Test that the compilation stages do not change the context.
    """
    qc = QuantumCircuit(5, 5)
    qc.x(0)
    qc.rx(0.1, 0)
    qc.r(0.1, 0.2, 1)
    qc.barrier(0, 1, 2, 3)
    qc.cz(0, 2)
    qc.measure_all()

    qreg = qc.qregs[0]
    layout = {qreg[k]: k for k in range(5)}
    qc = transpile(qc, backend=qiskit_backend_spark, initial_layout=layout)
    data, compiler = qiskit_to_pulla(pulla_on_spark, qiskit_backend_spark, qc)
    context = compiler.compiler_context()

    # Run compilation stages manually. Up to us to handle content, so we check whether context is unchanged
    data_after_stage_1, context_after_stage_1 = compiler.stages[0].run(data, context)
    data_after_stage_2, context_after_stage_2 = compiler.stages[1].run(data_after_stage_1, context_after_stage_1)
    data_after_stage_3, context_after_stage_3 = compiler.stages[2].run(data_after_stage_2, context_after_stage_2)
    data_after_stage_4, context_after_stage_4 = compiler.stages[3].run(data_after_stage_3, context_after_stage_3)
    data_after_stage_5, context_after_stage_5 = compiler.stages[4].run(data_after_stage_4, context_after_stage_4)
    _, context_after_stage_6 = compiler.stages[5].run(data_after_stage_5, context_after_stage_5)

    # Check if all IDs are unique by comparing the length of the list with the length of the set of IDs
    context_object_ids = [
        id(obj)
        for obj in [
            context,
            context_after_stage_1,
            context_after_stage_2,
            context_after_stage_3,
            context_after_stage_4,
            context_after_stage_5,
            context_after_stage_6,
        ]
    ]
    assert len(context_object_ids) == len(set(context_object_ids))

    # # If compilation stages are idempotent, repeated manual compilation should succeed
    data_after_stage_1, context_after_stage_1 = compiler.stages[0].run(data, context)
    data_after_stage_2, context_after_stage_2 = compiler.stages[1].run(data_after_stage_1, context_after_stage_1)
    data_after_stage_3, context_after_stage_3 = compiler.stages[2].run(data_after_stage_2, context_after_stage_2)
    data_after_stage_4, context_after_stage_4 = compiler.stages[3].run(data_after_stage_3, context_after_stage_3)
    data_after_stage_5, context_after_stage_5 = compiler.stages[4].run(data_after_stage_4, context_after_stage_4)
    _, context_after_stage_6 = compiler.stages[5].run(data_after_stage_5, context_after_stage_5)
