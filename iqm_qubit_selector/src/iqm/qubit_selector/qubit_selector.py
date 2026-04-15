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

"""Automatically select the best layout for running an algorithm on IQM's quantum computer."""

from enum import Enum, StrEnum, auto
import logging
import os
import re
from typing import Any, TypeAlias, cast
import warnings

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qubit_selector.qiskit_utils import deflate_circuit, extract_2q_interactions, perform_backend_transpilation
from qiskit import QuantumCircuit, transpile
import rustworkx as rx

from iqm.pulla.pulla import Pulla
from iqm.pulla.utils_qiskit import qiskit_to_pulla

warnings.filterwarnings("ignore", category=UserWarning, module="iqm.iqm_server_client")
logger = logging.getLogger(__name__)
Layout: TypeAlias = list[int]
CZ_LOCUS_LENGTH = 2


class CalibrationType(StrEnum):
    """Enumeration representing different types of calibration methods.

    Attributes:
        CZ: Calibration type for CZ.
        CLIFFORD: Calibration type for CLIFFORD.
        SQG: Calibration type for 1Q.
        READOUT: Calibration type for readout.
        READOUT_QNDNESS: Calibration type for readout QNDness.
        T1: Calibration type for T1.
        T2: Calibration type for T2.

    """

    # Uppercase/lowercase matters, because enum values are used as dictionary
    # keys; do not replace the actual hardcoded values with ``enum.auto()``
    CZ = "CZ"
    CLIFFORD = "CLIFFORD"
    SQG = "1Q"
    READOUT = "readout"
    READOUT_QNDNESS = "readout_qndness"
    T1 = "t1"
    T2 = "t2"


class CostFunction(StrEnum):
    """Enumeration representing different types of cost functions used in the application.

    Attributes:
        GATE_COST_CZ: Represents the cost associated with gate operations.
        GATE_COST_CLIFFORD: Represents the cost associated with gate operations and CZ as CLIFFORD.


    """

    GATE_COST_CZ = auto()
    GATE_COST_CLIFFORD = auto()


class ObservationType(Enum):
    """Enumeration representing relevant keys to fetch for each operation in the observations."""

    CZ = "cz"
    CLIFFORD = ("cz", "clifford")
    SQG = "prx"
    READOUT = ("measure_fidelity", ".fidelity")
    READOUT_QNDNESS = ("measure", "qndness", ".fidelity")
    T1 = ("t1", "QB")
    T2 = ("t2", "QB")


class ReadoutMode(Enum):
    """Enumeration representing how to use readout fidelity in cost function evaluation."""

    NONE = "none"
    FIDELITY = "fidelity"
    QNDNESS = "qndness"


class LayoutGenerator:
    """Generates unique layouts for quantum circuits based on calibration data and backend architecture.

    Args:
        backend: The IQM backend instance.
        quantum_circuit: The quantum circuit to generate layouts for.
        weights: Weights for calibration types.
        additional_qubits: Number of additional qubits to consider.
        remove_qubits: List of qubit indices (in qiskit) to remove from consideration.
        num_trials: The number of trials for layout generation.

    """

    def __init__(
        self,
        backend: IQMBackendBase,
        quantum_circuit: QuantumCircuit,
        remove_qubits: list[int] | None = None,
        num_trials: int = 10000,
    ):
        self.num_trials = num_trials
        self.backend = backend
        self.remove_qubits = remove_qubits
        self.quantum_circuit = quantum_circuit
        self.num_qubit = quantum_circuit.num_qubits
        self.transpiled_qc: QuantumCircuit | None = None

        self.transpiled_qc = transpile(self.quantum_circuit, backend=self.backend, optimization_level=3)

    def generate_unique_layouts(self) -> list[Layout]:
        """Generate unique layouts based on the architecture of the backend.

        Returns:
            A list of unique layouts represented as lists of qubit indices.

        """
        return self._generate_unique_layouts_for_crystal()

    def _generate_unique_layouts_for_crystal(self) -> list[Layout]:
        """Generate unique layouts for the crystal topology based on the transpiled quantum circuit.

        Returns:
            A list of unique layouts represented as lists of qubit indices.

        """
        deflated_qc = deflate_circuit(self.transpiled_qc)
        qubits = deflated_qc.qubits
        interactions = extract_2q_interactions(deflated_qc)
        qubit_list = list(range(self.backend.num_qubits))

        if self.remove_qubits is not None:
            updated_qubits = [qubit for qubit in qubit_list if qubit not in self.remove_qubits]
            try:
                cmap = self.backend.coupling_map.reduce(mapping=updated_qubits)
            except ValueError as e:
                raise ValueError(
                    f"Include the removed qubits: {self.remove_qubits}. Error during coupling map reduction: {e}"
                ) from e
        else:
            cmap = self.backend.coupling_map
            updated_qubits = qubit_list

        # Build graphs
        cm_graph = cmap.graph.to_undirected()
        im_graph: rx.PyGraph = rx.PyGraph(multigraph=False)
        im_graph.add_nodes_from(range(len(qubits)))
        im_graph.add_edges_from_no_data(interactions)
        cm_nodes = updated_qubits

        # Run VF2
        mappings = rx.vf2_mapping(
            cm_graph,
            im_graph,
            subgraph=True,
            id_order=False,
            induced=False,
            call_limit=self.num_trials,
        )

        layouts = []

        for mapping in mappings:
            temp_list: list[int | None] = [None] * deflated_qc.num_qubits
            for cm_i, im_i in mapping.items():
                physical_q = cm_nodes[cm_i]  # Correct mapping!
                logical_q = deflated_qc.find_bit(qubits[im_i]).index
                temp_list[logical_q] = physical_q

            if None in temp_list:
                raise ValueError(
                    f"Failed to find logical-to-physical qubit mapping for logical qubit no. {temp_list.index(None)}"
                )

            layouts.append(temp_list)

        # Unique sets
        sets = []

        for layout in layouts:
            s = set(layout)
            if s not in sets:
                sets.append(s)

        fitting_layouts = cast(list[Layout], [list(layout) for layout in sets])
        logger.info("Number of layouts to evaluate: %d", len(fitting_layouts))
        return fitting_layouts


class CostEvaluator:
    """Evaluates cost functions for quantum circuit layouts.

    This class is responsible for computing the costs associated with different layouts of quantum circuits
    based on specified cost functions. It utilizes calibration data and backend information to assess the
    fidelity of quantum operations and idle times, allowing for the optimization of quantum circuit layouts.

    Args:
        backend: The IQM backend instance.
        quantum_circuit: The quantum circuit to be managed.
        cost_function: The cost function to be used for optimization.
        readoutmode: The readoutmode to use for including readout errors in the cost calculation.
        weights: Weights for calibration types.
        remove_qubits: List of qubit indices (in qiskit) to remove from consideration.
        num_trials: The number of trials for layout generation.
        additional_qubits: Number of additional qubits to consider.
        layouts: Predefined layouts to evaluate.

    Raises:
        Exception: If there is an error during the transpilation process.

    """

    def __init__(
        self,
        backend: IQMBackendBase,
        quantum_circuit: QuantumCircuit,
        cost_function: CostFunction = CostFunction.GATE_COST_CZ,
        readoutmode: ReadoutMode = ReadoutMode.NONE,
        remove_qubits: list[int] | None = None,
        num_trials: int = 10000,
        layouts: list[Layout] | None = None,
    ):
        self.iqm_url = os.getenv("IQM_SERVER_URL")

        if self.iqm_url is None:
            raise ValueError("IQM_SERVER_URL environment variable not found. Please set it up.")

        self.backend = backend
        self.quantum_circuit = quantum_circuit
        self.cost_function = cost_function
        self.readoutmode = readoutmode
        self.cal_data = CalibrationDataManager().get_calibration_fidelities(self.backend)

        if layouts is not None:
            self.layouts = layouts
        else:
            self.layouts = LayoutGenerator(
                self.backend,
                self.quantum_circuit,
                remove_qubits=remove_qubits,
                num_trials=num_trials,
            ).generate_unique_layouts()

        self.transpiled_qcs = []
        optimization_level = 3  # By default is set to 3
        for layout in self.layouts:
            try:
                qc_transpiled = perform_backend_transpilation(
                    [self.quantum_circuit],
                    self.backend,
                    layout,
                    self.backend.coupling_map.reduce(mapping=layout),
                    qiskit_optim_level=optimization_level,
                    optimize_sqg=True,
                )
            except Exception:
                logger.exception("Failed to transpile a circuit %s", self.quantum_circuit.name)
                continue
            self.transpiled_qcs.append(qc_transpiled[0])

        self.circuit_compiler_data = qiskit_to_pulla(
            Pulla(
                self.iqm_url,
            ),
            self.backend,
            self.transpiled_qcs,
        )[0]

    def get_top_layouts(self, num_layouts: int = 1) -> tuple[list[Layout], list[float]]:
        """Compute the costs associated with different circuit layouts based on the specified cost function.

        Args:
            num_layouts: The number of top layouts to return based on their costs.

        Returns:
            A tuple containing two lists with layouts and scores.

        """
        logger.info('Cost evaluation has begun using cost function "%s".', self.cost_function)
        fidelities = []
        fidelities = self._get_gate_cost()
        errors = [1 - fid for fid in fidelities]
        final_costs = list(zip(self.layouts, errors, strict=True))

        # Sort the layouts by their respective costs, so that the first layout is the one with the lowest cost
        final_costs.sort(key=lambda x: x[1])

        # Limit the final costs to the top num_layouts
        top_costs = final_costs[:num_layouts]
        layouts = [layout for layout, _ in top_costs]
        scores = [score for _, score in top_costs]

        return layouts, scores

    def _prepare_calibration_data(self) -> dict[str, Any]:
        """Prepare calibration data based on cost function and readout mode.

        Returns:
            Dictionary containing calibration data for gates and readout.

        """
        calibration_data = {"prx": self.cal_data.get(CalibrationType.SQG)}
        if self.cost_function == CostFunction.GATE_COST_CZ:
            calibration_data.update(
                {
                    "cz": self.cal_data.get(CalibrationType.CZ),
                }
            )
        elif self.cost_function == CostFunction.GATE_COST_CLIFFORD:
            calibration_data.update(
                {
                    "cz": self.cal_data.get(CalibrationType.CLIFFORD),
                }
            )

        add_readout = self.readoutmode in {ReadoutMode.FIDELITY, ReadoutMode.QNDNESS}
        if add_readout:
            readout_data = {
                ReadoutMode.FIDELITY: {"measure": self.cal_data[CalibrationType.READOUT]},
                ReadoutMode.QNDNESS: {"measure": self.cal_data[CalibrationType.READOUT_QNDNESS]},
            }.get(self.readoutmode, {})
            calibration_data.update(readout_data)

        return calibration_data

    def _get_gate_cost(self) -> list[float]:
        """Evaluate the cost for each given layout based on the calibration data.

        Returns:
            List of gate fidelity for a given set of layouts.

        """
        fidelities = []
        calibration_data = self._prepare_calibration_data()
        add_readout = self.readoutmode in {ReadoutMode.FIDELITY, ReadoutMode.QNDNESS}

        for circuit in self.circuit_compiler_data:
            fid = 1.0
            for instruction in circuit.instructions:
                cal_value = None
                if instruction.name in calibration_data:
                    if len(instruction.locus) == CZ_LOCUS_LENGTH:
                        # 2-qubit gate operations
                        qubit_list = str(list(instruction.locus))
                    else:
                        # 1-qubit gate operations
                        qubit_list = str(instruction.locus[0])
                    cal_value = calibration_data[instruction.name].get(qubit_list)

                # For readout and reset operations
                if instruction.name in ["measure", "reset"] and add_readout:
                    qubit_list = str(instruction.locus[0])
                    cal_value = calibration_data[instruction.name].get(qubit_list, 1)

                if cal_value is not None:
                    if cal_value > 1:
                        fid = 0
                        break  # Early exit if fidelity is zero
                    fid *= cal_value
            fidelities.append(fid)

        return fidelities


class CalibrationDataManager:
    """Manage calibration data retrieval for the QPU."""

    def __init__(self) -> None:
        self.iqm_url = os.getenv("IQM_SERVER_URL")

        if self.iqm_url is None:
            raise ValueError("IQM_SERVER_URL environment variable not found. Please set it up.")

    def get_calibration_fidelities(self, backend: IQMBackendBase) -> dict[str, dict[str, float]]:
        """Fetch the latest calibration fidelity data for the specified hardware.

        Args:
            backend: The IQM backend instance.

        Returns:
            A dictionary with calibration data for TQG, SQG, readout fidelities, T1, and T2 times.

        """
        quality_metric_set = IQMClient(self.iqm_url).get_quality_metric_set()  # type: ignore[arg-type]
        calibration_metrics = quality_metric_set.observations
        return self._parse_calibration_metrics(calibration_metrics, backend)

    # Sort the dictionaries by keys
    @staticmethod
    def _qubit_sort_key(item: tuple[str, float]) -> int:
        """Extract numeric part from qubit key for natural sorting.

        Args:
        item: A tuple containing the qubit key and its value.

        Returns:
        The numeric part of the qubit key, or 0 if no match is found.

        """
        key = item[0]
        match = re.search(r"QB(\d+)", key)
        return int(match.group(1)) if match else 0

    def _extract_gates_info(self, backend: IQMBackendBase) -> dict[str, Any]:
        """Extract gate information from backend architecture.

        Args:
            backend: The IQM backend instance.

        Returns:
            Dictionary containing gate information.

        """
        gates_info = {}
        for gate in backend.architecture.gates.keys():
            gates_info[gate] = {
                x: backend.architecture.gates[gate].implementations[x].loci
                for x in backend.architecture.gates[gate].implementations.keys()
            }
        return gates_info

    def _create_dqa_keys(
        self, gates_info: dict[str, Any], backend: IQMBackendBase
    ) -> tuple[list[str], list[str], list[str]]:
        """Create DQA keys for CZ, CLIFFORD, and single-qubit operations.

        Args:
            gates_info: Dictionary containing gate information.
            backend: The IQM backend instance.

        Returns:
            Tuple of (dqa_cz_keys, dqa_clifford_keys, dqa_sq_keys).

        """
        cz_pairs = [
            pairs
            for x in gates_info[ObservationType.CZ.value].keys()
            for pairs in gates_info[ObservationType.CZ.value][x]
        ]
        dqa_cz_keys = [str(list(item)) for item in cz_pairs]
        symm_cz_keys = [
            f"[{pair.split(', ')[1].strip(']}')}, {pair.split(', ')[0].strip('[}')}]" for pair in dqa_cz_keys
        ]
        dqa_cz_keys.extend(symm_cz_keys)
        dqa_clifford_keys = dqa_cz_keys
        dqa_sq_keys = list(backend.architecture.components)
        return dqa_cz_keys, dqa_clifford_keys, dqa_sq_keys

    def _parse_calibration_metrics(  # noqa: C901
        self, calibration_metrics: list[Any], backend: IQMBackendBase
    ) -> dict[str, dict[str, float]]:
        """Process calibration metrics and return a dictionary with the corresponding metrics.

        Args:
            calibration_metrics: A list of dictionaries containing calibration metrics.
            backend: The IQM backend instance.

        Returns:
            A dictionary with the processed calibration metrics.

        """
        # Create a dictionary to map key names to their corresponding metrics
        cz_fidelity: dict[str, float] = {}
        clifford_fidelity: dict[str, float] = {}
        single_qubit_fidelity: dict[str, float] = {}
        readout_fidelity: dict[str, float] = {}
        readout_qndness: dict[str, float] = {}
        t1: dict[str, float] = {}
        t2: dict[str, float] = {}

        gates_info = self._extract_gates_info(backend)
        dqa_cz_keys, dqa_clifford_keys, dqa_sq_keys = self._create_dqa_keys(gates_info, backend)

        # Iterate over the calibration metrics
        for metrics in calibration_metrics:
            dut_field = metrics.dut_field
            values = metrics.value
            if all(obs in dut_field for obs in ObservationType.READOUT.value):
                qubit_name = "QB" + dut_field.split("QB")[1].split(".")[0]
                readout_fidelity[str(qubit_name)] = values
            elif all(obs in dut_field for obs in ObservationType.READOUT_QNDNESS.value):
                qubit_name = "QB" + dut_field.split("QB")[1].split(".")[0]
                readout_qndness[str(qubit_name)] = values
            elif ObservationType.SQG.value in dut_field and any(
                x in dut_field for x in gates_info[ObservationType.SQG.value]
            ):
                qubit_name = "QB" + dut_field.split("QB")[1].split(".")[0]
                single_qubit_fidelity[str(qubit_name)] = values
            elif all(obs in dut_field for obs in ObservationType.T1.value):
                qubit_name = "QB" + dut_field.split("QB")[1].split(".")[0]
                t1[str(qubit_name)] = values * 10**6
            elif all(obs in dut_field for obs in ObservationType.T2.value):
                qubit_name = "QB" + dut_field.split("QB")[1].split(".")[0]
                t2[str(qubit_name)] = values * 10**6
            elif ObservationType.CZ.value in dut_field and any(
                x in dut_field for x in gates_info[ObservationType.CZ.value]
            ):
                qb_matches = re.findall(r"QB\d+", dut_field)
                qbx, qby = qb_matches[0], qb_matches[1]
                cz_fidelity[str([qbx, qby])] = values
                cz_fidelity[str([qby, qbx])] = values
            elif all(obs in dut_field for obs in ObservationType.CLIFFORD.value):
                qb_matches = re.findall(r"QB\d+", dut_field)
                qbx, qby = qb_matches[0], qb_matches[1]
                clifford_fidelity[str([qbx, qby])] = values
                clifford_fidelity[str([qby, qbx])] = values

        # Sanity check by comparing with DQA pairs for CZ and CLIFFORD operations:
        cz_fidelity = {key: value for key, value in cz_fidelity.items() if key in dqa_cz_keys}
        clifford_fidelity = {key: value for key, value in clifford_fidelity.items() if key in dqa_clifford_keys}
        # Remove qubits from QM metrics  that are not present in the DQA.
        for metrics in [single_qubit_fidelity, readout_fidelity, readout_qndness, t1, t2]:
            metrics_keys = list(metrics.keys())
            sq_keys = [key for key in metrics_keys if key in dqa_sq_keys]
            for key in metrics_keys:
                if key not in sq_keys:
                    del metrics[key]

        fidelity_data = {
            CalibrationType.CZ.value: cz_fidelity,
            CalibrationType.CLIFFORD.value: clifford_fidelity,
            CalibrationType.SQG.value: dict(sorted(single_qubit_fidelity.items(), key=self._qubit_sort_key)),
            CalibrationType.READOUT.value: dict(sorted(readout_fidelity.items(), key=self._qubit_sort_key)),
            CalibrationType.READOUT_QNDNESS.value: dict(sorted(readout_qndness.items(), key=self._qubit_sort_key)),
            CalibrationType.T1.value: dict(sorted(t1.items(), key=self._qubit_sort_key)),
            CalibrationType.T2.value: dict(sorted(t2.items(), key=self._qubit_sort_key)),
        }

        return fidelity_data
