#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pytest

from exa.common.qcm_data.chip_topology import (
    DEFAULT_1QB_MAPPING,
    DEFAULT_2QB_MAPPING,
    ChipTopology,
    sort_components,
    sort_couplers,
)


@pytest.mark.parametrize(
    "components, expected",
    [
        (["COMPR1", "QB1"], ["QB1", "COMPR1"]),
        (["COMP_R", "QB1"], ["QB1", "COMP_R"]),
        (["QB2", "COMPR1", "QB1"], ["QB1", "QB2", "COMPR1"]),
        (["QB2", "COMP_R", "QB1"], ["QB1", "QB2", "COMP_R"]),
        (["COMPR2", "QB2", "COMP_R", "QB1"], ["QB1", "QB2", "COMP_R", "COMPR2"]),
    ],
)
def test_sort_components(components, expected):
    assert sort_components(components) == expected


@pytest.mark.parametrize(
    "couplers, expected",
    [
        (["TC2", "TC1"], ["TC1", "TC2"]),
        (["TC-2-R1", "TC-1-R1"], ["TC-1-R1", "TC-2-R1"]),
        (["TC-3-4", "TC-2-4", "TC-1-4"], ["TC-1-4", "TC-2-4", "TC-3-4"]),
        (["TC-1-4", "TC-1-3", "TC-1-2"], ["TC-1-2", "TC-1-3", "TC-1-4"]),
        (["TC-3-4", "TC-2-5"], ["TC-2-5", "TC-3-4"]),
    ],
)
def test_sort_couplers(couplers, expected):
    assert sort_couplers(couplers) == expected


def test_from_chad(cdr, chad):
    topo_chad = ChipTopology.from_chad(chad)
    # similar CHAD constructed manually
    topo = ChipTopology(
        qubits=["QB2", "QB0", "QB1", "QB3"],
        computational_resonators=["COMP_R"],
        couplers={
            "TC1": ["QB1", "COMP_R"],
            "TC3": ["QB3", "COMP_R"],
            "TC2": ["QB2", "COMP_R"],
        },
        probe_lines={
            "PL": ["QB1", "QB2", "QB0", "QB3"],
        },
    )
    assert topo.qubits == topo_chad.qubits
    assert topo.qubits_sorted == topo_chad.qubits_sorted
    assert topo.computational_resonators == topo_chad.computational_resonators
    assert topo.computational_resonators_sorted == topo_chad.computational_resonators_sorted


def test_init(cdr):
    topo_cdr = ChipTopology.from_chip_design_record(cdr)
    # similar CHAD constructed manually
    topo = ChipTopology(
        qubits=["QB2", "QB0", "QB1", "QB3"],
        computational_resonators=["COMP_R"],
        couplers={
            "TC1": ["QB1", "COMP_R"],
            "TC3": ["QB3", "COMP_R"],
            "TC2": ["QB2", "COMP_R"],
        },
        probe_lines={
            "PL": ["QB1", "QB2", "QB0", "QB3"],
        },
    )
    assert topo.qubits == topo_cdr.qubits
    assert topo.qubits_sorted == topo_cdr.qubits_sorted
    assert topo.computational_resonators == topo_cdr.computational_resonators
    assert topo.computational_resonators_sorted == topo_cdr.computational_resonators_sorted
    assert topo.couplers == topo_cdr.couplers
    assert topo.couplers_sorted == topo_cdr.couplers_sorted
    assert topo.probe_lines == topo_cdr.probe_lines
    assert topo.probe_lines_sorted == topo_cdr.probe_lines_sorted
    assert topo.all_components == topo_cdr.all_components

    assert topo.coupler_to_components == topo_cdr.coupler_to_components
    assert topo.component_to_couplers == topo_cdr.component_to_couplers
    assert topo.probe_line_to_components == topo_cdr.probe_line_to_components
    assert topo.component_to_probe_line == topo_cdr.component_to_probe_line

    assert topo._locus_mappings == topo_cdr._locus_mappings


def test_is_qubit(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert topo.is_qubit("QB1")
    assert not topo.is_qubit("TC1")
    assert not topo.is_qubit("COMP_R")
    assert not topo.is_qubit("PL")


def test_is_coupler(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert not topo.is_coupler("QB1")
    assert topo.is_coupler("TC1")
    assert not topo.is_coupler("COMP_R")
    assert not topo.is_coupler("PL")


def test_is_probe_line(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert not topo.is_probe_line("QB1")
    assert not topo.is_probe_line("TC1")
    assert not topo.is_probe_line("COMP_R")
    assert topo.is_probe_line("PL")


def test_is_computational_resonator(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert not topo.is_computational_resonator("QB1")
    assert not topo.is_computational_resonator("TC1")
    assert topo.is_computational_resonator("COMP_R")
    assert not topo.is_computational_resonator("PL")


def test_get_connecting_couplers(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert topo.get_connecting_couplers(["COMP_R", "QB3"]) == {"TC3"}
    assert topo.get_connecting_couplers(["QB1", "COMP_R"]) == {"TC1"}
    assert topo.get_connecting_couplers(["QB1", "QB0"]) == set()


def test_components_to_probelines(cheddar_data_1_1_fake):
    topo = ChipTopology.from_chip_design_record(cheddar_data_1_1_fake["data"])
    assert topo.component_to_probe_line["QB1"] == "PL_RO-1"
    assert topo.component_to_probe_line["QB2"] == "PL_RO-1"
    assert topo.component_to_probe_line["TC-2-5"] == "PL_RO-1"


def test_get_connecting_couplers_no_dangling_couplers():
    topo = ChipTopology(
        qubits=["A", "B", "C"],
        computational_resonators=[],
        couplers={
            "TC1": ["A", "B"],
            "TC2": ["B", "C"],
            "TC3": ["C"],  # does not connect anything
        },
        probe_lines={},
    )
    assert topo.get_connecting_couplers(["C", "B"]) == {"TC2"}


def test_get_coupler_for():
    topo = ChipTopology(
        qubits=["A", "B", "C"],
        computational_resonators=[],
        couplers={
            "TC1": ["A", "B"],
            "TC2": ["B", "C"],
            "TC3": ["B", "C"],
        },
        probe_lines={},
    )
    assert topo.get_coupler_for("A", "B") == "TC1"
    with pytest.raises(ValueError, match="have 0 connecting couplers"):
        topo.get_coupler_for("A", "C")
    with pytest.raises(ValueError, match="have 2 connecting couplers"):
        topo.get_coupler_for("B", "C")


def test_get_neighbor_couplers(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert topo.get_neighbor_couplers(["QB1", "QB2"]) == {"TC1", "TC2"}
    assert topo.get_neighbor_couplers(["QB1"]) == {"TC1"}
    assert topo.get_neighbor_couplers(["COMP_R"]) == {"TC1", "TC2", "TC3"}


def test_get_neighbor_couplers_uncoupled_component():
    topo = ChipTopology(
        qubits=["A", "B", "C", "X"],
        computational_resonators=[],
        couplers={
            "TC1": ["A", "B"],
            "TC2": ["B", "C"],
        },
        probe_lines={},
    )
    assert topo.get_neighbor_couplers(["X"]) == set()
    assert topo.get_neighbor_couplers(["A", "X"]) == {"TC1"}


def test_get_neighbor_locus_components(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    assert topo.get_neighbor_locus_components(["QB1", "QB2"]) == {"COMP_R"}
    assert topo.get_neighbor_locus_components(["QB1", "QB3", "COMP_R"]) == {"QB2"}


def test_set_locus_mapping(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    mapping = {frozenset(["QB3"]): ("TC1",)}
    topo.set_locus_mapping("stupid_gate", mapping)
    assert topo._locus_mappings["stupid_gate"] == mapping

    mapping = {("QB1", "QB2", "QB3"): ("TC1", "TC2")}
    topo.set_locus_mapping("my_3qb_gate", mapping)
    assert topo._locus_mappings["my_3qb_gate"] == {("QB1", "QB2", "QB3"): ("TC1", "TC2")}

    with pytest.raises(ValueError, match="Mapped loci need"):
        topo.set_locus_mapping("bad_mapping", {"QB1": ("TC1",)})

    with pytest.raises(ValueError, match="Mapped component"):
        topo.set_locus_mapping("bad_mapping", {frozenset(["QB1", "QB2"]): ("TC-101-3001",)})

    with pytest.raises(ValueError, match="Locus component"):
        topo.set_locus_mapping("bad_mapping", {("QB1", "QB666"): ("TC1",)})


def test_map_locus(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    mapping = {("QB1", "QB2", "QB3"): ("TC1", "TC2")}
    topo.set_locus_mapping("my_3qb_gate", mapping)

    # test default qubit-coupler mapping:
    assert topo.map_locus(("QB2",)) == ("QB2",)
    assert topo.map_locus(frozenset(["QB1", "QB3"])) is None

    # test added custom mapping:
    assert topo.map_locus(("QB1", "QB2", "QB3"), "my_3qb_gate") == ("TC1", "TC2")

    # test non-existent mapping:
    assert topo.map_locus({"QB1", "QB3"}, "horse_mapping") is None


def test_map_to_locus(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    mapping = {("QB1", "QB2", "QB3"): ("TC1", "TC2")}
    topo.set_locus_mapping("my_3qb_gate", mapping)

    # test default mapping
    assert topo.map_to_locus("TC3", DEFAULT_2QB_MAPPING) == None  # noqa: E711

    # test added custom mapping:
    assert topo.map_to_locus(("TC1", "TC2"), "my_3qb_gate") == ("QB1", "QB2", "QB3")

    # test non-existent mapping:
    assert topo.map_to_locus("TC3", "horse_mapping") is None


def get_loci(cdr):
    topo = ChipTopology.from_chip_design_record(cdr)
    mapping = {("QB1", "QB2", "QB3"): ("TC1", "TC2")}
    topo.set_locus_mapping("my_3qb_gate", mapping)

    loci = topo.get_loci(DEFAULT_2QB_MAPPING)
    assert len(loci) == 2
    assert {"QB2", "QB3"} in loci
    assert {"QB1", "COMP_R"} in loci

    loci = topo.get_loci("my_3qb_gate")
    assert len(loci) == 1
    assert ("QB1", "QB2", "QB3") in loci

    loci = topo.get_loci("horse_gate", default_mapping_dimension=1)
    assert loci == topo.get_loci(DEFAULT_1QB_MAPPING)

    loci = topo.get_loci("horse_gate", default_mapping_dimension=2)
    assert loci == topo.get_loci(DEFAULT_2QB_MAPPING)

    loci = topo.get_loci("horse_gate", default_mapping_dimension=3)
    assert loci == []


def test_get_common_computational_resonator():
    topo = ChipTopology(
        qubits=["QB2", "QB0", "QB1", "QB3"],
        computational_resonators=["COMPR1", "COMPR2"],
        couplers={
            "TC1": ["QB1", "COMPR1"],
            "TC3": ["QB3", "COMPR1"],
            "TC2": ["QB2", "COMPR2"],
            "TC4": ["QB1", "COMPR2"],
            "TC5": ["QB3", "COMPR2"],
        },
        probe_lines={
            "PL": ["QB1", "QB2", "QB0", "QB3"],
        },
    )

    assert topo.get_common_computational_resonator("QB1", "QB2") == "COMPR2"
    assert topo.get_common_computational_resonator("QB1", "QB3") == "COMPR1"
    assert topo.get_common_computational_resonator("QB2", "QB3") == "COMPR2"
    assert topo.coupler_to_components == {
        "TC1": ("QB1", "COMPR1"),
        "TC3": ("QB3", "COMPR1"),
        "TC2": ("QB2", "COMPR2"),
        "TC4": ("QB1", "COMPR2"),
        "TC5": ("QB3", "COMPR2"),
    }
    with pytest.raises(
        ValueError,
        match="No computational resonator was found, that is connected to both qubits QB0 and QB3 via tunable"
        " couplers.",
    ):
        topo.get_common_computational_resonator("QB0", "QB3")
