"""Test layering stages"""

from iqm.cpc.compiler.layering_stages import CircuitLayer, layer_circuits, set_layer_transformations
from iqm.cpc.core.config import ComponentGrouping
from iqm.pulse import Circuit, CircuitOperation


def test_layering_and_set_transformations():
    circuit = Circuit(
        name="test_circuit",
        instructions=tuple(
            [CircuitOperation("X", ("QB1",)), CircuitOperation("cz", ("QB2", "QB3")), CircuitOperation("Y", ("QB1",))]
        ),
    )
    # layer circuits
    layered_circuits = layer_circuits([circuit], ComponentGrouping(["QB1", "QB2", "QB3"]), skip_layering=False)
    assert len(layered_circuits) == 1
    assert len(layered_circuits[0]) == 2  # 2 layers
    assert isinstance(layered_circuits[0][0], CircuitLayer)
    assert layered_circuits[0][0].layer_type == "flux"
    assert layered_circuits[0][0].locus_components == {"QB2", "QB3"}
    assert {inst.name for inst in layered_circuits[0][0].instructions} == {"cz"}
    assert isinstance(layered_circuits[0][1], CircuitLayer)
    assert layered_circuits[0][1].layer_type == "non-flux"
    assert layered_circuits[0][1].locus_components == {"QB1"}
    assert {inst.name for inst in layered_circuits[0][1].instructions} == {"X", "Y"}
    # test set transformations
    set_layer_transformations(
        circuits=layered_circuits,
        flux_layer_transformations=["horse_transformation"],
        non_flux_layer_transformations=["duck_transformation", "goose_transformation"],
    )
    assert layered_circuits[0][0].transformations == ["horse_transformation"]
    assert layered_circuits[0][1].transformations == ["duck_transformation", "goose_transformation"]
    # test trivial layering
    layered_circuits = layer_circuits([circuit], ComponentGrouping(["QB1", "QB2", "QB3"]), skip_layering=True)
    assert len(layered_circuits[0]) == 1
    assert isinstance(layered_circuits[0][0], CircuitLayer)
    assert layered_circuits[0][0].layer_type == "full_circuit"
    assert layered_circuits[0][0].locus_components == {"QB1", "QB2", "QB3"}
    assert {inst.name for inst in layered_circuits[0][0].instructions} == {"X", "Y", "cz"}
