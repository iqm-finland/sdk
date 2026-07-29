"""Test CompilationStage."""

from typing import Any

import pytest

from iqm.cpc.compiler.compilation_stage import CompilationStage, StagesList, compiler_pass


@compiler_pass
def _stage0_pass0(_data: Any, context: dict[str, Any], horse: int) -> Any:
    context["horse"] = horse
    _data.append(horse)
    return _data


@compiler_pass
def _stage0_pass1(_data: Any, context: dict[str, Any], duck: int) -> Any:
    context["duck"] = duck
    _data.append(duck)
    return _data


@compiler_pass
def _stage1_pass0(_data: Any, context: dict[str, Any], goose: int, cat: int) -> Any:
    context["goose"] = goose
    context["cat"] = cat
    _data.extend([goose, cat])
    return _data


def test_add_passes():
    stage = CompilationStage(name="stage0")
    stage.add_passes(_stage0_pass0, _stage0_pass1)
    assert stage.passes == [_stage0_pass0, _stage0_pass1]
    stage.add_passes(_stage1_pass0, index=1)
    assert stage.passes == [_stage0_pass0, _stage1_pass0, _stage0_pass1]


def test_run_stages():
    stage0 = CompilationStage(name="stage0")
    stage0.add_passes(_stage0_pass0, _stage0_pass1)
    stage1 = CompilationStage(name="stage1")
    stage1.add_passes(_stage1_pass0)
    context = {}
    options = {
        "stage0": {"_stage0_pass0": {"horse": 1}, "_stage0_pass1": {"duck": 2}},
        "stage1": {"_stage1_pass0": {"goose": 3, "cat": 4}},
    }
    data = []
    data, context = stage0.run(data, context, options)
    data, context = stage1.run(data, context, options)
    assert data == [1, 2, 3, 4]
    assert context == {
        "horse": 1,
        "duck": 2,
        "goose": 3,
        "cat": 4,
    }


def test_stages_list():
    stage0 = CompilationStage(name="stage0")
    stage0.add_passes(_stage0_pass0)
    stage1 = CompilationStage(name="stage1")
    stage1.add_passes(_stage1_pass0)
    stage2 = CompilationStage(name="stage2")
    stage2.add_passes(_stage1_pass0)
    stage3 = CompilationStage(name="stage3")
    stage3.add_passes(_stage1_pass0)
    stage4 = CompilationStage(name="stage4")
    stage4.add_passes(_stage1_pass0)
    stage5 = CompilationStage(name="stage5")
    stage5.add_passes(_stage1_pass0)

    stages = StagesList([stage0])

    # test usual list syntax works
    stages.append(stage1)
    assert len(stages) == 2
    stages.insert(1, stage2)
    assert stages[1].name == "stage2"
    stages.extend([stage3, stage4])
    assert len(stages) == 5
    assert stages[3].name == "stage3"

    # test __getattr__
    assert isinstance(stages.stage0, CompilationStage)
    assert stages.stage0.name == "stage0"
    assert isinstance(stages.stage4, CompilationStage)
    assert stages.stage4.name == "stage4"
    with pytest.raises(AttributeError, match="'StagesList' object has no attribute 'stage6'"):
        stages.stage6

    # test validation errors
    with pytest.raises(ValueError, match="Duplicate stage name stage1"):
        stages.append(stage1)
    with pytest.raises(ValueError, match="Duplicate stage name stage2"):
        stages.insert(3, stage2)
    with pytest.raises(ValueError, match="Only objects of type CompilationStage are allowed"):
        stages.append("this is not a CompilationStage")
    with pytest.raises(ValueError, match="Duplicate stage name stage5"):
        stages.extend([stage5, stage5])
    with pytest.raises(ValueError, match="Only objects of type CompilationStage are allowed"):
        StagesList(["this is not a CompilationStage", stage0])
