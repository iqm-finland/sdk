#  ********************************************************************************
#  Copyright (c) 2019-2024 IQM Finland Oy.
#  All rights reserved. Confidential and proprietary.
#
#  Distribution or reproduction of any information contained herein
#  is prohibited without IQM Finland Oy’s prior written permission.
#  ********************************************************************************
import pytest

from exa.common.data.parameter import CollectionType, DataType, Parameter


@pytest.fixture
def parameter():
    return Parameter("frequency", "Frequency", "Hz")


def test_model_copy(parameter):
    model_copy = parameter.model_copy()

    assert model_copy is parameter  # Safe for immutable models
    assert parameter == model_copy
    assert model_copy.name == parameter.name
    assert model_copy.label == parameter.label
    assert model_copy.unit == parameter.unit
    assert model_copy.data_type == parameter.data_type
    assert model_copy.collection_type == parameter.collection_type
    assert model_copy.element_indices == parameter.element_indices


def test_model_copy_update(parameter):
    assert parameter.name == "frequency"
    assert parameter.label == "Frequency"
    assert parameter.unit == "Hz"
    assert parameter.data_type == DataType.FLOAT
    assert parameter.collection_type == CollectionType.SCALAR
    assert parameter.element_indices is None

    model_copy = parameter.model_copy(
        update={
            "name": "modified_name",
            "label": "Modified Label",
            "unit": "Modified Unit",
            "data_type": DataType.STRING,
            "collection_type": CollectionType.LIST,
            "element_indices": [0],
        }
    )

    assert model_copy is not parameter
    assert parameter != model_copy
    assert model_copy.name == "modified_name"
    assert model_copy.label == "Modified Label"
    assert model_copy.unit == "Modified Unit"
    assert model_copy.data_type == DataType.STRING
    assert model_copy.collection_type == CollectionType.LIST
    assert model_copy.element_indices == [0]


def test_copy_works_but_is_deprecated(parameter):
    with pytest.warns(DeprecationWarning, match="`copy` method is deprecated since 2025-03-28"):
        model_copy = parameter.copy()

    assert model_copy is not parameter
    assert parameter == model_copy
    assert model_copy.name == parameter.name
    assert model_copy.label == parameter.label
    assert model_copy.unit == parameter.unit
    assert model_copy.data_type == parameter.data_type
    assert model_copy.collection_type == parameter.collection_type
    assert model_copy.element_indices == parameter.element_indices


def test_copy_update_works_but_is_deprecated(parameter):
    assert parameter.name == "frequency"
    assert parameter.label == "Frequency"
    assert parameter.unit == "Hz"
    assert parameter.data_type == DataType.FLOAT
    assert parameter.collection_type == CollectionType.SCALAR
    assert parameter.element_indices is None

    with pytest.warns(DeprecationWarning, match="`copy` method is deprecated since 2025-03-28"):
        model_copy = parameter.copy(
            name="modified_name",
            label="Modified Label",
            unit="Modified Unit",
            data_type=DataType.STRING,
            collection_type=CollectionType.LIST,
            element_indices=[0],
        )

    assert model_copy is not parameter
    assert parameter != model_copy
    assert model_copy.name == "modified_name"
    assert model_copy.label == "Modified Label"
    assert model_copy.unit == "Modified Unit"
    assert model_copy.data_type == DataType.STRING
    assert model_copy.collection_type == CollectionType.LIST
    assert model_copy.element_indices == [0]
