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

from iqm.cpc.compiler.compiler import STANDARD_CIRCUIT_EXECUTION_OPTIONS, pass_function_idempotent
from iqm.cpc.compiler.standard_stages import get_standard_stages
from iqm.pulla.utils import calset_to_cal_data_tree


def test_standard_compiler_context(pulla_on_spark):
    """
    Test that the standard compiler is initialized with the correct context.
    """
    compiler = pulla_on_spark.get_standard_compiler()
    compiler_context = compiler.compiler_context()

    default_calibration_set, _ = pulla_on_spark.fetch_default_calibration_set()
    assert compiler_context["calibration_set"] == default_calibration_set
    assert compiler_context["calibration_set"] == compiler._calibration_set_values
    assert compiler_context["component_mapping"] == compiler.component_mapping
    assert compiler_context["options"] == STANDARD_CIRCUIT_EXECUTION_OPTIONS
    assert compiler_context["builder"] == compiler.builder
    assert compiler_context["options"] == compiler.options


def test_standard_compiler_custom_calibration_set(pulla_on_spark):
    """
    Test that the standard compiler can be initialized with a custom calibration set.
    """
    custom_cal_set = pulla_on_spark.fetch_default_calibration_set()[0]
    custom_cal_set["custom_key"] = "custom_value"
    compiler = pulla_on_spark.get_standard_compiler(calibration_set_values=custom_cal_set)
    compiler_context = compiler.compiler_context()

    assert compiler_context["calibration_set"]["custom_key"] == "custom_value"


def test_standard_compiler_has_standard_stages(pulla_on_spark):
    """
    Test that the standard compiler is initialized by default with standard stages.
    """
    compiler = pulla_on_spark.get_standard_compiler()
    # we can't compare stages directly because get_standard_stages() returns a copy, so let's just compare names
    for couple in list(zip(compiler.stages, get_standard_stages())):
        assert couple[0].name == couple[1].name


def test_standard_compiler_builder_consistent_with_calibration_set(pulla_on_spark):
    """
    Test that the calibration set in the compiler context is consistent with the calibration set in the builder.
    """
    compiler = pulla_on_spark.get_standard_compiler()
    compiler_context = compiler.compiler_context()
    assert compiler_context["builder"].calibration == calset_to_cal_data_tree(compiler_context["calibration_set"])


def test_pass_function_idempotent():
    """
    Test that the pass_function_idempotent wrapper works correctly.
    """

    class Object:
        pass

    def dummy(data, dictionary):
        data.prop = "new_prop"
        dictionary["k"] = "new_value"
        return data, dictionary

    obj = Object()
    obj.prop = "old_prop"
    dictionary = {"k": "old_value"}

    idempotent_dummy = pass_function_idempotent(dummy)
    new_obj, new_dictionary = idempotent_dummy(obj, dictionary)

    # mutable objects should not be mutated by the function call
    assert obj != new_obj
    assert obj.prop != new_obj.prop
    assert dictionary != new_dictionary
    assert dictionary["k"] != new_dictionary["k"]
