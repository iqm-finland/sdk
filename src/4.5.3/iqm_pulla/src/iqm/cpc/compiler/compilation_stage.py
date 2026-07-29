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
"""Compilation stages and passes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import copy, deepcopy
import functools
import inspect
from typing import Any, SupportsIndex

from iqm.cpc.compiler.errors import ClientError

DEFAULT_CONTEXT_KEYS = [
    "dut_label",
    "components",
    "timebox_input",
    "builder",
    "settings",
    "component_mapping",
    "chip_topology",
    "circuit_metrics",
    "mapped_readout_keys",
    "readout_metrics",
    "software_version_set_id",
    "soft_sweeps",
    "hard_sweeps",
]
"""Keys that always exist in the Compiler context. These keys can exist as compiler pass args. Any other contents of
the context must be referenced through the context (i.e. having `context` as a pass function argument)."""

DEFAULT_STAGE_ARGS = [*DEFAULT_CONTEXT_KEYS, "data", "context", "options"]
"""Default compiler pass function arguments. Any other pass function arg/kwarg names will be interpreted as
pass function options (i.e. compiler pass settings)."""

PassFunction = Callable
"""A function that takes the data, context, and options as arguments and returns the modified data and context.
The context is a dictionary that can contain any information that needs to be passed between the passes."""

PullaInputType = Any  # TODO: is there a way to type the input data correctly?


def pass_function_idempotent(function: PassFunction) -> PassFunction:
    """Wrap a pass function to make it idempotent."""

    @functools.wraps(function)
    def pass_with_idempotency(
        data_: Any, context_: dict[str, Any], options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        data = deepcopy(data_)
        context = deepcopy(context_)
        return function(data, context, options)

    pass_with_idempotency.__name__ = function.__name__
    return pass_with_idempotency


def compiler_pass(function: Callable) -> PassFunction:
    """Convenience wrapper to create a valid compiler pass.

    When the wrapped function is called, the compilation data (e.g. circuits) is passed as the first argument.
    If ``function`` has any other arguments, they are either sweepable stage options or default members of the
    stateful compiler context. If a compiler pass needs as its input data that is not a sweepable compiler option
    or one of the default context members, this data must be provided into the context by some other pass function
    or manually by the user, and the pass function should have ``context`` as one of its arguments. The final wrapped
    compiler then have uniform signatures: ``(data_, context, options)``.

    Args:
        function: The function to be wrapped into a valid compiler pass. It must return the ``data`` for the subsequent
            passes and any stateful side effects can be achieved via manipulation the compiler context.

    Returns:
          ``function`` wrapped into a compiler pass, i.e. the same functionality as originally but with the signature
            ``(data_, context, options)``.

    """
    sig = inspect.signature(function)
    if not sig.parameters:
        raise ValueError(f"Callable {function} wrapped with 'compiler_pass' should have at least one input argument.")

    @functools.wraps(function)
    def pass_with_converted_args(data_: Any, context: dict[str, Any], options: dict[str, Any]) -> Any:
        kwargs = {}
        for required_key in [key for key, param in sig.parameters.items()][1:]:
            if required_key == "context":
                kwargs[required_key] = context
            elif required_key in DEFAULT_STAGE_ARGS:
                kwargs[required_key] = context[required_key]
            else:
                kwargs[required_key] = options[required_key]
        return function(data_, **kwargs), context

    pass_with_converted_args.__name__ = function.__name__
    return pass_with_converted_args


def resolve_circuit_function_and_stages(
    stages: list[CompilationStage],
    input: PullaInputType,
) -> list[CompilationStage]:
    """Insert the circuit generation as the first stage in the provided ``stages``.

    Args:
        stages: The stages into which the circuit should be inserted.
        input: The circuit generation function or a static circuit object. If a static circuit object is inputted, it
            will be wrapped into a trivial circuit generation function.

    Returns:
        The compilation stages such that the first stage is now the circuit generation.

    """
    stages = stages.copy()
    if not isinstance(input, Callable):  # type: ignore[arg-type]

        def circuit(data: Any) -> Any:
            return deepcopy(input)
    else:
        sig = inspect.signature(input)
        args = list(sig.parameters.keys())
        if "data" not in args:

            def circuit(data: Any, *args, **kwargs) -> Any:  # type:ignore[misc]
                return input(*args, **kwargs)

            tmp_params = list(inspect.signature(circuit).parameters.values())
            new_sig = sig.replace(parameters=(tmp_params[0],) + tuple(sig.parameters.values()))
            circuit.__signature__ = new_sig  # type:ignore[attr-defined]
        else:
            circuit = copy(input)
            circuit.__name__ = "circuit"

    generate_circuit_stage = CompilationStage(
        name="circuit_generation", info="Programmatic generation of quantum circuits or timeboxes."
    )
    generate_circuit_stage.add_passes(circuit)
    stages.insert(0, generate_circuit_stage)
    return stages


def format_stages(stages: list[CompilationStage], idempotent: bool = True) -> list[CompilationStage]:
    """Format user inputted stages and their passes to their final runnable form (see :func:`.compiler_pass`).

    Args:
        stages: The compilation stages containing function passes that should be formatted.
        idempotent: Optionally make the stages also idempotent (they deepcopy both the context and the circuits at
            each pass). NOTE: if performance is prioritized, this option should not be turned on.

    """
    stages = deepcopy(stages)
    for stage in stages:
        if idempotent:
            stage.passes = [pass_function_idempotent(compiler_pass(f)) for f in stage.passes]
        else:
            stage.passes = [compiler_pass(f) for f in stage.passes]
    return stages


class CompilationStage:
    """Sequence of compiler passes that are applied to the data.

    The data and context are returned after all passes have been applied.
    A pass is a function that takes the data, context and pass options as arguments and
    returns the modified data. The context is a dictionary that can contain any information that needs to be
    passed between the passes and accumulates stateful side effects.
    """

    def __init__(self, name: str, info: str = "") -> None:
        self.name: str = name
        self.passes: list[PassFunction] = []
        self.info: str = info

    def add_passes(self, *passes: PassFunction, index: int | None = None) -> None:
        """Add multiple passes to the stage.

        Args:
            passes: One or more passes to be added to the stage.
            prepend: If ``True``, prepend the passes to the stage instead of appending them.

        """
        for added_passes, pas in enumerate(passes):
            _index = len(self.passes) if index is None else index + added_passes
            self.passes.insert(_index, pas)

    def get_pass_names(self) -> list[str]:
        """Get pass function names"""
        return [f.__name__ for f in self.passes]

    def get_pass_args(self) -> list[str]:
        """Get pass function argument names."""
        args = []
        for pas in self.passes:
            sig = list(inspect.signature(pas).parameters.keys())[1:]  # 1st arg name (the circuits) is arbitrary
            args.extend([n for n in sig if n not in args])
        return args

    def run(
        self, data: Any, context: dict[str, Any], options: dict[str, dict[str, dict[str, Any]]]
    ) -> tuple[Any, dict[str, Any]]:
        """Run all the passes in the stage on the data and context. The data and context are returned after all
        passes have been applied.

        Args:
            data: The data to be processed.
            context: A dictionary containing any additional information that needs to be passed between the passes.
            options: A dictionary of stage's pass function arguments and their values (mapped to the pass function name)

        Returns:
            The processed data and context.

        """
        stage_options = options.get(self.name, {})
        common_args = {k: v for k, v in options.items() if not isinstance(v, dict)}
        for pass_function in self.passes:
            try:
                pass_options = stage_options.get(pass_function.__name__, {}) | common_args
                data, context = pass_function(data, context, pass_options)
            except Exception as exc:
                error_msg = f'Error in stage "{self.name}" pass "{pass_function.__name__}": {exc}'
                if isinstance(exc, ClientError):
                    raise type(exc)(error_msg) from exc
                raise RuntimeError(error_msg) from exc

        return data, context

    @staticmethod
    def get_pass_metadata(pass_fn: Callable) -> dict[str, Any]:
        """Returns all signature metadata in a single call."""
        try:
            sig = inspect.signature(pass_fn)
            arg_list = [str(param).strip() for param in sig.parameters.values()]

            return_annotation = sig.return_annotation
            if return_annotation is inspect.Signature.empty or return_annotation is None:
                return_type = "Any"
            else:
                return_type = str(return_annotation)

                # If "return_annotation" is a standard class, str() might return "<class 'list'>"
                # This logic strips those brackets and quotes.
                if return_type.startswith("<class '") and return_type.endswith("'>"):
                    return_type = return_type[8:-2]

                # Clean up verbose library paths for the UI
                return_type = return_type.replace("collections.abc.", "").replace("typing.", "")

            return {
                "arg_list": arg_list,
                "return_type": return_type,
                "doc": inspect.cleandoc(pass_fn.__doc__) if pass_fn.__doc__ else "No docstring available.",
            }
        except Exception:
            return {"arg_list": [], "return_type": "Any", "doc": "Error parsing metadata."}

    def __repr__(self) -> str:
        return f"CompilationStage(name='{self.name}')"

    def __str__(self) -> str:
        return f"CompilationStage(name='{self.name}')"


class StagesList(list[CompilationStage]):
    """Extend ``list[CompilationStage]`` to allow accessing the stages via their names and to include validation."""

    def __init__(self, stages: list[CompilationStage]) -> None:
        _validate_stages([], stages)
        super().__init__(stages)

    def __getattribute__(self, item: str) -> Any:
        for stage in self:
            if stage.name == item:
                return stage
        return super().__getattribute__(item)

    def append(self, stage: CompilationStage) -> None:
        _validate_stages(self, [stage])
        super().append(stage)

    def insert(self, index: SupportsIndex, stage: CompilationStage) -> None:
        _validate_stages(self, [stage])
        super().insert(index, stage)

    def extend(self, stages: Iterable[CompilationStage]) -> None:
        _validate_stages(self, stages)
        super().extend(stages)


def _validate_stages(previous_stages: Iterable[CompilationStage], new_stages: Iterable[CompilationStage]) -> None:
    """Validate that stages contain no duplicate names and are of the type CompilationStage."""
    names = []
    for stage in new_stages:
        if not isinstance(stage, CompilationStage):
            raise ValueError("Only objects of type CompilationStage are allowed")
        if stage.name in names:
            raise ValueError(f"Duplicate stage name {stage.name}")
        names.append(stage.name)
    for already_existing_stage in previous_stages:
        if already_existing_stage.name in names:
            raise ValueError(f"Duplicate stage name {already_existing_stage.name}")
