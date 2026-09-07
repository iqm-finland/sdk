# Copyright 2019-2025 IQM
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
"""Helper to generate python sources from protobuf definitions."""

import os
from pathlib import Path
import shutil
import subprocess

if __name__ == "__main__":
    PROTO_DIR = "protos"
    NAMESPACE_ROOT = "iqm"
    PYTHON_OUT = "src"

    python_out_path = Path(PYTHON_OUT) / NAMESPACE_ROOT / "data_definitions"
    if python_out_path.exists():
        print(f"Removing directory '{python_out_path}' for generated sources")
        shutil.rmtree(python_out_path)

    print(f"Creating directory '{python_out_path}' for generated sources")
    os.makedirs(python_out_path)

    (python_out_path / "__init__.py").touch(exist_ok=True)
    (python_out_path / "py.typed").touch(exist_ok=True)
    descriptor_set_out = python_out_path / "descriptor_set.bin"
    proto_files_paths = [str(path) for path in Path(PROTO_DIR).rglob("*.proto")]
    cmd = [
        "protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={PYTHON_OUT}",
        f"--mypy_out={PYTHON_OUT}",
        f"--descriptor_set_out={descriptor_set_out}",
        "--include_imports",
        # "--include_source_info",
    ] + proto_files_paths

    print(f"Invoking protoc compiler in a subprocess, command line: '{subprocess.list2cmdline(cmd)}'")
    subprocess.check_output(cmd)

    for path in (python_out_path).rglob("*/**"):
        if path.is_dir():
            init_py_path = path / "__init__.py"
            print(f"Touching {str(init_py_path)} for generated sources")
            init_py_path.touch()
