# iqm-data-definitions

A common place for data definitions shared inside IQM. The package is meant to
stay independent of any other IQM project, so that everything in the
control software stack can depend on it. Concretely it contains:

- The Protobuf message formats (`protos/`) used by the quantum computer control
  software stack, and the serialization code generated from them
  (`src/iqm/data_definitions/`).
- Hand-written Python-native representations of some of those messages
  (`src/iqm/models/`).

## Protocol versioning

Breaking changes to data definitions require a major version update. Major
versions are described in the directory paths inside `protos/`. For instance,
version 1.x Protobuf definitions live in
`protos/iqm/data_definitions/<subpackage>/v1/*.proto`, and there can be several
subpackages per version.

Backwards-compatible changes are handled as minor version upgrades. Minor
versions are not part of the path, only of the package version: version 1.2
Protobuf definitions live in the same place as version 1.1 ones, but the
distributable packages differ. The generated Python code is importable as
`iqm.data_definitions.<subpackage>.v1.*_pb2`.
