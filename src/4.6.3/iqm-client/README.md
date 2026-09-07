# IQM Client

Client-side Python library for connecting to an [IQM](https://iqm.tech/)
quantum computer.

Includes as an optional feature [Qiskit](https://qiskit.org/) and
[Cirq](https://quantumai.google/cirq)
adapters for IQM's quantum computers, which allow you to:

- Transpile arbitrary quantum circuits for IQM quantum architectures
- Simulate execution on IQM quantum architectures with IQM-specific noise models
  (currently only the Qiskit adapter contains IQM noise models)
- Run quantum circuits on an IQM quantum computer

## Migration from legacy packages

If you have previously installed the (now obsolete) `qiskit-iqm`
or `cirq-iqm` packages in your Python environment, you should first
uninstall them:

```bash
uv pip uninstall qiskit-iqm cirq-iqm
```

Then reinstall `iqm-client` with the `--force-reinstall` option to ensure
a clean state:

```bash
uv pip install --force-reinstall "iqm-client[qiskit,cirq]"
```
