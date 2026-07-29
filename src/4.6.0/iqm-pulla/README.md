# IQM Pulla

Pulla (pulse-level access) is a client-side Python library which enables
the generation and execution of pulse-level jobs on an [IQM](https://iqm.tech/)
quantum computer.
Within the existing IQM QCCSW stack, Pulla is somewhere between
circuit-level execution and EXA experiments.

An interactive user guide is available as a Jupyter notebook in the `docs`
folder.

## Testing

If you want to run a particular notebook and see the output cells printed in
the terminal, you can use `nbconvert` with [`jq`](https://jqlang.github.io/jq/download/)
like so:

<!-- markdownlint-disable MD013 -->
```bash
jupyter nbconvert --to notebook --execute  "docs/Quick Start.ipynb" --stdout |
  jq -r '.cells[] | select(.outputs) | .outputs[] | select(.output_type == "stream") | .text[]'
```
<!-- markdownlint-enable MD013 -->
