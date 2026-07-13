
# This file is appended to docs/conf.py file of each project when building
# public documentation.

python_version = ".".join(map(str, sys.version_info[0:2]))

# Overwrite original mapping with a manually-compiled dictionary of packages
# and links to their documentation and "objects.inv" files.

intersphinx_mapping = {
    "dimod": ("https://docs.dwavequantum.com/en/latest", None),  # External packages
    "matplotlib": ("https://matplotlib.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": (f"https://docs.python.org/{python_version}", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer", None),
    "quimb": ("https://quimb.readthedocs.io/en/latest", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),

    "exa-common": ("../iqm-exa-common", "https://docs.iqm.tech/iqm-exa-common/objects.inv"),  # Internal packages
    "iqm-benchmarks": ("../iqm-benchmarks", "https://docs.iqm.tech/iqm-benchmarks/objects.inv"),
    "iqm-client": ("../iqm-client", "https://docs.iqm.tech/iqm-client/objects.inv"),
    "iqm-data-definitions": ("../iqm-data-definitions", "https://docs.iqm.tech/iqm-data-definitions/objects.inv"),
    "iqm-pulla": ("../iqm-pulla", "https://docs.iqm.tech/iqm-pulla/objects.inv"),
    "iqm-pulse": ("../iqm-pulse", "https://docs.iqm.tech/iqm-pulse/objects.inv"),
    "iqm-qaoa": ("../iqm-qaoa", "https://docs.iqm.tech/iqm-qaoa/objects.inv"),
    "iqm-qubit-selector": ("../iqm-qubit-selector", "https://docs.iqm.tech/iqm-qubit-selector/objects.inv"),
    "iqm-station-control-client": ("../iqm-station-control-client", "https://docs.iqm.tech/iqm-station-control-client/objects.inv"),
}

extlinks = {}
