
# This file is appended to docs/conf.py file of each project when building
# public documentation.

intersphinx_mapping.update(
    {
        "iqm.station_control.client": (
            "../iqm-station-control-client",
            "https://docs.meetiqm.com/iqm-station-control-client/objects.inv",
        ),

        "iqm.station_control.interface": (
            "../iqm-station-control-client",
            "https://docs.meetiqm.com/iqm-station-control-client/objects.inv",
        ),

        "iqm.pulse": (
            "../iqm-pulse",
            "https://docs.meetiqm.com/iqm-pulse/objects.inv"
        ),

        "exa.common": (
            "../iqm-exa-common",
            "https://docs.meetiqm.com/iqm-exa-common/objects.inv",
        ),

        "iqm.data_definitions": (
            "../iqm-data-definitions",
            "https://docs.meetiqm.com/iqm-data-definitions/objects.inv",
        ),

        "iqm.data_definitions": (
            "../iqm-pulla",
            "https://docs.meetiqm.com/iqm-pulla/objects.inv",
        ),

        "iqm.benchmarks": (
            "../iqm-benchmarks",
            "https://docs.meetiqm.com/iqm-benchmarks/objects.inv",
        ),

        "iqm.iqm_client": (
            "../iqm-client",
            "https://docs.meetiqm.com/iqm-client/objects.inv",
        ),
        "iqm.qiskit_iqm": (
            "../iqm-client",
            "https://docs.meetiqm.com/iqm-client/objects.inv",
        ),
    }
)

extlinks = {}
