"""Verify and benchmark sweep results performance between legacy and streaming formats."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import statistics
import sys
import time
import uuid

from iqm.models.playlist import Playlist
import numpy as np
import requests

from exa.common.data.parameter import Parameter, Setting
from exa.common.data.setting_node import SettingNode
from exa.common.sweep.util import linear_index_sweep
from iqm.station_control.client.serializers import (
    SweepResultsArtifactFormat,
    deserialize_sweep_results,
    deserialize_sweep_results_chunks,
    serialize_sweep_job_request,
    serialize_sweep_results,
    serialize_sweep_results_chunk,
)
from iqm.station_control.interface.models import SweepDefinition


def _assert_results_equal(
    legacy: dict[str, list[np.ndarray]],
    streaming: dict[str, list[np.ndarray]],
    *,
    assert_values: bool = True,
) -> None:
    """Raise ValueError if legacy and streaming result dicts differ in keys, lengths, shapes, or values."""
    if legacy.keys() != streaming.keys():
        raise ValueError(
            f"Result keys do not match.\n"
            f"Legacy keys:    {sorted(legacy.keys())}\n"
            f"Streaming keys: {sorted(streaming.keys())}"
        )

    for key, legacy_list in legacy.items():
        streaming_list = streaming[key]
        if len(legacy_list) != len(streaming_list):
            raise ValueError(
                f"Result lengths for key {key} do not match.\n"
                f"Legacy:    {len(legacy_list)} spots\n"
                f"Streaming: {len(streaming_list)} spots"
            )

        for idx, (leg_arr, str_arr) in enumerate(zip(legacy_list, streaming_list, strict=True)):
            if leg_arr.shape != str_arr.shape:
                raise ValueError(
                    f"Array shapes at index {idx} for key {key} do not match.\n"
                    f"Legacy shape:    {leg_arr.shape}\n"
                    f"Streaming shape: {str_arr.shape}"
                )
            if assert_values and not np.array_equal(leg_arr, str_arr):
                raise ValueError(
                    f"Data arrays at index {idx} for key {key} do not match.\n"
                    f"Legacy array:    {leg_arr}\n"
                    f"Streaming array: {str_arr}"
                )


def _get_urls(
    args_core_url: str | None,
    args_station_url: str | None,
    args_server_url: str | None = None,
) -> tuple[str, str, str]:
    """Resolve and return (core_url, station_url, server_url) from CLI args and environment variables."""
    # Resolve IQM Server base URL (used for job submission via IQM Server API)
    server_base: str | None = None
    if args_server_url:
        server_base = args_server_url.rstrip("/")
    elif "IQM_SERVER_URL" in os.environ:
        server_base = os.environ["IQM_SERVER_URL"].rstrip("/")
        for suffix in ("/station", "/core"):
            if server_base.endswith(suffix):
                server_base = server_base[: -len(suffix)]

    # Resolve Core URL (used for DUT label lookup and artifact fetching)
    if args_core_url:
        core_url = args_core_url.rstrip("/")
    elif "CORE_API_SERVICE_URL" in os.environ:
        core_url = os.environ["CORE_API_SERVICE_URL"].rstrip("/")
    elif server_base:
        core_url = f"{server_base}/core"
    else:
        raise ValueError(
            "Core API URL not provided. Please set IQM_SERVER_URL, CORE_API_SERVICE_URL, or supply --core-url."
        )

    # Resolve Station URL (used by EXA provider for high-fidelity sweep construction)
    if args_station_url:
        station_url = args_station_url.rstrip("/")
    elif "STATION_CONTROL_SERVICE_URL" in os.environ:
        station_url = os.environ["STATION_CONTROL_SERVICE_URL"].rstrip("/")
    elif server_base:
        station_url = f"{server_base}/station"
    else:
        station_url = core_url.replace("/core", "/station")

    # Derive server_url if not already set
    if server_base is None:
        server_base = core_url[: -len("/core")] if core_url.endswith("/core") else core_url

    return core_url, station_url, server_base


def _get_auth_headers() -> dict[str, str]:
    """Return Authorization headers from IQM_TOKEN, IQM_ADMIN_TOKEN, or a local mise TOML file."""
    headers = {}
    token = os.getenv("IQM_TOKEN")
    if not token:
        # Fallback for local mise setups when shell integration is not active
        mise_env = os.getenv("MISE_ENV")
        if mise_env:
            for candidate in [f"mise.{mise_env}.toml", f"../mise.{mise_env}.toml", f"../../mise.{mise_env}.toml"]:
                if os.path.exists(candidate):
                    try:
                        with open(candidate) as f:
                            for line in f:
                                # Look for IQM_TOKEN = "..." (single or double quotes, optional spaces)
                                line_stripped = line.strip()
                                if "IQM_TOKEN" in line_stripped:
                                    parts = line_stripped.split("=", 1)
                                    if len(parts) == 2 and parts[0].strip() == "IQM_TOKEN":  # noqa: PLR2004
                                        val = parts[1].strip().strip("\"'")
                                        if val:
                                            token = val
                                            break
                            if token:
                                break
                    except Exception:
                        pass

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif admin_token := os.getenv("IQM_ADMIN_TOKEN"):
        headers["Authorization"] = f"Bearer {admin_token}"
    return headers


def _get_dut_label(core_url: str, headers: dict[str, str]) -> str:
    """Return the first DUT label from Core API configuration, or a hardcoded fallback."""
    try:
        response = requests.get(f"{core_url}/configuration", headers=headers, timeout=10)
        if response.ok:
            config = response.json()
            encoded_duts_dict = config["static_configs_b64"]["duts"]
            content_type = list(encoded_duts_dict.keys())[0]
            raw_duts = base64.b64decode(encoded_duts_dict[content_type])
            duts_data = json.loads(raw_duts)
            if isinstance(duts_data, list) and duts_data:
                return duts_data[0]["label"]
    except Exception:
        pass
    return "M101_W1_X12_A01"


def _get_quantum_computer(server_url: str, headers: dict[str, str]) -> str:
    """Return the quantum computer alias from IQM_QUANTUM_COMPUTER env or IQM Server discovery."""
    if qc := os.getenv("IQM_QUANTUM_COMPUTER"):
        return qc
    try:
        response = requests.get(f"{server_url}/api/v1/quantum-computers", headers=headers, timeout=10)
        if response.ok:
            qcs = response.json().get("quantum_computers", [])
            if qcs:
                return qcs[0]["alias"]
    except Exception:
        pass
    raise RuntimeError("Could not auto-discover quantum computer. Set IQM_QUANTUM_COMPUTER or use --quantum-computer.")


try:
    from exa.core import Experiment
    from exa.core.provider import provider
    from iqm.pulse.builder import ScheduleBuilder
    from iqm.pulse.timebox import TimeBox

    HAS_EXA = True
except ImportError:
    Experiment = None  # type: ignore[assignment]
    provider = None  # type: ignore[assignment]
    ScheduleBuilder = None  # type: ignore[assignment]
    TimeBox = None  # type: ignore[assignment]
    HAS_EXA = False


def _some_circuit(components: list[str], builder: ScheduleBuilder) -> list[TimeBox]:
    """Build a two-element circuit list (RX gate repeated) for use as a benchmark payload."""
    circuit = TimeBox.composite([builder.prx([qubit]).rx(3.14) for qubit in components])
    return [circuit, circuit]


def _create_sweep_payload(
    sweep_id: uuid.UUID,
    dut_label: str,
    format: SweepResultsArtifactFormat,
    num_spots: int = 0,
    station_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Serialize a sweep job request payload, using EXA for a high-fidelity sweep if available."""
    if HAS_EXA and num_spots > 0 and station_url:
        try:
            print("Initializing EXA provider to build high-fidelity simulated sweep...")
            token = None
            if headers and "Authorization" in headers:
                token_part = headers["Authorization"]
                if token_part.startswith("Bearer "):
                    token = token_part[len("Bearer ") :]
            provider.init_station_control_runtime(station_url, token=token, username="benchmark_user")

            topo = Experiment().chip_topology
            qubits = list(topo.qubits_sorted)
            if not qubits:
                raise ValueError("No qubits found in chip topology!")

            # Use the first qubit
            target_qubits = qubits[:1]

            experiment = Experiment(circuit_generation_function=_some_circuit)
            experiment.options.repetitions = 500  # realistic shots
            experiment.options.readout_mode = "single_shot_threshold"

            sweep_definition = experiment.create(components=target_qubits).sweep_definition
            sweep_definition.sweep_id = sweep_id

            # Add a real sweep over options.end_delay for num_spots
            sweep_definition.sweeps = linear_index_sweep(Parameter("options.end_delay"), num_spots)

            return serialize_sweep_job_request(
                sweep_definition,
                queue_name="sweeps",
                sweep_results_format=format,
            )
        except Exception as e:
            print(f"Warning: Failed to construct EXA sweep payload ({e}). Falling back to dummy.")

    settings = SettingNode(
        "root",
        options=SettingNode(
            "options",
            end_delay=Setting(Parameter("options.end_delay"), 350e-6),
            playlist_repeats=Setting(Parameter("options.playlist_repeats"), 10),
        ),
    )
    # Submitting actual sweeps + return parameters on mock setups without compiling and
    # having actual readout triggers raises NotFoundError, so we default to empty setup.
    sweeps = []
    return_parameters = []

    sweep_definition = SweepDefinition(
        sweep_id=sweep_id,
        dut_label=dut_label,
        settings=settings,
        sweeps=sweeps,
        return_parameters=return_parameters,
        playlist=Playlist(channel_descriptions={}, segments=[]),
    )
    return serialize_sweep_job_request(
        sweep_definition,
        queue_name="sweeps",
        sweep_results_format=format,
    )


def _submit_job(server_url: str, quantum_computer: str, payload: bytes, headers: dict[str, str]) -> uuid.UUID:
    """Submit a sweep job via IQM Server API and return the server-assigned job ID."""
    request_headers = {"Content-Type": "application/protobuf", **headers}
    response = requests.post(
        f"{server_url}/api/v1/jobs/{quantum_computer}/sweep",
        data=payload,
        headers=request_headers,
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"Failed to submit sweep job to IQM Server: {response.text}")
    return uuid.UUID(response.json()["id"])


def _wait_for_job(server_url: str, job_id: uuid.UUID, headers: dict[str, str]) -> None:
    """Poll IQM Server until the job reaches a terminal status, raising on failure or cancellation."""
    terminal_statuses = {"completed", "failed", "cancelled"}
    print(f"Waiting for job {job_id} to finish...")
    while True:
        response = requests.get(f"{server_url}/api/v1/jobs/{job_id}", headers=headers, timeout=10)
        if not response.ok:
            raise RuntimeError(f"Failed to query job status for {job_id}: {response.text}")
        status = response.json()["status"]
        if status in terminal_statuses:
            if status != "completed":
                raise RuntimeError(f"Job {job_id} terminated with unsuccessful status: {status}")
            break
        time.sleep(0.5)


def _measure_get_artifact(url: str, headers: dict[str, str], iterations: int) -> tuple[list[float], int, int, str]:
    """Return (times, logical_bytes, wire_bytes, content_encoding)."""
    times = []
    logical_size = 0
    wire_size = 0
    encoding = "identity"

    for i in range(iterations):
        if i == 0:
            # First pass: measure wire size by disabling auto-decompression
            raw = requests.get(url, headers=headers, timeout=30, stream=True)
            encoding = raw.headers.get("content-encoding", "identity")
            raw.raw.decode_content = False
            wire_bytes = raw.raw.read()
            wire_size = len(wire_bytes)
            # Logical size: decompress if needed
            logical_size = len(gzip.decompress(wire_bytes)) if encoding == "gzip" else wire_size

        start = time.perf_counter()
        requests.get(url, headers=headers, timeout=30)
        times.append(time.perf_counter() - start)

    return times, logical_size, wire_size, encoding


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=1)
_SYNTHETIC_CASES = [
    {
        "name": "random_float64",
        "desc": "random float64, 64 samples  (worst case: incompressible)",
        "params": ["p_a", "p_b", "p_c", "p_d"],
        "gen": lambda: rng.random(64).astype(np.float64),
    },
    {
        "name": "bitstring",
        "desc": "0/1 bitstring, 1000 shots  [QECI single-shot, best case]",
        "params": ["readout"],
        "gen": lambda: rng.integers(0, 2, size=1000).astype(np.float64),
    },
    {
        "name": "low_entropy_int",
        "desc": "integers 0-3, 500 samples  (moderate compression)",
        "params": ["counts"],
        "gen": lambda: rng.integers(0, 4, size=500).astype(np.float64),
    },
    {
        "name": "sparse_float64",
        "desc": "mostly-zero float64, 64 samples  (realistic QECI threshold)",
        "params": ["p_a", "p_b"],
        "gen": lambda: (rng.random(64) < 0.05).astype(np.float64),  # noqa: PLR2004
    },
]


def _run_synthetic_case(case: dict, num_spots: int) -> dict:
    """Run one synthetic benchmark case and return a dict of size and timing metrics."""
    sweep_id = uuid.uuid4()
    params: list[str] = case["params"]
    gen = case["gen"]

    synthetic_results: dict[str, list[np.ndarray]] = {p: [gen() for _ in range(num_spots)] for p in params}

    t0 = time.perf_counter()
    legacy_bytes = serialize_sweep_results(sweep_id, synthetic_results)
    legacy_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    chunks = []
    for spot_idx in range(1, num_spots + 1):
        for key, val_list in synthetic_results.items():
            chunks.append(serialize_sweep_results_chunk(sweep_id, spot_idx, key, val_list[spot_idx - 1]))
    streaming_bytes = b"".join(chunks)
    streaming_ser = time.perf_counter() - t0

    t0 = time.perf_counter()
    legacy_data = deserialize_sweep_results(legacy_bytes)
    legacy_deser = time.perf_counter() - t0

    t0 = time.perf_counter()
    streaming_data = deserialize_sweep_results_chunks(streaming_bytes)
    streaming_deser = time.perf_counter() - t0

    _assert_results_equal(legacy_data, streaming_data)

    return {
        "name": case["name"],
        "desc": case["desc"],
        "legacy_size": len(legacy_bytes),
        "streaming_size": len(streaming_bytes),
        "legacy_ser": legacy_ser,
        "streaming_ser": streaming_ser,
        "legacy_deser": legacy_deser,
        "streaming_deser": streaming_deser,
    }


def _print_synthetic_table(results: list[dict]) -> None:
    """Print a formatted comparison table of synthetic benchmark results to stdout."""
    name_w = max(len(r["name"]) for r in results)
    desc_w = max(len(r["desc"]) for r in results)

    def fmt_kb(n: int) -> str:
        return f"{n / 1024:>9,.1f} KB"

    def fmt_size_delta(streaming: int, legacy: int) -> str:
        pct = (streaming / legacy - 1.0) * 100.0
        return f"{pct:>+7.1f}%"

    def fmt_speedup(legacy_t: float, v2_t: float) -> str:
        if v2_t < 1e-9:  # noqa: PLR2004
            return "    ∞"
        x = legacy_t / v2_t
        return f"{x:>5.2f}x"

    col_name = f"{'Case':<{name_w}}"
    col_desc = f"{'Description':<{desc_w}}"
    header = f"{col_name}  {col_desc}  {'Legacy':>12}  {'V2':>12}  {'Size Δ':>8}  {'Ser ×':>7}  {'De ×':>7}"
    rule = "─" * len(header)

    print(rule)
    print(header)
    print(rule)
    for r in results:
        print(
            f"{r['name']:<{name_w}}  "
            f"{r['desc']:<{desc_w}}  "
            f"{fmt_kb(r['legacy_size'])}  "
            f"{fmt_kb(r['streaming_size'])}  "
            f"{fmt_size_delta(r['streaming_size'], r['legacy_size'])}  "
            f"{fmt_speedup(r['legacy_ser'], r['streaming_ser'])}  "
            f"{fmt_speedup(r['legacy_deser'], r['streaming_deser'])}"
        )
    print(rule)
    print("  Ser ×, De × = legacy_time / V2_time  (>1.0x means V2 is faster)")


def _run_synthetic_benchmark(num_spots: int) -> None:
    """Run all synthetic benchmark cases and print the results table."""
    print(f"\nRunning synthetic benchmark: {num_spots:,} spots, {len(_SYNTHETIC_CASES)} cases\n")

    results = []
    for case in _SYNTHETIC_CASES:
        print(f"  {case['name']}...", end="", flush=True)
        result = _run_synthetic_case(case, num_spots)
        results.append(result)
        print(" ✓")

    print()
    _print_synthetic_table(results)
    print()


# ---------------------------------------------------------------------------
# Live endpoint benchmark
# ---------------------------------------------------------------------------


def run_benchmark(  # noqa: PLR0915
    core_url: str,
    station_url: str,
    server_url: str,
    legacy_job_id: str | None,
    streaming_job_id: str | None,
    iterations: int,
    num_spots: int = 0,
    synthetic_only: bool = False,
) -> int:
    """Run verification and benchmark GET retrival timings."""
    if synthetic_only:
        _run_synthetic_benchmark(num_spots if num_spots > 0 else 100)
        return 0

    headers = _get_auth_headers()

    if not legacy_job_id or not streaming_job_id:
        print("Auto-submitting new legacy and streaming sweep jobs for benchmarking...")
        dut_label = _get_dut_label(core_url, headers)
        print(f"Detected DUT: {dut_label}")

        quantum_computer = _get_quantum_computer(server_url, headers)
        print(f"Quantum computer: {quantum_computer}")

        legacy_sweep_id = uuid.uuid4()
        streaming_sweep_id = uuid.uuid4()

        legacy_payload = _create_sweep_payload(
            legacy_sweep_id,
            dut_label,
            SweepResultsArtifactFormat.LEGACY_PROTOBUF,
            num_spots,
            station_url=station_url,
            headers=headers,
        )
        streaming_payload = _create_sweep_payload(
            streaming_sweep_id,
            dut_label,
            SweepResultsArtifactFormat.STREAMING_V2,
            num_spots,
            station_url=station_url,
            headers=headers,
        )

        legacy_uuid = _submit_job(server_url, quantum_computer, legacy_payload, headers)
        streaming_uuid = _submit_job(server_url, quantum_computer, streaming_payload, headers)

        _wait_for_job(server_url, legacy_uuid, headers)
        _wait_for_job(server_url, streaming_uuid, headers)

        legacy_job_id = str(legacy_uuid)
        streaming_job_id = str(streaming_uuid)

    legacy_url = f"{server_url}/api/v1/jobs/{legacy_job_id}/artifacts/sweep_results"
    streaming_url = f"{server_url}/api/v1/jobs/{streaming_job_id}/artifacts/sweep_results_streaming"

    print(f"\nLegacy Artifact URL:    {legacy_url}")
    print(f"Streaming Artifact URL: {streaming_url}")
    print("\nFetching and validating results match...")

    try:
        r_legacy_initial = requests.get(legacy_url, headers=headers, timeout=30)
        r_legacy_initial.raise_for_status()
    except Exception as e:
        print(f"Error fetching legacy artifact from {legacy_url}: {e}", file=sys.stderr)
        return 1

    try:
        r_streaming_initial = requests.get(streaming_url, headers=headers, timeout=30)
        r_streaming_initial.raise_for_status()
    except Exception as e:
        print(f"Error fetching streaming artifact from {streaming_url}: {e}", file=sys.stderr)
        return 1

    try:
        legacy_data = deserialize_sweep_results(r_legacy_initial.content)
        streaming_data = deserialize_sweep_results_chunks(r_streaming_initial.content)
        _assert_results_equal(legacy_data, streaming_data, assert_values=False)
        print("✓ Verified! Sweep results are identical.")
    except Exception as e:
        print(f"✖ Verification failed: {e}", file=sys.stderr)
        return 1

    print(f"\nBenchmarking GET endpoints over {iterations} iterations...")
    legacy_times, legacy_logical, legacy_wire, legacy_enc = _measure_get_artifact(legacy_url, headers, iterations)
    streaming_times, streaming_logical, streaming_wire, streaming_enc = _measure_get_artifact(
        streaming_url, headers, iterations
    )

    legacy_median = statistics.median(legacy_times)
    streaming_median = statistics.median(streaming_times)

    def _fmt_size(logical: int, wire: int, enc: str) -> str:
        if enc == "gzip":
            return f"{logical:,} bytes  ({wire:,} bytes wire, gzip)"
        return f"{logical:,} bytes"

    print("\n--- RESULTS ---")
    print(f"Legacy Format:     {_fmt_size(legacy_logical, legacy_wire, legacy_enc)}")
    print(f"Streaming V2:      {_fmt_size(streaming_logical, streaming_wire, streaming_enc)}")
    if legacy_logical > 0:
        print(f"Size Ratio (stream/legacy):  logical {streaming_logical / legacy_logical:.3%}", end="")
        if legacy_enc == "gzip" and streaming_enc == "gzip":
            print(f"  |  wire {streaming_wire / legacy_wire:.3%}", end="")
        print()

    print("\nLegacy Retrieval Times:")
    print(f"  Min: {min(legacy_times):.6f} s  |  Max: {max(legacy_times):.6f} s  |  Median: {legacy_median:.6f} s")

    print("Streaming V2 Retrieval Times:")
    print(
        f"  Min: {min(streaming_times):.6f} s  |  "
        f"Max: {max(streaming_times):.6f} s  |  "
        f"Median: {streaming_median:.6f} s"
    )

    if streaming_median > 0:
        print(f"\nRatio (legacy/streaming): {legacy_median / streaming_median:.2f}x speedup")

    return 0


def main() -> int:
    """Run the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Verify and benchmark sweep results performance between legacy and streaming formats."
    )
    parser.add_argument(
        "--legacy-job-id",
        help="UUID of the job run with the legacy sweep results format. Optional (will auto-submit if omitted).",
    )
    parser.add_argument(
        "--streaming-job-id",
        help="UUID of the job run with the streaming_v2 format. Optional (will auto-submit if omitted).",
    )
    parser.add_argument(
        "--server-url",
        help="Base URL of the IQM Server (for job submission). Defaults to IQM_SERVER_URL env var.",
    )
    parser.add_argument(
        "--core-url",
        help="Base URL of the Core API (for DUT lookup). Defaults to CORE_API_SERVICE_URL or IQM_SERVER_URL.",
    )
    parser.add_argument(
        "--station-url",
        help=(
            "Base URL of the Station Control (for EXA sweep construction). "
            "Defaults to STATION_CONTROL_SERVICE_URL or derived from IQM_SERVER_URL."
        ),
    )
    parser.add_argument(
        "--quantum-computer",
        help="Quantum computer alias on the IQM Server. Defaults to IQM_QUANTUM_COMPUTER or auto-discovered.",
    )
    parser.add_argument(
        "--num-spots",
        type=int,
        default=100,
        help="Make submitted sweeps define this many spots to generate non-trivial payload sizes. Default: 100.",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only run the local synthetic in-memory serialization/deserialization benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of GET requests to perform for each format to record median timing. Default: 5.",
    )

    args = parser.parse_args()

    if args.quantum_computer:
        os.environ["IQM_QUANTUM_COMPUTER"] = args.quantum_computer

    try:
        core_url, station_url, server_url = _get_urls(args.core_url, args.station_url, args.server_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return run_benchmark(
        core_url=core_url,
        station_url=station_url,
        server_url=server_url,
        legacy_job_id=args.legacy_job_id,
        streaming_job_id=args.streaming_job_id,
        iterations=args.iterations,
        num_spots=args.num_spots,
        synthetic_only=args.synthetic_only,
    )


if __name__ == "__main__":
    sys.exit(main())
