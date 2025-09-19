# Copyright 2021-2023 IQM client developers
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
"""Token manager for authentication and authorization to IQM's quantum computers. Part of IQM Client CLI."""

from datetime import datetime
import json
import os
from pathlib import Path
import time

from iqm.iqm_client.cli.auth import AUTH_REQUESTS_TIMEOUT, ClientAuthenticationError, refresh_request
from iqm.iqm_client.cli.models import ConfigFile
from psutil import pid_exists
from requests.exceptions import ConnectionError, Timeout


def daemonize_token_manager(cycle: int, config: ConfigFile, logfile: str | None = None) -> None:
    """Start a daemon process.

    Args:
        cycle: refresh cycle in seconds
        config: IQM Client CLI configuration
        logfile: path to file for writing errors

    """
    import daemon

    logfile = (
        logfile
        if logfile is not None
        else f"{os.environ.get('XDG_STATE_HOME', f'{Path.home()}/.local/state')}/iqm-client-cli/token_manager.log"
    )
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "w", encoding="UTF-8") as output:
        with daemon.DaemonContext(stdout=output, stderr=output):
            start_token_manager(cycle, config)


def start_token_manager(cycle: int, config: ConfigFile, single_run: bool = False) -> None:
    """Refresh tokens periodically.

    For each refresh cycle new tokens are requested from auth server.

    - If refresh is successful next refresh is attempted in the next cycle.
    - If auth server does not respond refresh is attempted repeatedly until it succeeds or
      the existing refresh token expires.
    - If auth server responds but returns an error code or invalid tokens token manager is stopped.

    Args:
        cycle: refresh cycle in seconds
        config: IQM Client CLI configuration
        single_run: if True, refresh tokens only once and exit; otherwise repeat refreshing indefinitely

    """
    while True:
        tokens_file = str(config.tokens_file)
        tokens = read_tokens(tokens_file)

        new_tokens, status, sleep_time = refresh_tokens(config, tokens, cycle)
        if new_tokens is None:
            break

        write_tokens(tokens_file, str(config.auth_server_url), status, **new_tokens)

        if single_run:
            break

        time.sleep(sleep_time)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}: Token manager stopped")


def read_tokens(path_to_tokens_file: str) -> dict:
    """Read current tokens from the tokens file.

    Args:
        path_to_tokens_file: path to the tokens file

    Returns:
        dict containing the tokens

    """
    with open(path_to_tokens_file, "r", encoding="utf-8") as file:
        tokens = json.load(file)
    return tokens


def refresh_tokens(config: ConfigFile, current_tokens: dict, cycle: int) -> tuple[dict | None, bool, int]:
    """Request new tokens from auth server.

    Args:
        config: IQM Client CLI configuration
        current_tokens: dict containing the current tokens from the tokens file
        cycle: refresh cycle length in seconds

    Returns:
        * dict containing new tokens or current tokens if auth server could not be connected or
          None if auth server refused to provide new tokens
        * bool, True if tokens were refreshed successfully, False otherwise
        * time to sleep before next refresh attempt

    """
    access_token = current_tokens.get("access_token", "")
    refresh_token = current_tokens.get("refresh_token", "")
    try:
        tokens = refresh_request(str(config.auth_server_url), config.realm, config.client_id, refresh_token)
        status = True
        sleep_time = cycle
        log_timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        print(f"{log_timestamp}: Tokens refreshed successfully.")
    except (Timeout, ConnectionError) as ex:
        # No connection to auth server or auth server did not respond, keep current tokens
        tokens = {"access_token": access_token, "refresh_token": refresh_token}
        status = False
        if isinstance(ex, ConnectionError):
            sleep_time = AUTH_REQUESTS_TIMEOUT
        else:
            sleep_time = 1
        log_timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        print(f"{log_timestamp}: No response from auth server: {ex}")
    except ClientAuthenticationError as ex:
        # Auth server responded but no valid tokens were received
        tokens = None
        status = False
        sleep_time = cycle
        log_timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        print(f"{log_timestamp}: Failed to authenticate with auth server: {ex}")

    return tokens, status, sleep_time


def write_tokens(
    path_to_tokens_file: str,
    auth_server_url: str,
    status: bool,
    *,
    access_token: str = "",
    refresh_token: str = "",
) -> None:
    """Write new tokens into the tokens file.

    Args:
        path_to_tokens_file: path to the tokens file
        auth_server_url: base URL of the auth server
        status: refresh status, True when successful, False otherwise
        access_token: new access token
        refresh_token: new refresh token

    """
    tokens_json = json.dumps(
        {
            "pid": os.getpid(),
            "timestamp": datetime.now().isoformat(),
            "refresh_status": "SUCCESS" if status else "FAILED",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "auth_server_url": auth_server_url,
        }
    )

    try:
        Path(path_to_tokens_file).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(path_to_tokens_file), "w", encoding="UTF-8") as file:
            file.write(tokens_json)
    except OSError as error:
        print("Error writing tokens file", error)


def check_token_manager(tokens_file: str) -> int | None:
    """Check whether a token manager related to the given tokens_file is running.

    Args:
        tokens_file: Path to a tokens JSON file.

    Returns:
        Optional[int]: PID of the process if process is running, None otherwise.

    """
    with open(tokens_file, "r", encoding="utf-8") as file:
        tokens_data = json.load(file)
    pid = tokens_data["pid"] if "pid" in tokens_data else None

    if pid and pid_exists(pid):
        return pid
    return None
