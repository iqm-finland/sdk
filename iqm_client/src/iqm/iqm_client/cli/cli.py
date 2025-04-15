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
"""Command line interface for managing user authentication when using IQM quantum computers."""

import contextlib
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any

import click
from iqm.iqm_client import __version__
from iqm.iqm_client.cli.auth import (
    AUTH_REQUESTS_TIMEOUT,
    ClientAccountSetupError,
    ClientAuthenticationError,
    login_request,
    logout_request,
    refresh_request,
    slash_join,
    time_left_seconds,
)
from iqm.iqm_client.cli.models import ConfigFile, TokensFile
from iqm.iqm_client.cli.token_manager import check_token_manager, daemonize_token_manager, start_token_manager
from psutil import NoSuchProcess, Process
from pydantic import ValidationError
import requests
from requests.exceptions import ConnectionError, Timeout

HOME_PATH = str(Path.home())
DEFAULT_CONFIG_PATH = os.path.join(HOME_PATH, ".config", "iqm-client-cli", "config.json")
DEFAULT_TOKENS_PATH = os.path.join(HOME_PATH, ".cache", "iqm-client-cli", "tokens.json")
REALM_NAME = "cortex"
CLIENT_ID = "iqm_client"
USERNAME = ""
REFRESH_PERIOD = 3 * 60  # in seconds


class ClickLoggingHandler(logging.Handler):
    """Simple log handler using click's echo function."""

    def __init__(self):
        super().__init__(level=logging.NOTSET)
        self.formatter = logging.Formatter("%(message)s")

    def emit(self, record):
        click.echo(self.format(record))


logger = logging.getLogger("iqm_client.cli.cli")
logger.addHandler(ClickLoggingHandler())
logger.setLevel(logging.INFO)


def _set_log_level_by_verbosity(verbose: bool) -> int:
    """Sets logger log level to DEBUG if verbose is True, to INFO otherwise.

    Args:
        verbose: whether logging should be verbose (i.e. DEBUG level)

    Returns:
        int: logging level which was set

    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        return logging.DEBUG
    logger.setLevel(logging.INFO)
    return logging.INFO


class ResolvedPath(click.Path):
    """A click parameter type for a resolved path.
    Normal ``click.Path(resolve_path=True)`` fails under Windows running python <= 3.9.
    See https://github.com/pallets/click/issues/2466
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> Any:
        abspath = Path(value).absolute()
        # fsdecode to ensure that the return value is a str.
        # (with click<8.0.3 Path.convert will return Path if passed a Path)
        return os.fsdecode(super().convert(abspath, param, ctx))


def _read_json(path: str) -> dict:
    """Read a JSON file.

    Args:
        path: path to the file to read
    Raises:
        click.FileError: if file is not a valid JSON file
    Returns:
        dict: data parsed from the file

    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise click.FileError(path, "file not found") from error
    except json.decoder.JSONDecodeError as error:
        raise click.FileError(path, f"file is not a valid JSON file: {error}") from error
    return data


def _is_parameter_source_cmd_line(ctx: click.Context, param_name: str | None) -> bool:
    """Check if the given parameter is submitted via command line"""
    if isinstance(param_name, str):
        param_source = ctx.get_parameter_source(param_name)
        if isinstance(param_source, click.core.ParameterSource):
            if param_source.name == "COMMANDLINE":
                return True
    return False


def _validate_path(ctx: click.Context, param: click.core.Option, path: str) -> str:
    """Callback for CLI prompt. If needed, confirmation to overwrite is prompted.

    The validation logic is not applied if the parameter is supplied via the
    command line.

    Args:
        ctx: click context
        param: click prompt param object
        path: path provided by user
    Returns:
        str: confirmed and finalized path

    """
    if ctx.obj is None:
        ctx.obj = {}
    if param.name in ctx.obj:
        return path
    ctx.obj[param.name] = True

    # Skip parameter validation completely if the parameter comes from command line
    if _is_parameter_source_cmd_line(ctx, param.name):
        msg = click.style(
            f'Skipping validation of "{param.opts[0]}", using the provided value "{path}" as is',
            fg="yellow",
        )
        click.echo(msg)
        return path

    # File doesn't exist, no need to confirm overwriting
    if not Path(path).is_file():
        return path

    # File exists, so user must either overwrite or enter a new path
    while True:
        msg = f"{click.style('File at given path already exists. Overwrite?', fg='red')}"
        if click.confirm(msg, default=None):
            return path

        new_path = click.prompt("New file path", type=ResolvedPath(dir_okay=False, writable=True, resolve_path=True))

        if new_path == path:
            continue
        return new_path


def _validate_config_file(config_file: str) -> ConfigFile:
    """Checks if provided config file is valid, i.e. it:
       - is valid JSON
       - satisfies IQM Client CLI format

    Args:
        config_file (str): --config-file option value
    Raises:
        click.FileError: if config_file is not valid JSON
        click.FileError: if config_file does not satisfy IQM Client CLI format
    Returns:
        ConfigFile: validated config loaded from config_file

    """
    # config_file must be in correct format
    config = _read_json(config_file)
    try:
        validated_config = ConfigFile(**config)
    except ValidationError as ex:
        raise click.FileError(
            config_file,
            f"""Provided config file is valid JSON, but does not satisfy IQM Client CLI format. Possible reasons:
- IQM Client CLI was upgraded and config file format is changed. Check the changelog.
- Config file was manually edited by someone.

Re-generate a valid config file by running 'iqmclient init'.

Full validation error:
{ex}""",
        )

    return validated_config


def _validate_tokens_file(tokens_file: str) -> TokensFile:
    """Checks if provided tokens file is valid, i.e. it:
       - is valid JSON
       - satisfies IQM Client CLI format

    Args:
        tokens_file (str): path to tokens file
    Raises:
        click.FileError: if tokens file is not valid JSON
        click.FileError: if tokens file does not satisfy IQM Client CLI format
    Returns:
        TokensFile: validated tokens loaded from tokens_file

    """
    # tokens_file must be in correct format
    tokens = _read_json(tokens_file)
    try:
        validated_tokens = TokensFile(**tokens)
    except ValidationError as ex:
        raise click.FileError(
            tokens_file,
            f"""Provided tokens file is valid JSON, but does not satisfy IQM Client CLI format. Possible reasons:
- IQM Client CLI was upgraded and tokens file format is changed. Check the changelog.
- Tokens file was manually edited by someone.

Re-generate a valid tokens file by running 'iqmclient auth login'.

Full validation error:
{ex}""",
        )

    return validated_tokens


def _validate_auth_server_url(ctx: click.Context, param: click.Option, base_url: str) -> str:
    """Checks if provided auth server URL is valid, i.e. it:
       - is a valid HTTP/HTTPS URL
       - is accessible
       - points to an authentication server

    The validation logic is not applied if the parameter is supplied via the
    command line.

    Args:
        ctx: click context
        param: click prompt param object
        base_url (str): auth server base URL to validate
    Returns:
        str: validated auth server base URL

    """
    if ctx.obj is None:
        ctx.obj = {}
    if param.name in ctx.obj:
        return base_url

    if _is_parameter_source_cmd_line(ctx, param.name):
        msg = click.style(
            f'Skipping validation of "{param.opts[0]}", using the provided value "{base_url}" as is',
            fg="yellow",
        )
        click.echo(msg)
        ctx.obj[param.name] = base_url
        return base_url

    is_valid = False
    while not is_valid:
        try:
            master = requests.get(f"{base_url}/realms/master", timeout=AUTH_REQUESTS_TIMEOUT)
            assert master.status_code == 200
            assert "public_key" in master.json()
            is_valid = True
        except (ConnectionError, AssertionError, ValueError):
            click.echo(f"No auth server could be accessed with URL {base_url}")
            is_valid = click.confirm("Do you still want to use it?", default=False)
        if not is_valid:
            base_url = click.prompt(str(param.prompt))

    ctx.obj[param.name] = base_url
    return base_url


def _validate_auth_realm(ctx: click.Context, param: click.Option, realm: str) -> str:
    """Checks if provided realm exists on auth server.

    The validation logic is not applied if the parameter is supplied via the
    command line.

    Args:
        ctx: click context
        param: click prompt param object
        realm (str): name of the realm
    Returns:
        str: validated realm name

    """
    if ctx.obj is None:
        ctx.obj = {}
    if param.name in ctx.obj:
        return realm

    if _is_parameter_source_cmd_line(ctx, param.name):
        msg = click.style(
            f'Skipping validation of "{param.opts[0]}", using the provided value "{realm}" as is',
            fg="yellow",
        )
        click.echo(msg)
        ctx.obj[param.name] = realm
        return realm

    base_url = ctx.obj.get("auth_server_url", None)
    if base_url is None:
        raise click.UsageError("Can not set realm name before setting auth server URL.")

    is_valid = False
    while not is_valid:
        try:
            realm_data = requests.get(f"{base_url}/realms/{realm}", timeout=AUTH_REQUESTS_TIMEOUT)
            assert realm_data.status_code == 200
            assert "public_key" in realm_data.json()
            is_valid = True
        except (ConnectionError, AssertionError, ValueError):
            click.echo(f"No auth realm could be accessed with URL {base_url}/realms/{realm}")
            is_valid = click.confirm("Do you still want to use it?", default=False)
        if not is_valid:
            realm = click.prompt(str(param.prompt))

    ctx.obj[param.name] = realm
    return realm


class IQMClientCliCommand(click.Group):
    """A custom click command group class to wrap global constants."""

    default_config_path: str = DEFAULT_CONFIG_PATH
    default_tokens_path: str = DEFAULT_TOKENS_PATH


@click.group(cls=IQMClientCliCommand)
@click.version_option(__version__)
def iqmclient() -> None:
    """IQM Client CLI for managing user authentication when using IQM quantum computers"""
    return


@iqmclient.command()
@click.help_option()
@click.option(
    "--config-file",
    prompt="Where to save config",
    callback=_validate_path,
    default=IQMClientCliCommand.default_config_path,
    type=ResolvedPath(dir_okay=False, writable=True, resolve_path=True),
    is_eager=True,
    help="Location where the configuration file will be saved.",
)
@click.option(
    "--tokens-file",
    prompt="Where to save auth tokens",
    callback=_validate_path,
    default=IQMClientCliCommand.default_tokens_path,
    type=ResolvedPath(dir_okay=False, writable=True, resolve_path=True),
    is_eager=True,
    help="Location where the tokens file will be saved.",
)
@click.option(
    "--auth-server-url",
    prompt="Authentication server URL",
    callback=_validate_auth_server_url,
    is_eager=True,
    help="Authentication server URL.",
)
@click.option(
    "--realm",
    prompt="Realm on IQM auth server",
    prompt_required=False,
    default=REALM_NAME,
    show_default=True,
    callback=_validate_auth_realm,
    help="Name of the realm on the IQM authentication server.",
)
@click.option(
    "--client-id",
    prompt="Client ID",
    prompt_required=False,
    default=CLIENT_ID,
    show_default=True,
    help="Client ID on the IQM authentication server.",
)
@click.option(
    "--username",
    prompt="Username (optional)",
    required=False,
    default=USERNAME,
    help="Username. If not provided, it will be asked for at login.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print extra information.")
def init(
    config_file: str, tokens_file: str, auth_server_url: str, realm: str, client_id: str, username: str, verbose: bool
) -> None:
    """Initialize configuration and authentication."""
    _set_log_level_by_verbosity(verbose)

    path_to_dir = Path(config_file).parent
    config_json = json.dumps(
        {
            "auth_server_url": auth_server_url,
            "realm": realm,
            "client_id": client_id,
            "username": username,
            "tokens_file": tokens_file,
        },
        indent=2,
    )

    # Tokens file exist, so token manager may be running. Notify user and kill token manager.
    if Path(tokens_file).is_file():
        pid = check_token_manager(tokens_file)
        if pid:
            logger.info("Active token manager (PID %s) will be killed.", pid)
            _safe_process_terminate(pid)

        # Remove tokens file to start from scratch after init
        os.remove(tokens_file)

    try:
        path_to_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(config_file), "w", encoding="UTF-8") as file:
            file.write(config_json)
            logger.debug("Saved configuration file: %s", config_file)
    except OSError as error:
        raise click.ClickException(f"Error writing configuration file, {error}") from error

    logger.info(
        "IQM Client CLI initialized successfully. Login and start the token manager with 'iqmclient auth login'."
    )


@iqmclient.group()
def auth() -> None:
    """Manage authentication."""
    return


@auth.command()
@click.option(
    "--config-file",
    default=IQMClientCliCommand.default_config_path,
    type=ResolvedPath(exists=True, dir_okay=False, resolve_path=True),
    help="Location of the configuration file to be used.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print extra information.")
def status(config_file, verbose) -> None:
    """Check status of authentication."""
    _set_log_level_by_verbosity(verbose)

    logger.debug("Using configuration file: %s", config_file)
    config = _validate_config_file(config_file)
    tokens_file = str(config.tokens_file)
    if not config.tokens_file.is_file():
        click.echo('Not logged in. Use "iqmclient auth login" to login.')
        return
    try:
        tokens_data = _validate_tokens_file(tokens_file)
    except click.FileError:
        click.echo('Provided tokens.json file is invalid. Use "iqmclient auth login" to generate new tokens.')
        return

    click.echo(f"Tokens file: {tokens_file}")
    if not tokens_data.pid:
        click.echo(
            "Tokens file doesn't contain PID. Probably, 'iqmclient auth login' was launched with '--no-refresh'\n"
        )

    refresh_status = tokens_data.refresh_status or "SUCCESS"
    styled_status = click.style(refresh_status, fg="green" if refresh_status == "SUCCESS" else "red")
    refresh_timestamp = tokens_data.timestamp.strftime("%m/%d/%Y %H:%M:%S")
    click.echo(f"Last refresh: {refresh_timestamp} from {tokens_data.auth_server_url} {styled_status}")
    seconds_at = time_left_seconds(tokens_data.access_token)
    time_left_at = str(timedelta(seconds=seconds_at))
    click.echo(f"Time left on access token (hh:mm:ss): {time_left_at}")
    seconds_rt = time_left_seconds(tokens_data.refresh_token)
    time_left_rt = str(timedelta(seconds=seconds_rt))
    click.echo(f"Time left on refresh token (hh:mm:ss): {time_left_rt}")

    active_pid = check_token_manager(tokens_file)
    if active_pid:
        click.echo(f"Token manager: {click.style('RUNNING', fg='green')} (PID {active_pid})")
    else:
        click.echo(f"Token manager: {click.style('NOT RUNNING', fg='red')}")


def _validate_iqm_client_cli_auth_login(no_daemon, no_refresh, config_file) -> ConfigFile:
    """Checks if provided combination of auth login options is valid:
       - no_daemon and no_refresh are mutually exclusive
       - config file should pass validation

    Args:
        config_file (str): --config-file option value
        no_daemon (bool): --no-daemon option value
        no_refresh (bool): --no-refresh option value
    Raises:
        click.BadOptionUsage: if both mutually exclusive --no-daemon and --no-refresh are set
        click.BadParameter: if config_file does not exist
    Returns:
        ConfigFile: validated config loaded from config_file

    """
    # --no-refresh and --no-daemon are mutually exclusive
    if no_refresh and no_daemon:
        raise click.BadOptionUsage(
            "--no-refresh", "Cannot request a non-daemonic (foreground) token manager when using '--no-refresh'."
        )

    # config file, even the default one, should exist
    if not Path(config_file).is_file():
        raise click.BadParameter(
            f"Provided config {config_file} does not exist. "
            + "Provide a different file or run 'iqmclient auth init' to create a new config file."
        )

    # config file should be valid JSON and satisfy IQM Client CLI format
    config = _validate_config_file(config_file)

    return config


def _refresh_tokens(
    refresh_period: int,
    no_daemon: bool,
    no_refresh: bool,
    config: ConfigFile,
) -> bool:
    """Refreshes token and returns success status

    Args:
        refresh_period (int): --refresh_period option value
        no_daemon (bool): --no-daemon option value
        no_refresh (bool): --no-refresh option value
        config (ConfigFile): IQM Client CLI config
    Returns:
        bool: whether token refresh was successful or not

    """
    # Tokens file exists; Refresh tokens without username/password
    tokens_file = str(config.tokens_file)
    try:
        refresh_token = _validate_tokens_file(tokens_file).refresh_token
    except (click.FileError, ValidationError):
        click.echo("Provided tokens.json file is invalid, continuing with login with username and password.")
        os.remove(tokens_file)
        return False

    logger.debug("Attempting to refresh tokens by using existing refresh token from file: %s", tokens_file)

    new_tokens = None
    try:
        new_tokens = refresh_request(str(config.auth_server_url), config.realm, config.client_id, refresh_token)
    except (Timeout, ConnectionError, ClientAuthenticationError):
        logger.info("Failed to refresh tokens by using existing token. Switching to username/password.")

    if new_tokens:
        save_tokens_file(tokens_file, new_tokens, str(config.auth_server_url))
        logger.debug("Saved new tokens file: %s", tokens_file)
        if no_refresh:
            logger.info("Existing token used to refresh session. Token manager not started due to '--no-refresh' flag.")
        elif no_daemon:
            logger.info("Existing token was used to refresh the auth session. Token manager started in foreground...")
            start_token_manager(refresh_period, config)
        else:
            logger.info("Existing token was used to refresh the auth session. Token manager daemon started.")
            daemonize_token_manager(refresh_period, config)
        return True
    return False


@auth.command()
@click.option(
    "--config-file",
    default=IQMClientCliCommand.default_config_path,
    type=ResolvedPath(exists=True, dir_okay=False, resolve_path=True),
    help="Location of the configuration file to be used.",
)
@click.option("--username", help="Username for authentication.")
@click.option("--password", help="Password for authentication.")
@click.option(
    "--refresh-period", default=REFRESH_PERIOD, show_default=True, help="How often to refresh tokens (in seconds)."
)
@click.option("--no-daemon", is_flag=True, default=False, help="Start token manager in foreground, not as daemon.")
@click.option(
    "--no-refresh", is_flag=True, default=False, help="Login, but do not start token manager to refresh tokens."
)
@click.option("-v", "--verbose", is_flag=True, help="Print extra information.")
def login(
    config_file: str,
    username: str,
    password: str,
    refresh_period: int,
    no_daemon: bool,
    no_refresh: bool,
    verbose: bool,
) -> None:
    """Authenticate on the IQM server, and optionally start a token manager to maintain the session."""
    _set_log_level_by_verbosity(verbose)

    if platform.system().lower().startswith("win") and not no_refresh and not no_daemon:
        click.echo(
            click.style("Warning", fg="yellow")
            + ": Daemonizing is not supported on Windows, and the application has started in foreground mode; "
            "please keep this terminal session open in order for IQM Client CLI to keep refreshing the tokens and "
            "maintaining the authentication.\n"
        )
        no_daemon = True

    # Validate whether the combination of options makes sense
    config = _validate_iqm_client_cli_auth_login(no_daemon, no_refresh, config_file)

    auth_server_url, realm, client_id = str(config.auth_server_url), config.realm, config.client_id
    tokens_file = str(config.tokens_file)

    if config.tokens_file.is_file():
        if check_token_manager(tokens_file):
            logger.info("Login aborted, because token manager is already running. See 'iqmclient auth status'.")
            return

        if _refresh_tokens(refresh_period, no_daemon, no_refresh, config):
            return

    # Login with username and password
    username = username or config.username or click.prompt("Username")
    if config.username:
        click.echo(f"Username: {username}")
    password = password or click.prompt("Password", hide_input=True)
    tokens = None

    while tokens is None:
        try:
            tokens = login_request(auth_server_url, realm, client_id, username, password)
        except ConnectionError as exc:
            raise click.ClickException(f"Authentication server at {auth_server_url} is not accessible") from exc
        except Timeout as exc:
            raise click.ClickException(f"Authentication server at {auth_server_url} is not responding") from exc
        except ClientAuthenticationError as exc:
            raise click.ClickException(f"Failed to authenticate, {exc}") from exc
        except ClientAccountSetupError as exc:
            password_update_form_url = slash_join(auth_server_url, f"realms/{realm}/account")
            raise click.ClickException(
                f"""
Failed to authenticate, because your account is not fully set up yet.
Please update your password at {password_update_form_url}
"""
            ) from exc

    logger.info("Logged in successfully as %s", username)
    save_tokens_file(tokens_file, tokens, auth_server_url)
    env_var_command = "set" if platform.system().lower().startswith("win") else "export"
    click.echo(
        f"""
To use the tokens file with IQM Client or IQM Client-based software, set the environment variable:

{env_var_command} IQM_TOKENS_FILE={tokens_file}

Refer to IQM Client documentation for details: https://docs.meetiqm.com/iqm-client/
"""
    )

    if no_refresh:
        logger.info("Token manager not started due to '--no-refresh' flag.")
    elif no_daemon:
        logger.info("Starting token manager in foreground...")
        start_token_manager(refresh_period, config)
    else:
        logger.info("Starting token manager daemon...")
        daemonize_token_manager(refresh_period, config)


@auth.command()
@click.option(
    "--config-file",
    type=ResolvedPath(exists=True, dir_okay=False, resolve_path=True),
    default=IQMClientCliCommand.default_config_path,
)
@click.option("--keep-tokens", is_flag=True, default=False, help="Don't delete tokens file, but kill token manager.")
@click.option("-f", "--force", is_flag=True, default=False, help="Don't ask for confirmation.")
def logout(config_file: str, keep_tokens: str, force: bool) -> None:
    """Either logout completely, or just stop token manager while keeping tokens file."""
    config = _validate_config_file(config_file)
    auth_server_url, realm, client_id = str(config.auth_server_url), config.realm, config.client_id
    tokens_file = config.tokens_file

    if not tokens_file.is_file():
        click.echo("Not logged in.")
        return

    try:
        tokens = _validate_tokens_file(str(tokens_file))
    except click.FileError:
        click.echo("Found invalid tokens.json, cannot perform any logout steps.")
        return

    pid = check_token_manager(str(tokens_file))
    refresh_token = tokens.refresh_token

    # 1. Keep tokens, kill daemon
    if keep_tokens and pid:
        if force or click.confirm("Keep tokens file and kill token manager. OK?", default=None):
            _safe_process_terminate(pid, "Token manager killed.")
            return

    # 2. Keep tokens, do nothing
    if keep_tokens and not pid:
        click.echo("Token manager is not running, and you chose to keep tokens. Nothing to do, exiting.")
        return

    # 3. Delete tokens, perform logout, kill daemon
    if not keep_tokens and pid:
        if force or click.confirm("Logout from server, delete tokens and kill token manager. OK?", default=None):
            try:
                logout_request(auth_server_url, realm, client_id, refresh_token)
            except (Timeout, ConnectionError, ClientAuthenticationError) as error:
                logger.warning(
                    "Failed to revoke tokens due to error when connecting to authentication server: %s", error
                )

            _safe_process_terminate(pid)
            os.remove(tokens_file)
            logger.info("Tokens file deleted. Logged out.")
            return

    # 4. Delete tokens, perform logout
    if not keep_tokens and not pid:
        if force or click.confirm("Logout from server and delete tokens. OK?", default=None):
            try:
                logout_request(auth_server_url, realm, client_id, refresh_token)
            except (Timeout, ConnectionError, ClientAuthenticationError) as error:
                logger.warning(
                    "Failed to revoke tokens due to error when connecting to authentication server: %s", error
                )

            os.remove(tokens_file)
            logger.info("Tokens file deleted. Logged out.")
            return

    logger.info("Logout aborted.")


def save_tokens_file(path: str, tokens: dict[str, str], auth_server_url: str) -> None:
    """Saves tokens as JSON file at given path.

    Args:
        path (str): path to the file to write
        tokens (dict[str, str]): authorization access and refresh tokens
        auth_server_url (str): base url of the authorization server
    Raises:
        OSError: if writing to file fails

    """
    path_to_dir = Path(path).parent
    tokens_data = {
        "timestamp": datetime.now().isoformat(),
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "auth_server_url": auth_server_url,
    }

    try:
        path_to_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(path), "w", encoding="UTF-8") as file:
            file.write(json.dumps(tokens_data))
    except OSError as error:
        raise click.ClickException(f"Error writing tokens file, {error}") from error


def _safe_process_terminate(pid: int, msg: str = "") -> None:
    """Try to terminate a process given a process ID (PID).
    Suppress the exception in case the process is not existing
    """
    with contextlib.suppress(NoSuchProcess):
        Process(pid).terminate()
        if msg:
            logger.info(msg)


if __name__ == "__main__":
    iqmclient(sys.argv[1:])
