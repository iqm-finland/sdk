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
"""Pydantic models for IQM Client CLI files."""

from datetime import datetime
from pathlib import Path

from pydantic import AnyUrl, BaseModel


class ConfigFile(BaseModel):
    """Model of configuration file, used for validating JSON."""

    auth_server_url: AnyUrl
    realm: str
    client_id: str
    username: str | None = None
    tokens_file: Path


class TokensFile(BaseModel):
    """Model of tokens file, used for validating JSON."""

    pid: int | None = None
    timestamp: datetime
    access_token: str
    refresh_token: str
    refresh_status: str | None = None
    auth_server_url: AnyUrl
