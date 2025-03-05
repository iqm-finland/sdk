# Copyright 2024 IQM
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

"""Pydantic model for SettingNode."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from exa.common.api.model.setting_model import SettingModel
from exa.common.data.setting_node import SettingNode


class SettingNodeModel(BaseModel):
    """Pydantic setting node model"""  # noqa: D200

    name: str
    settings: Dict[str, SettingModel]
    subtrees: Dict[str, SettingNodeModel]

    def __hash__(self):
        return hash((self.name, self.settings, self.subtrees))

    def __eq__(self, other: SettingNodeModel):
        return isinstance(other, SettingNodeModel) and (
            (self.name, self.settings, self.subtrees) == (other.name, other.settings, other.subtrees)
        )

    model_config = ConfigDict(extra="allow")

    def decode(self) -> SettingNode:
        """Creates instance of :class:`exa.common.data.setting_node.SettingNode` out of pydantic `SettingNodeModel`"""
        return SettingNode(self.name, **SettingNodeModel._get_children(self))

    @staticmethod
    def _get_children(root: SettingNodeModel) -> Dict[str, Any]:
        children = {}
        for k_s, v_s in root.settings.items():
            children[k_s] = v_s.decode()
        for k, v in root.subtrees.items():
            children[k] = SettingNode(v.name, **SettingNodeModel._get_children(v))
        return children

    @classmethod
    def encode(cls, node: SettingNode) -> SettingNodeModel:
        """Creates pydantic `SettingNodeModel` out of :class:`exa.common.data.setting_node.SettingNode` instance"""
        return SettingNodeModel(**cls._get_dict(node))

    @classmethod
    def _get_dict(cls, root: SettingNode) -> Dict[str, Any]:
        return {
            "name": root.name,
            "settings": {key: SettingModel.encode(setting) for key, setting in root._settings.items()},
            "subtrees": {key: cls._get_dict(node) for key, node in root._subtrees.items()},
        }


# Since model references itself as a field type need to resolve ForwardRef during model creation
SettingNodeModel.model_rebuild()
