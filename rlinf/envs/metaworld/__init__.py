# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from rlinf.envs.metaworld.utils import load_prompt_from_json


class MetaWorldBenchmark:
    def __init__(self, task_suite_name):
        assert task_suite_name in [
            "metaworld_50",
            "metaworld_45_ind",
            "metaworld_45_ood",
            "metaworld_10",
        ]
        self.task_suite_name = task_suite_name
        config_path = os.path.join(os.path.dirname(__file__), "metaworld_config.json")
        self.task_description_dict = load_prompt_from_json(
            config_path, "TASK_DESCRIPTIONS"
        )
        self.ML45_dict = load_prompt_from_json(config_path, "ML45")
        # MT10 official task set from MetaWorld (v3 names).
        self.MT10_envs = [
            "reach-v3",
            "push-v3",
            "pick-place-v3",
            "door-open-v3",
            "door-close-v3",
            "drawer-open-v3",
            "drawer-close-v3",
            "button-press-topdown-v3",
            "peg-insert-side-v3",
            "window-open-v3",
        ]

    def get_num_tasks(self):
        if self.task_suite_name == "metaworld_50":
            return 50
        elif self.task_suite_name == "metaworld_45_ind":
            return 45
        elif self.task_suite_name == "metaworld_45_ood":
            return 5
        elif self.task_suite_name == "metaworld_10":
            return 10

    def get_task_num_trials(self):
        if self.task_suite_name == "metaworld_50":
            return 10
        elif self.task_suite_name == "metaworld_45_ind":
            return 10
        elif self.task_suite_name == "metaworld_45_ood":
            return 20
        elif self.task_suite_name == "metaworld_10":
            # Use the same number of trials as metaworld_50 for now.
            return 10

    def get_env_names(self):
        if self.task_suite_name == "metaworld_50":
            return list(self.task_description_dict.keys())
        elif self.task_suite_name == "metaworld_45_ind":
            return self.ML45_dict["train"]
        elif self.task_suite_name == "metaworld_45_ood":
            return self.ML45_dict["test"]
        elif self.task_suite_name == "metaworld_10":
            return self.MT10_envs

    def get_task_description(self):
        task_descriptions = []
        for env_name in self.get_env_names():
            task_descriptions.append(self.task_description_dict[env_name])
        return task_descriptions
