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

"""Task samplers for MetaWorld: decide which (task, trial) to sample next rollout.

Samplers are updated with per-episode (task_id, success) from env infos and
produce reset_state_ids for the next epoch.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseTaskSampler(ABC):
    """Base class for MetaWorld task samplers.

    A sampler maintains state from rollout episode outcomes and, given the
    number of group envs, returns reset_state_ids for the next rollout epoch.
    """

    @abstractmethod
    def update(self, epoch_env_stats: Optional[dict[str, np.ndarray]] = None) -> None:
        """Update sampler state from one rollout epoch's episode outcomes.

        Args:
            epoch_env_stats: Optional dict with at least "task_ids" and
                "successes" (both 1D numpy arrays, same length). Each element
                is one episode's task_id and success (0/1 or bool). If None,
                no update is performed (e.g. first epoch).
        """
        pass

    @abstractmethod
    def sample(
        self,
        num_group: int,
        *,
        rng: np.random.Generator,
        cumsum_trial_id_bins: np.ndarray,
        task_num_trials: list,
    ) -> np.ndarray:
        """Sample reset_state_ids for the next epoch.

        Args:
            num_group: Number of group envs (num_envs // group_size).
            rng: Random generator for reproducibility.
            cumsum_trial_id_bins: Cumulative sum of trial counts per task,
                shape (num_tasks,). Used to map (task_id, trial_id) to
                reset_state_id.
            task_num_trials: List of trial counts per task.

        Returns:
            reset_state_ids: int array of shape (num_group,) in [0,
                total_num_group_envs).
        """
        pass


def _task_trial_to_reset_state_id(
    task_id: int,
    trial_id: int,
    cumsum_trial_id_bins: np.ndarray,
) -> int:
    """Map (task_id, trial_id) to a single reset_state_id."""
    start = 0 if task_id == 0 else int(cumsum_trial_id_bins[task_id - 1])
    return start + trial_id


class SuccessRateAdaptiveSampler(BaseTaskSampler):
    """Success-rate adaptive sampling: p_t ∝ (1 - S_t)^α.

    Sampling probability for task t is proportional to (1 - S_t)^α, where
    S_t is the current success rate of task t. Lower success rate → higher
    sampling probability. α is a temperature (default 1.0).
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.0,
        min_count: float = 1.0,
    ):
        """Initialize the sampler.

        Args:
            num_tasks: Number of tasks (e.g. 50 for metaworld_50).
            alpha: Exponent in (1 - S_t)^α. Larger α emphasizes harder tasks.
            min_count: Pseudocount per task for success rate estimation
                (avoid S_t = 0 or 1 causing degenerate probs). Applied as
                S_t = (success_count + min_count/2) / (total_count + min_count).
        """
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.min_count = min_count
        self.success_counts = np.zeros(num_tasks, dtype=np.float64)
        self.total_counts = np.zeros(num_tasks, dtype=np.float64)

    def update(self, epoch_env_stats: Optional[dict[str, np.ndarray]] = None) -> None:
        if epoch_env_stats is None:
            return
        task_ids = epoch_env_stats.get("task_ids")
        successes = epoch_env_stats.get("successes")
        if task_ids is None or successes is None:
            return
        task_ids = np.asarray(task_ids).ravel()
        successes = np.asarray(successes).ravel()
        if len(task_ids) != len(successes):
            return
        for t, s in zip(task_ids, successes):
            if 0 <= t < self.num_tasks:
                self.total_counts[t] += 1
                self.success_counts[t] += float(s)

    def sample(
        self,
        num_group: int,
        *,
        rng: np.random.Generator,
        cumsum_trial_id_bins: np.ndarray,
        task_num_trials: list,
    ) -> np.ndarray:
        # Success rate with pseudocount: S_t = (success + min_count/2) / (total + min_count)
        total = self.total_counts + self.min_count
        success = self.success_counts + self.min_count / 2.0
        S = np.where(total > 0, success / total, 0.5)
        # p_t ∝ (1 - S_t)^α
        weights = np.power(np.clip(1.0 - S, 1e-8, 1.0), self.alpha)
        if weights.sum() <= 0:
            weights = np.ones(self.num_tasks) / self.num_tasks
        else:
            weights /= weights.sum()
        task_ids_sampled = rng.choice(
            self.num_tasks, size=num_group, replace=True, p=weights
        )
        reset_state_ids = np.zeros(num_group, dtype=np.int64)
        for i, task_id in enumerate(task_ids_sampled):
            n_trials = task_num_trials[task_id]
            trial_id = rng.integers(0, n_trials)
            reset_state_ids[i] = _task_trial_to_reset_state_id(
                task_id, trial_id, cumsum_trial_id_bins
            )
        return reset_state_ids
